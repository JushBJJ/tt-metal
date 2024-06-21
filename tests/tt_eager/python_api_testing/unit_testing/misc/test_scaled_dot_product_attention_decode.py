# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import torch
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)
import tt_lib
import ttnn
from loguru import logger
import pytest
from models.utility_functions import skip_for_grayskull, skip_for_wormhole_b0
import math


def is_watcher_enabled():
    return os.environ.get("TT_METAL_WATCHER") is not None


def nearest_n(x, n):
    return ((x + n - 1) // n) * n


def nearest_pow_2(x):
    if x < 1:
        raise ValueError("x must be >= 1")
    import math

    power = math.ceil(math.log2(x))
    return 1 << power
    # if (2**math.log2(x) == x):
    #     return x
    # return 2**(int(x).bit_length())


def num_to_corerange(x):
    assert x < 8 or x % 8 == 0
    num_x = min(x, 8)
    num_y = x // num_x
    assert num_x * num_y == x
    return ttnn.experimental.tensor.CoreRange(
        ttnn.experimental.tensor.CoreCoord(0, 0),
        ttnn.experimental.tensor.CoreCoord(num_x - 1, num_y - 1),
    )


def get_chunk_size(s):
    # Got to test this!
    if s <= 32:
        return 32
    if s <= 64:
        return 32
    if s <= 128:
        return 32
    if s <= 256:
        return 256
    if s <= 2048:
        return 512
    return 512


def flash_attention_loop(q, K, V, mask, scale, k_chunk_size):
    seqlen = K.shape[-2]
    padded_num_heads = q.shape[-2]
    Tc = seqlen // k_chunk_size
    O = torch.zeros_like(q)
    l = torch.zeros([1, 1, padded_num_heads, 1])
    m = torch.ones([1, 1, padded_num_heads, 1]) * torch.finfo(torch.float32).min
    for t in range(Tc):
        K_chunk = K[:, :, t * k_chunk_size : (t + 1) * k_chunk_size, :]
        V_chunk = V[:, :, t * k_chunk_size : (t + 1) * k_chunk_size, :]
        mask_chunk = mask[:, :, :, t * k_chunk_size : (t + 1) * k_chunk_size]

        attn = torch.matmul(q, K_chunk.transpose(-2, -1)) * scale + mask_chunk
        m_old = m
        m = torch.max(m_old, torch.max(attn, dim=-1, keepdim=True)[0])
        P = torch.exp(attn - m)
        l = torch.exp(m_old - m) * l + torch.sum(P, dim=-1, keepdim=True)
        O = torch.matmul(P, V_chunk) + torch.matmul(torch.eye(padded_num_heads) * torch.exp(m_old - m), O)
    return O, m, l


def scaled_dot_product_attention_simulated(
    tt_Q,
    tt_K,
    tt_V,
    tt_attn_mask,
    is_causal,
    scale,
    program_config,
    valid_seq_len,
    compute_kernel_config,
    output_mem_config,
):
    # inputs
    tt_Q = ttnn.to_torch(tt_Q).to(torch.float32)
    tt_K = ttnn.to_torch(tt_K).to(torch.float32)
    tt_V = ttnn.to_torch(tt_V).to(torch.float32)
    tt_attn_mask = ttnn.to_torch(tt_attn_mask).to(torch.float32)

    # shapes
    k_chunk_size = program_config.k_chunk_size
    batch = tt_Q.shape[-3]
    head_dim = tt_Q.shape[-1]
    padded_num_heads = tt_Q.shape[-2]
    seqlen = tt_K.shape[-2]
    core_grid = program_config.compute_with_storage_grid_size
    num_cores = core_grid.x * core_grid.y

    # split to cores
    num_cores_per_batch = num_cores // batch
    num_active_cores = num_cores_per_batch * batch
    active_cores = [[i + k * num_cores_per_batch for i in range(num_cores_per_batch)] for k in range(batch)]

    # sequence length assignment
    assert valid_seq_len % k_chunk_size == 0
    num_chunks = valid_seq_len // k_chunk_size
    chunks_per_core = math.ceil(num_chunks // num_cores_per_batch)
    chunk_assignment = [[i * chunks_per_core, (i + 1) * chunks_per_core] for i in range(num_cores_per_batch)]
    chunk_assignment[-1][-1] += num_chunks % num_cores_per_batch

    # loop over batches
    output_tensor = torch.zeros_like(tt_Q)
    for b, batch_cores in enumerate(active_cores):
        O_intermed = []
        m_intermed = []
        l_intermed = []
        for i, core in enumerate(batch_cores):
            chunk_start, chunk_end = chunk_assignment[i]
            if chunk_start == chunk_end:
                continue
            O, m, l = flash_attention_loop(
                tt_Q[:, [b]],
                tt_K[:, [b], chunk_start * k_chunk_size : chunk_end * k_chunk_size, :],
                tt_V[:, [b], chunk_start * k_chunk_size : chunk_end * k_chunk_size, :],
                tt_attn_mask[:, [b], :, chunk_start * k_chunk_size : chunk_end * k_chunk_size],
                scale,
                k_chunk_size,
            )
            O_intermed.append(O)
            m_intermed.append(m)
            l_intermed.append(l)
        O, m, l = O_intermed[0], m_intermed[0], l_intermed[0]
        for O_2, m_2, l_2 in zip(O_intermed[1:], m_intermed[1:], l_intermed[1:]):
            O_1, m_1, l_1 = O, m, l
            m = torch.max(m_1, m_2)
            l = torch.exp(m_2 - m) * l_2 + torch.exp(m_1 - m) * l_1
            O = torch.matmul(torch.eye(padded_num_heads) * torch.exp(m_2 - m), O_2) + torch.matmul(
                torch.eye(padded_num_heads) * torch.exp(m_1 - m), O_1
            )
        output_tensor[:, b] = torch.matmul(torch.eye(padded_num_heads) * 1 / l, O)
    return output_tensor


def run_test_sdpa_decode(device, b, nh, nkv, s, d, dtype, grid_size, q_dtype=ttnn.bfloat16, mask_dtype=ttnn.bfloat16):
    padded_num_heads = nearest_pow_2(nearest_n(nh, n=32))
    torch.manual_seed(1234)

    min_pcc = 0.97 if dtype == tt_lib.tensor.DataType.BFLOAT4_B else 0.99

    compute_kernel_config = tt_lib.tensor.WormholeComputeKernelConfig(
        math_fidelity=tt_lib.tensor.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    dram_memcfg = ttnn.types.MemoryConfig(ttnn.types.TensorMemoryLayout.INTERLEAVED, ttnn.types.BufferType.DRAM)

    K = torch.randn(nkv, b, s, d)
    V = torch.randn(nkv, b, s, d)

    tt_K = ttnn.as_tensor(K, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram_memcfg)
    tt_V = ttnn.as_tensor(V, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram_memcfg)

    start_idx = 31

    while start_idx < s:
        scale = d**-0.5

        k_chunk_size = get_chunk_size(start_idx)
        program_config = tt_lib.operations.primary.transformers.SDPAMultiCoreProgramConfig(
            compute_with_storage_grid_size=grid_size,  # device.compute_with_storage_grid_size(),
            q_chunk_size=padded_num_heads,
            k_chunk_size=k_chunk_size,
        )

        padded_layer_len = nearest_n(start_idx, n=k_chunk_size)

        # Test various sequence lengths
        logger.info(f"Testing with sequence length: {start_idx}")
        logger.info(f"Using chunk size: {k_chunk_size}")
        logger.info(f"Using padded layer length: {padded_layer_len}")
        logger.info(f"Using padded num heads: {padded_num_heads}")

        attn_mask = torch.zeros((1, b, padded_num_heads, padded_layer_len))
        # Assume all users are at same position
        attn_mask[:, :, :, start_idx:] = torch.finfo(torch.float32).min

        Q = torch.randn(1, b, padded_num_heads, d)
        # Q = torch.eye(padded_num_heads, d).expand(1, b, padded_num_heads, d)
        # Q = torch.ones(1, b, padded_num_heads, d) * 1

        tt_Q = ttnn.as_tensor(
            Q, device=device, dtype=q_dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram_memcfg  # height_sharded_memcfg
        )
        # print(f"Q memcfg: {tt_Q.memory_config()}")

        tt_attn_mask = ttnn.as_tensor(
            attn_mask, device=device, dtype=mask_dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram_memcfg
        )

        # logger.info(f"Q shape: {Q.shape}")
        # logger.info(f"K shape: {K.shape}")
        # logger.info(f"V shape: {V.shape}")
        # logger.info(f"attn_mask shape: {attn_mask.shape}")

        tt_back = tt_lib.operations.primary.transformers.scaled_dot_product_attention_decode(
            tt_Q,
            tt_K,
            tt_V,
            tt_attn_mask,
            scale=scale,
            program_config=program_config,
            valid_seq_len=padded_layer_len,
            compute_kernel_config=compute_kernel_config,
            output_mem_config=dram_memcfg,  # height_sharded_memcfg,
        )

        tt_back = ttnn.to_torch(tt_back)

        tt_back = tt_back[:, :, :nh, :]

        Q_slice = Q[:, :, :nh, :].permute(1, 2, 0, 3)  # b, nh, 1, d
        K_slice = K[:, :, :padded_layer_len, :].permute(1, 0, 2, 3)  # nh, b, S, d
        V_slice = V[:, :, :padded_layer_len, :].permute(1, 0, 2, 3)  # nh, b, S, d
        attn_mask_slice = attn_mask[:, :, :nh, :].permute(1, 2, 0, 3)  # b, nh, 1, S

        expect = torch.nn.functional.scaled_dot_product_attention(
            Q_slice, K_slice, V_slice, attn_mask_slice, scale=scale, is_causal=False
        )  # b, nh, 1, d
        expect = expect.squeeze().unsqueeze(0)

        out_pass, out_pcc = comp_pcc(expect, tt_back, min_pcc)

        logger.debug(f"python vs pytorch: {out_pcc}")

        assert out_pass

        start_idx += 601 if start_idx < 4096 else 3001


@skip_for_grayskull("Unsupported in GS since L1 runs OOM with most configs")
@pytest.mark.parametrize(
    "dtype, q_dtype, mask_dtype",
    [
        # [tt_lib.tensor.DataType.BFLOAT8_B, tt_lib.tensor.DataType.BFLOAT8_B, tt_lib.tensor.DataType.BFLOAT8_B],
        [tt_lib.tensor.DataType.BFLOAT16, tt_lib.tensor.DataType.BFLOAT16, tt_lib.tensor.DataType.BFLOAT16],
        [tt_lib.tensor.DataType.BFLOAT8_B, tt_lib.tensor.DataType.BFLOAT16, tt_lib.tensor.DataType.BFLOAT8_B],
        [tt_lib.tensor.DataType.BFLOAT8_B, tt_lib.tensor.DataType.BFLOAT16, tt_lib.tensor.DataType.BFLOAT4_B],
        [tt_lib.tensor.DataType.BFLOAT4_B, tt_lib.tensor.DataType.BFLOAT16, tt_lib.tensor.DataType.BFLOAT4_B],
    ],
    ids=[
        # "all_bfp8",
        "all_bfp16",
        "kvmask_bfp8",
        "kv_bfp8_mask_bfp4",
        "kvmask_bfp4",
    ],
)
@pytest.mark.parametrize(
    "b, nh, nkv, s, d, grid_size",
    (
        # [32, 8, 1, 32768, 128, (8, 6)],  # Llama2-70B
        [16, 8, 1, 32768, 128, (8, 6)],  # Llama2-70B
        [8, 8, 1, 32768, 128, (8, 6)],  # Llama2-70B
        [4, 8, 1, 32768, 128, (8, 6)],  # Llama2-70B
        # [16, 8, 1, 32768, 128, (8,6)],  # Llama2-70B
        # [16, 8, 1, 32768, 128, (8,7)],  # Llama2-70B
        # [16, 8, 1, 32768, 128, (8,8)],  # Llama2-70B
        # [1, 8, 1, 2048, 128],  # Llama2-70B
        # [32, 16, 1, 2048, 64],  # Falcon-40B
        # [32, 71, 1, 2048, 64],  # Falcon-7B
        # [8, 8, 1, 2048, 128],  # Llama2-70B large batch
        # [1, 8, 1, 8192, 128],  # Llama2-70B large sequence
    ),
)
def test_sdpa_decode(device, b, nh, nkv, s, d, dtype, grid_size, q_dtype, mask_dtype):
    tt_lib.device.DisablePersistentKernelCache()
    run_test_sdpa_decode(device, b, nh, nkv, s, d, dtype, grid_size, q_dtype, mask_dtype)


def run_test_sdpa_decode_single_iter(device, b, nh, nkv, s, d, dtype):
    padded_num_heads = nearest_pow_2(nearest_n(nh, n=32))
    torch.manual_seed(1234)

    compute_kernel_config = tt_lib.tensor.WormholeComputeKernelConfig(
        math_fidelity=tt_lib.tensor.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    dram_memcfg = ttnn.types.MemoryConfig(ttnn.types.TensorMemoryLayout.INTERLEAVED, ttnn.types.BufferType.DRAM)

    K = torch.randn(nkv, b, s, d)
    V = torch.randn(nkv, b, s, d)

    tt_K = ttnn.as_tensor(K, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram_memcfg)
    tt_V = ttnn.as_tensor(V, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram_memcfg)

    start_idx = s // 2
    scale = d**-0.5

    k_chunk_size = get_chunk_size(start_idx)
    program_config = tt_lib.operations.primary.transformers.SDPAMultiCoreProgramConfig(
        compute_with_storage_grid_size=(8, 6),
        q_chunk_size=padded_num_heads,
        k_chunk_size=k_chunk_size,
    )

    padded_layer_len = nearest_n(start_idx, n=k_chunk_size)

    # Test various sequence lengths
    logger.debug(f"Testing with sequence length: {start_idx}")
    logger.debug(f"Using chunk size: {k_chunk_size}")
    logger.debug(f"Using padded layer length: {padded_layer_len}")
    logger.debug(f"Using padded num heads: {padded_num_heads}")

    attn_mask = torch.zeros((1, b, padded_num_heads, padded_layer_len))
    # Assume all users are at same position
    attn_mask[:, :, :, start_idx:] = torch.finfo(torch.float32).min

    Q = torch.randn(1, b, padded_num_heads, d)

    tt_Q = ttnn.as_tensor(Q, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram_memcfg)

    tt_attn_mask = ttnn.as_tensor(
        attn_mask, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram_memcfg
    )

    tt_back = tt_lib.operations.primary.transformers.scaled_dot_product_attention_decode(
        tt_Q,
        tt_K,
        tt_V,
        tt_attn_mask,
        scale=scale,
        program_config=program_config,
        valid_seq_len=padded_layer_len,
        compute_kernel_config=compute_kernel_config,
        output_mem_config=dram_memcfg,
    )

    tt_back = ttnn.to_torch(tt_back)
    tt_back = tt_back[:, :, :nh, :]

    Q_slice = Q[:, :, :nh, :].permute(1, 2, 0, 3)  # b, nh, 1, d
    K_slice = K[:, :, :padded_layer_len, :].permute(1, 0, 2, 3)  # nh, b, S, d
    V_slice = V[:, :, :padded_layer_len, :].permute(1, 0, 2, 3)  # nh, b, S, d
    attn_mask_slice = attn_mask[:, :, :nh, :].permute(1, 2, 0, 3)  # b, nh, 1, S
    expect = torch.nn.functional.scaled_dot_product_attention(
        Q_slice, K_slice, V_slice, attn_mask_slice, scale=scale, is_causal=False
    )  # b, nh, 1, d
    expect = expect.squeeze().unsqueeze(0)

    out_pass, out_pcc = comp_pcc(expect, tt_back, 0.99)

    logger.debug(f"python vs pytorch: {out_pcc}")
    assert out_pass


@skip_for_grayskull("Unsupported in GS since L1 runs OOM with most configs")
@pytest.mark.parametrize(
    "dtype",
    [tt_lib.tensor.DataType.BFLOAT8_B, tt_lib.tensor.DataType.BFLOAT16],
    ids=["bfp8", "bf16"],
)
@pytest.mark.parametrize(
    "b, nh, nkv, s, d",
    ([16, 8, 1, 8192, 128],),  # Llama2-70B
)
def test_sdpa_decode_program_cache(device, b, nh, nkv, s, d, dtype, use_program_cache):
    tt_lib.device.DisablePersistentKernelCache()

    dummy_tensors = []
    for _ in range(2):
        dummy_tensors.append(
            ttnn.as_tensor(
                torch.zeros(32, 32),
                device=device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.types.MemoryConfig(
                    ttnn.types.TensorMemoryLayout.INTERLEAVED, ttnn.types.BufferType.DRAM
                ),
            )
        )
        dummy_tensors.append(
            ttnn.as_tensor(
                torch.zeros(1, 1, 32, 32 * 32),
                device=device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.types.MemoryConfig(
                    ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED,
                    ttnn.types.BufferType.L1,
                    ttnn.experimental.tensor.ShardSpec(
                        ttnn.experimental.tensor.CoreRangeSet({num_to_corerange(32)}),
                        (32, 32),
                        ttnn.experimental.tensor.ShardOrientation.ROW_MAJOR,
                        False,
                    ),
                ),
            )
        )
        run_test_sdpa_decode_single_iter(device, b, nh, nkv, s, d, dtype)

    assert device.num_program_cache_entries() == 1
