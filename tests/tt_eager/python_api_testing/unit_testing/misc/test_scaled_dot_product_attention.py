# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import math
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
    # Not sure if optimal
    if s <= 32:
        return 32
    if s <= 64:
        return 64
    if s <= 128:
        return 128
    if s <= 256:
        return 256
    if s <= 2048:
        return 512
    return 512


def run_test_sdpa_decode(device, b, nh, nkv, s, d, dtype):
    padded_num_heads = nearest_pow_2(nearest_n(nh, n=32))
    torch.manual_seed(1234)

    compute_kernel_config = tt_lib.tensor.WormholeComputeKernelConfig(
        math_fidelity=tt_lib.tensor.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    dram_memcfg = ttnn.types.MemoryConfig(ttnn.types.TensorMemoryLayout.INTERLEAVED, ttnn.types.BufferType.DRAM)
    shard_grid = ttnn.experimental.tensor.CoreRangeSet({num_to_corerange(b)})
    shard_spec = ttnn.experimental.tensor.ShardSpec(
        shard_grid, (padded_num_heads, d), ttnn.experimental.tensor.ShardOrientation.ROW_MAJOR, False
    )

    height_sharded_memcfg = ttnn.types.MemoryConfig(
        ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )

    K = torch.randn(nkv, b, s, d)
    V = torch.randn(nkv, b, s, d)

    tt_K = ttnn.as_tensor(K, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram_memcfg)
    tt_V = ttnn.as_tensor(V, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram_memcfg)

    start_idx = 31

    while start_idx < s:
        Q = torch.randn(1, b, padded_num_heads, d)
        pcc_list = []

        for i in range(4096):
            scale = d**-0.5

            k_chunk_size = get_chunk_size(start_idx)
            program_config = tt_lib.operations.primary.transformers.SDPAMultiCoreProgramConfig(
                compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
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

            tt_Q = ttnn.as_tensor(
                Q, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=height_sharded_memcfg
            )

            tt_attn_mask = ttnn.as_tensor(
                attn_mask, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram_memcfg
            )

            tt_back = tt_lib.operations.primary.transformers.scaled_dot_product_attention(
                tt_Q,
                tt_K,
                tt_V,
                tt_attn_mask,
                is_causal=False,
                scale=scale,
                program_config=program_config,
                valid_seq_len=padded_layer_len,
                compute_kernel_config=compute_kernel_config,
                output_mem_config=height_sharded_memcfg,
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
            pcc_list.append(out_pcc)

        logger.info(f"pcc_list: {pcc_list}")
        # make sure all pccs are the same for testing nd pcc issue
        assert all([pcc == pcc_list[0] for pcc in pcc_list])

        start_idx += 601 if start_idx < 4096 else 3001


@skip_for_grayskull("Unsupported in GS since L1 runs OOM with most configs")
@pytest.mark.parametrize(
    "dtype",
    [tt_lib.tensor.DataType.BFLOAT8_B],
    ids=["bfp8"],
)
@pytest.mark.parametrize(
    "b, nh, nkv, s, d",
    (
        [1, 8, 1, 8192 * 4, 128],  # Llama2-70B
        # [32, 16, 1, 2048, 64],  # Falcon-40B
        # [32, 4, 1, 8192, 128],  # Mixtral
    ),
)
def test_sdpa_decode(device, b, nh, nkv, s, d, dtype):
    tt_lib.device.DisablePersistentKernelCache()
    run_test_sdpa_decode(device, b, nh, nkv, s, d, dtype)
