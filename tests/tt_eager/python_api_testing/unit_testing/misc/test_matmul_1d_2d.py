# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import tt_lib as ttl
from models.utility_functions import is_wormhole_b0, is_grayskull, skip_for_wormhole_b0
from models.utility_functions import torch2tt_tensor, tt2torch_tensor, pad_by_zero, roundup32
import torch
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_equal,
    comp_pcc,
)


def find_max_subblock(out_block_h, out_block_w):
    max_product = 0
    best_h = 1
    best_w = 1

    for h in range(1, out_block_h + 1):
        if out_block_h % h == 0:  # h is a divisor of out_block_h
            for w in range(1, out_block_w + 1):
                if out_block_w % w == 0 and h * w <= 8:  # w is a divisor and product condition met
                    if h * w > max_product:
                        max_product = h * w
                        best_h = h
                        best_w = w
    if out_block_w > best_w:
        best_h = 1
    return best_h, best_w, max_product


from models.utility_functions import is_wormhole_b0, is_grayskull, skip_for_wormhole_b0


@skip_for_wormhole_b0()
@pytest.mark.skipif(is_grayskull(), reason="no llama2 test on GS")
@pytest.mark.parametrize(
    "packer_l1_acc",
    [
        True,
    ],
    ids=["pack_l1"],
)
@pytest.mark.parametrize(
    "fp32_acc_mode",
    [
        False,
    ],
    ids=["no_fp32"],
)
@pytest.mark.parametrize(
    "fidelity",
    [
        ttl.tensor.MathFidelity.LoFi,
    ],
    ids=["LoFi"],
)
@pytest.mark.parametrize(
    "has_bias",
    [
        False,
    ],
    ids=["no_bias"],
)
@pytest.mark.parametrize(
    "in1_in_dram, out_sharded, in0_sharded, M, K, N, activation, grid_size",
    [
        (False, True, True, 32, 8192, 1280, None, (8, 1)),
        (False, True, True, 32, 8192, 4096, None, (8, 4)),
        (False, True, True, 32, 8192, 1024, None, (8, 4)),
        (False, True, True, 32, 32768, 1024, None, (8, 4)),
    ],
)
def test_llama2_matmul(
    device,
    in0_sharded,
    out_sharded,
    in1_in_dram,
    M,
    K,
    N,
    fidelity,
    has_bias,
    activation,
    packer_l1_acc,
    fp32_acc_mode,
    grid_size,
    function_level_defaults,
):
    in0_shape = [1, 1, M, K]
    in1_shape = [1, 1, K, N]
    bias_shape = [1, 1, N]
    num_cores = grid_size[0] * grid_size[1]

    in0_block_h = M // 32
    in0_block_w = K // num_cores // 32
    out_block_h = M // 32
    out_block_w = N // num_cores // 32

    out_subblock_h, out_subblock_w, _ = find_max_subblock(out_block_h, out_block_w)

    logger.debug("in0 block h w " + str(in0_block_h * 32) + " " + str(in0_block_w * 32))
    logger.debug("in1 block h w " + str(in0_block_w * 32) + " " + str(out_block_w * 32))
    logger.debug("out block h w " + str(out_block_h * 32) + " " + str(out_block_w * 32))
    logger.debug("out subblock h w " + str(out_subblock_h * 32) + " " + str(out_subblock_w * 32))

    interleaved_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.DRAM,
    )
    sharded_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()
    bias = torch.randn(bias_shape).bfloat16().float()

    output_mem_config = sharded_mem_config

    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config, tt_dtype=ttl.tensor.DataType.BFLOAT16)
    in1_t = torch2tt_tensor(
        in1, device, tt_memory_config=interleaved_mem_config, tt_dtype=ttl.tensor.DataType.BFLOAT8_B
    )

    if in0_sharded:
        in0_t = ttl.tensor.interleaved_to_sharded(
            in0_t,
            grid_size,
            [M, int(in0_block_w * 32)],
            ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED,
            ttl.tensor.ShardOrientation.ROW_MAJOR,
        )

    program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=out_block_h,
        per_core_N=out_block_w,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )

    compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=fidelity,
        math_approx_mode=True,
        fp32_dest_acc_en=fp32_acc_mode,
        packer_l1_acc=packer_l1_acc,
    )

    output_t = ttl.operations.primary.matmul_1d(
        in0_t,
        in1_t,
        program_config=program_config,
        output_mem_config=output_mem_config,
        output_dtype=ttl.tensor.DataType.BFLOAT8_B,
        compute_kernel_config=compute_kernel_config,
    )
    if out_sharded:
        output_t = ttl.tensor.sharded_to_interleaved(output_t, interleaved_mem_config)
    pt_out = in0 @ in1 + bias

    tt_out = tt2torch_tensor(output_t)

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing


@skip_for_wormhole_b0()
@pytest.mark.skipif(is_grayskull(), reason="GS does not support fp32")
@pytest.mark.parametrize("has_bias", [False], ids=["no_bias"])
@pytest.mark.parametrize(
    "in1_in_dram, out_sharded, in0_sharded, M, K, N, activation, dtype, fidelity, packer_l1_acc, fp32_acc_mode",
    [
        # 256 256 256
        (
            False,
            True,
            True,
            1792,
            2048,
            4096,
            None,
            ttl.tensor.DataType.BFLOAT8_B,
            ttl.tensor.MathFidelity.LoFi,
            False,
            False,
        ),
        (
            False,
            True,
            True,
            1792,
            2048,
            4096,
            None,
            ttl.tensor.DataType.BFLOAT8_B,
            ttl.tensor.MathFidelity.HiFi2,
            False,
            False,
        ),
        (
            False,
            True,
            True,
            1792,
            2048,
            4096,
            None,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.MathFidelity.LoFi,
            False,
            False,
        ),
        (
            False,
            True,
            True,
            1792,
            2048,
            4096,
            None,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.MathFidelity.HiFi2,
            False,
            False,
        ),
        (
            False,
            True,
            True,
            1792,
            2048,
            4096,
            None,
            ttl.tensor.DataType.BFLOAT8_B,
            ttl.tensor.MathFidelity.LoFi,
            True,
            False,
        ),
        (
            False,
            True,
            True,
            1792,
            2048,
            4096,
            None,
            ttl.tensor.DataType.BFLOAT8_B,
            ttl.tensor.MathFidelity.HiFi2,
            True,
            False,
        ),
        (
            False,
            True,
            True,
            1792,
            2048,
            4096,
            None,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.MathFidelity.LoFi,
            True,
            False,
        ),
        (
            False,
            True,
            True,
            1792,
            2048,
            4096,
            None,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.MathFidelity.HiFi2,
            True,
            False,
        ),
        # 512 512 512 x 8 subblock 4 2
        (
            False,
            True,
            True,
            1792,
            2048,
            2048,
            None,
            ttl.tensor.DataType.BFLOAT8_B,
            ttl.tensor.MathFidelity.LoFi,
            False,
            True,
        ),
        (
            False,
            True,
            True,
            1792,
            2048,
            2048,
            None,
            ttl.tensor.DataType.BFLOAT8_B,
            ttl.tensor.MathFidelity.HiFi2,
            False,
            True,
        ),
        (
            False,
            True,
            True,
            1792,
            2048,
            2048,
            None,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.MathFidelity.LoFi,
            False,
            True,
        ),
        (
            False,
            True,
            True,
            1792,
            2048,
            2048,
            None,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.MathFidelity.HiFi2,
            False,
            True,
        ),
        (
            False,
            True,
            True,
            1792,
            2048,
            2048,
            None,
            ttl.tensor.DataType.BFLOAT8_B,
            ttl.tensor.MathFidelity.LoFi,
            True,
            True,
        ),
        (
            False,
            True,
            True,
            1792,
            2048,
            2048,
            None,
            ttl.tensor.DataType.BFLOAT8_B,
            ttl.tensor.MathFidelity.HiFi2,
            True,
            True,
        ),
        (
            False,
            True,
            True,
            1792,
            2048,
            2048,
            None,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.MathFidelity.LoFi,
            True,
            True,
        ),
        (
            False,
            True,
            True,
            1792,
            2048,
            2048,
            None,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.MathFidelity.HiFi2,
            True,
            True,
        ),
    ],
)
def test_multi_core_matmul_2d_wh(
    device,
    dtype,
    fidelity,
    in0_sharded,
    out_sharded,
    in1_in_dram,
    has_bias,
    fp32_acc_mode,
    packer_l1_acc,
    M,
    K,
    N,
    activation,
    function_level_defaults,
):
    in0_shape = [1, 1, M, K]
    in1_shape = [1, 1, K, N]
    bias_shape = [1, 1, N]
    grid_size = (8, 7)

    in0_block_h = M // grid_size[1] // 32
    in0_block_w = K // grid_size[0] // 32
    out_block_h = M // grid_size[1] // 32
    out_block_w = N // grid_size[0] // 32

    if fp32_acc_mode == True:
        out_subblock_w = 4
        out_subblock_h = 1
    else:
        out_subblock_w = 8
        out_subblock_h = 1

    logger.debug("in0 block h w " + str(in0_block_h * 32) + " " + str(in0_block_w * 32))
    logger.debug("in1 block h w " + str(in0_block_w * 32) + " " + str(out_block_w * 32))
    logger.debug("out block h w " + str(out_block_h * 32) + " " + str(out_block_w * 32))
    logger.debug("out subblock h w " + str(out_subblock_h * 32) + " " + str(out_subblock_w * 32))

    interleaved_mem_config_L1 = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.L1,
    )
    interleaved_mem_config_DRAM = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.DRAM,
    )
    sharded_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()
    bias = torch.randn(bias_shape).bfloat16().float()

    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config_DRAM, tt_dtype=dtype)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=interleaved_mem_config_DRAM, tt_dtype=dtype)

    output_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config_L1

    if in0_sharded:
        in0_t = ttl.tensor.interleaved_to_sharded(
            in0_t,
            grid_size,
            [M // grid_size[1], K // grid_size[0]],
            ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
            ttl.tensor.ShardOrientation.ROW_MAJOR,
        )

    program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=out_block_h,
        per_core_N=out_block_w,
        transpose_mcast=False,
        fused_activation=activation,
    )

    compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=fidelity,
        math_approx_mode=True,
        fp32_dest_acc_en=fp32_acc_mode,
        packer_l1_acc=packer_l1_acc,
    )

    output_t = ttl.operations.primary.matmul(
        in0_t,
        in1_t,
        program_config=program_config,
        output_mem_config=output_mem_config,
        compute_kernel_config=compute_kernel_config,
    )

    if out_sharded:
        output_t = ttl.tensor.sharded_to_interleaved(output_t, interleaved_mem_config_L1)

    pt_out = in0 @ in1

    if has_bias:
        pt_out = pt_out + bias

    if activation != None:
        pt_out = torch.nn.functional.gelu(pt_out)
    tt_out = tt2torch_tensor(output_t)

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing


@skip_for_wormhole_b0()
@pytest.mark.skipif(is_grayskull(), reason="GS does not support fp32")
@pytest.mark.parametrize("has_bias", [False], ids=["no_bias"])
@pytest.mark.parametrize(
    "in1_in_dram, out_sharded, in0_sharded, M, K, N, activation, dtype, fidelity, packer_l1_acc, fp32_acc_mode",
    [
        # 512, 8192, 8192
        (
            False,
            True,
            True,
            512,
            8192,
            8192,
            None,
            ttl.tensor.DataType.BFLOAT8_B,
            ttl.tensor.MathFidelity.LoFi,
            False,
            False,
        ),
        (
            False,
            True,
            True,
            512,
            8192,
            8192,
            None,
            ttl.tensor.DataType.BFLOAT8_B,
            ttl.tensor.MathFidelity.HiFi2,
            False,
            False,
        ),
        (
            False,
            True,
            True,
            512,
            8192,
            8192,
            None,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.MathFidelity.LoFi,
            False,
            False,
        ),
        (
            False,
            True,
            True,
            512,
            8192,
            8192,
            None,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.MathFidelity.HiFi2,
            False,
            False,
        ),
        (
            False,
            True,
            True,
            512,
            8192,
            8192,
            None,
            ttl.tensor.DataType.BFLOAT8_B,
            ttl.tensor.MathFidelity.LoFi,
            True,
            False,
        ),
        (
            False,
            True,
            True,
            512,
            8192,
            8192,
            None,
            ttl.tensor.DataType.BFLOAT8_B,
            ttl.tensor.MathFidelity.HiFi2,
            True,
            False,
        ),
        (
            False,
            True,
            True,
            512,
            8192,
            8192,
            None,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.MathFidelity.LoFi,
            True,
            False,
        ),
        (
            False,
            True,
            True,
            512,
            8192,
            8192,
            None,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.MathFidelity.HiFi2,
            True,
            False,
        ),
        # 256, 8192, 8192
        (
            False,
            True,
            True,
            256,
            8192,
            8192,
            None,
            ttl.tensor.DataType.BFLOAT8_B,
            ttl.tensor.MathFidelity.LoFi,
            False,
            True,
        ),
        (
            False,
            True,
            True,
            256,
            8192,
            8192,
            None,
            ttl.tensor.DataType.BFLOAT8_B,
            ttl.tensor.MathFidelity.HiFi2,
            False,
            True,
        ),
        (
            False,
            True,
            True,
            256,
            8192,
            8192,
            None,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.MathFidelity.LoFi,
            False,
            True,
        ),
        (
            False,
            True,
            True,
            256,
            8192,
            8192,
            None,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.MathFidelity.HiFi2,
            False,
            True,
        ),
        (
            False,
            True,
            True,
            256,
            8192,
            8192,
            None,
            ttl.tensor.DataType.BFLOAT8_B,
            ttl.tensor.MathFidelity.LoFi,
            True,
            True,
        ),
        (
            False,
            True,
            True,
            256,
            8192,
            8192,
            None,
            ttl.tensor.DataType.BFLOAT8_B,
            ttl.tensor.MathFidelity.HiFi2,
            True,
            True,
        ),
        # (
        #     False,
        #     True,
        #     True,
        #     256,
        #     8192,
        #     8192,
        #     None,
        #     ttl.tensor.DataType.BFLOAT16,
        #     ttl.tensor.MathFidelity.LoFi,
        #     True,
        #     True,
        # ),
        # (
        #     False,
        #     True,
        #     True,
        #     256,
        #     8192,
        #     8192,
        #     None,
        #     ttl.tensor.DataType.BFLOAT16,
        #     ttl.tensor.MathFidelity.HiFi2,
        #     True,
        #     True,
        # ),
    ],
)
def test_multi_core_matmul_1d_wh(
    device,
    dtype,
    fidelity,
    in0_sharded,
    out_sharded,
    in1_in_dram,
    has_bias,
    fp32_acc_mode,
    packer_l1_acc,
    M,
    K,
    N,
    activation,
    function_level_defaults,
):
    in0_shape = [1, 1, M, K]
    in1_shape = [1, 1, K, N]
    bias_shape = [1, 1, N]
    grid_size = (8, 4)
    num_cores = grid_size[0] * grid_size[1]

    in0_block_h = M // 32
    in0_block_w = K // num_cores // 32
    out_block_h = M // 32
    out_block_w = N // num_cores // 32

    if fp32_acc_mode == True:
        out_subblock_w = 4
        out_subblock_h = 1
    else:
        out_subblock_w = 8
        out_subblock_h = 1

    logger.debug("in0 block h w " + str(in0_block_h * 32) + " " + str(in0_block_w * 32))
    logger.debug("in1 block h w " + str(in0_block_w * 32) + " " + str(out_block_w * 32))
    logger.debug("out block h w " + str(out_block_h * 32) + " " + str(out_block_w * 32))
    logger.debug("out subblock h w " + str(out_subblock_h * 32) + " " + str(out_subblock_w * 32))

    interleaved_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.DRAM,
    )
    sharded_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()
    bias = torch.randn(bias_shape).bfloat16().float()

    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config, tt_dtype=dtype)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=interleaved_mem_config, tt_dtype=dtype)

    output_mem_config = sharded_mem_config

    if in0_sharded:
        in0_t = ttl.tensor.interleaved_to_sharded(
            in0_t,
            grid_size,
            [M, int(out_block_w * 32)],
            ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED,
            ttl.tensor.ShardOrientation.ROW_MAJOR,
        )

    program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=out_block_h,
        per_core_N=out_block_w,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )

    compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=fidelity,
        math_approx_mode=True,
        fp32_dest_acc_en=fp32_acc_mode,
        packer_l1_acc=packer_l1_acc,
    )

    output_t = ttl.operations.primary.matmul_1d(
        in0_t,
        in1_t,
        program_config=program_config,
        output_mem_config=output_mem_config,
        output_dtype=dtype,
        compute_kernel_config=compute_kernel_config,
    )
    if out_sharded:
        output_t = ttl.tensor.sharded_to_interleaved(output_t, interleaved_mem_config)
    pt_out = in0 @ in1 + bias

    tt_out = tt2torch_tensor(output_t)

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing


@skip_for_wormhole_b0()
@pytest.mark.parametrize("has_bias", [False], ids=["no_bias"])
@pytest.mark.parametrize(
    "in1_in_dram, out_sharded, in0_sharded, M, K, N, activation, dtype, fidelity",
    [
        # 256 256 256
        (
            False,
            True,
            True,
            3072,
            2048,
            4096,
            None,
            ttl.tensor.DataType.BFLOAT8_B,
            ttl.tensor.MathFidelity.LoFi,
        ),
        (
            False,
            True,
            True,
            3072,
            2048,
            4096,
            None,
            ttl.tensor.DataType.BFLOAT8_B,
            ttl.tensor.MathFidelity.HiFi2,
        ),
        (
            False,
            True,
            True,
            3072,
            2048,
            4096,
            None,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.MathFidelity.LoFi,
        ),
        (
            False,
            True,
            True,
            3072,
            2048,
            4096,
            None,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.MathFidelity.HiFi2,
        ),
        # 512 512 512 x 8 subblock 4 2
        (
            False,
            True,
            True,
            3072,
            2048,
            2048,
            None,
            ttl.tensor.DataType.BFLOAT8_B,
            ttl.tensor.MathFidelity.LoFi,
        ),
        (
            False,
            True,
            True,
            3072,
            2048,
            2048,
            None,
            ttl.tensor.DataType.BFLOAT8_B,
            ttl.tensor.MathFidelity.HiFi2,
        ),
        (
            False,
            True,
            True,
            3072,
            2048,
            2048,
            None,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.MathFidelity.LoFi,
        ),
        (
            False,
            True,
            True,
            3072,
            2048,
            2048,
            None,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.MathFidelity.HiFi2,
        ),
    ],
)
def test_multi_core_matmul_2d_gs(
    device,
    dtype,
    fidelity,
    in0_sharded,
    out_sharded,
    in1_in_dram,
    has_bias,
    M,
    K,
    N,
    activation,
    function_level_defaults,
):
    in0_shape = [1, 1, M, K]
    in1_shape = [1, 1, K, N]
    bias_shape = [1, 1, N]
    grid_size = (12, 8)

    in0_block_w = K // grid_size[1] // 32  # 16
    in0_block_h = M // grid_size[0] // 32
    out_block_h = M // grid_size[0] // 32
    out_block_w = N // grid_size[1] // 32

    if out_block_w <= 8:
        out_subblock_w = out_block_w
        out_subblock_h = 8 // out_subblock_w
    else:
        out_subblock_h = 1
        out_subblock_w = 8 // out_subblock_h
        while out_block_w % out_subblock_w != 0:
            out_subblock_w = out_block_w // 2

    logger.debug("in0 block w h " + str(in0_block_w * 32) + " " + str(in0_block_h * 32))
    logger.debug("in1 block w h " + str(out_block_w * 32) + " " + str(in0_block_w * 32))
    logger.debug("out block w h " + str(out_block_w * 32) + " " + str(out_block_h * 32))
    logger.debug("out subblock w h " + str(out_subblock_w * 32) + " " + str(out_subblock_h * 32))

    interleaved_mem_config_L1 = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.L1,
    )
    interleaved_mem_config_DRAM = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.DRAM,
    )
    sharded_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()
    bias = torch.randn(bias_shape).bfloat16().float()

    in0_t = torch2tt_tensor(
        in0, device, tt_memory_config=interleaved_mem_config_DRAM, tt_dtype=ttl.tensor.DataType.BFLOAT8_B
    )

    in1_t = torch2tt_tensor(
        in1, device, tt_memory_config=interleaved_mem_config_DRAM, tt_dtype=ttl.tensor.DataType.BFLOAT8_B
    )

    output_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config_L1
    bias_t = pad_by_zero(
        bias, device, tt_memory_config=interleaved_mem_config_L1, tt_dtype=ttl.tensor.DataType.BFLOAT8_B
    )[0]

    if in0_sharded:
        in0_t = ttl.tensor.interleaved_to_sharded(
            in0_t,
            grid_size,
            [M // grid_size[0], K // grid_size[1]],
            ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
            ttl.tensor.ShardOrientation.COL_MAJOR,
        )

    program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=out_block_h,
        per_core_N=out_block_w,
        transpose_mcast=True,
        fused_activation=activation,
    )

    compute_kernel_config = ttl.tensor.GrayskullComputeKernelConfig(math_fidelity=fidelity, math_approx_mode=True)

    if has_bias:
        output_t = ttl.operations.primary.matmul(
            in0_t,
            in1_t,
            bias=bias_t,
            program_config=program_config,
            output_mem_config=output_mem_config,
            compute_kernel_config=compute_kernel_config,
        )
    else:
        output_t = ttl.operations.primary.matmul(
            in0_t,
            in1_t,
            program_config=program_config,
            output_mem_config=output_mem_config,
            compute_kernel_config=compute_kernel_config,
        )

    if out_sharded:
        output_t = ttl.tensor.sharded_to_interleaved(output_t, interleaved_mem_config_L1)

    pt_out = in0 @ in1

    if has_bias:
        pt_out = pt_out + bias

    if activation != None:
        pt_out = torch.nn.functional.gelu(pt_out)
    tt_out = tt2torch_tensor(output_t)

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing


@skip_for_wormhole_b0()
@pytest.mark.parametrize("has_bias", [False], ids=["no_bias"])
@pytest.mark.parametrize(
    "in1_in_dram, out_sharded, in0_sharded, M, K, N, activation, dtype, fidelity",
    [
        # 256, 8192, 8192
        (
            False,
            True,
            True,
            256,
            8192,
            8192,
            None,
            ttl.tensor.DataType.BFLOAT8_B,
            ttl.tensor.MathFidelity.LoFi,
        ),
        (
            False,
            True,
            True,
            256,
            8192,
            8192,
            None,
            ttl.tensor.DataType.BFLOAT8_B,
            ttl.tensor.MathFidelity.HiFi2,
        ),
        (
            False,
            True,
            True,
            256,
            8192,
            8192,
            None,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.MathFidelity.LoFi,
        ),
        (
            False,
            True,
            True,
            256,
            8192,
            8192,
            None,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.MathFidelity.HiFi2,
        ),
    ],
)
def test_multi_core_matmul_1d_gs(
    device,
    dtype,
    fidelity,
    in0_sharded,
    out_sharded,
    in1_in_dram,
    has_bias,
    M,
    K,
    N,
    activation,
    function_level_defaults,
):
    in0_shape = [1, 1, M, K]
    in1_shape = [1, 1, K, N]
    bias_shape = [1, 1, N]
    grid_size = (8, 4)
    num_cores = grid_size[0] * grid_size[1]

    in0_block_h = M // 32
    in0_block_w = K // num_cores // 32
    out_block_h = M // 32
    out_block_w = N // num_cores // 32

    out_subblock_w = 8
    out_subblock_h = 1

    logger.debug("in0 block h w " + str(in0_block_h * 32) + " " + str(in0_block_w * 32))
    logger.debug("in1 block h w " + str(in0_block_w * 32) + " " + str(out_block_w * 32))
    logger.debug("out block h w " + str(out_block_h * 32) + " " + str(out_block_w * 32))
    logger.debug("out subblock h w " + str(out_subblock_h * 32) + " " + str(out_subblock_w * 32))

    interleaved_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.DRAM,
    )
    sharded_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()
    bias = torch.randn(bias_shape).bfloat16().float()

    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config, tt_dtype=dtype)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=interleaved_mem_config, tt_dtype=dtype)

    output_mem_config = sharded_mem_config

    if in0_sharded:
        in0_t = ttl.tensor.interleaved_to_sharded(
            in0_t,
            grid_size,
            [M, int(out_block_w * 32)],
            ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED,
            ttl.tensor.ShardOrientation.ROW_MAJOR,
        )

    program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=out_block_h,
        per_core_N=out_block_w,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )

    compute_kernel_config = ttl.tensor.GrayskullComputeKernelConfig(
        math_fidelity=fidelity,
        math_approx_mode=True,
    )

    output_t = ttl.operations.primary.matmul_1d(
        in0_t,
        in1_t,
        program_config=program_config,
        output_mem_config=output_mem_config,
        output_dtype=dtype,
        compute_kernel_config=compute_kernel_config,
    )
    if out_sharded:
        output_t = ttl.tensor.sharded_to_interleaved(output_t, interleaved_mem_config)
    pt_out = in0 @ in1 + bias

    tt_out = tt2torch_tensor(output_t)

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing


def run_bert_linear_batch8(
    device,
    in0_sharded,
    out_sharded,
    in1_in_dram,
    M,
    K,
    N,
    fidelity,
    has_bias,
    activation,
    packer_l1_acc,
    fp32_acc_mode,
    enable_opt,
    function_level_defaults,
):
    in0_shape = [1, 1, M, K]
    in1_shape = [1, 1, K, N]
    bias_shape = [1, 1, N]
    grid_size = (4, 8)

    in0_block_h = M // grid_size[0] // 32
    in0_block_w = K // grid_size[1] // 32
    out_block_h = M // grid_size[0] // 32
    out_block_w = N // grid_size[1] // 32

    out_subblock_h, out_subblock_w, _ = find_max_subblock(out_block_h, out_block_w)

    logger.debug("in0 block w h " + str(in0_block_w) + " " + str(in0_block_h))
    logger.debug("in1 block w h " + str(out_block_w) + " " + str(in0_block_w))
    logger.debug("out block w h " + str(out_block_w) + " " + str(out_block_h))
    logger.debug("out subblock w h " + str(out_subblock_w) + " " + str(out_subblock_h))

    interleaved_mem_config_L1 = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.L1,
    )
    interleaved_mem_config_DRAM = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.DRAM,
    )
    sharded_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()
    bias = torch.randn(bias_shape).bfloat16().float()

    in0_t = torch2tt_tensor(
        # in0, device, tt_memory_config=interleaved_mem_config_DRAM, tt_dtype=ttl.tensor.DataType.BFLOAT8_B
        in0,
        device,
        tt_memory_config=interleaved_mem_config_DRAM,
        tt_dtype=ttl.tensor.DataType.BFLOAT8_B,
    )
    in1_t = torch2tt_tensor(
        # in1, device, tt_memory_config=interleaved_mem_config_DRAM, tt_dtype=ttl.tensor.DataType.BFLOAT8_B
        in1,
        device,
        tt_memory_config=interleaved_mem_config_DRAM,
        tt_dtype=ttl.tensor.DataType.BFLOAT8_B,
    )

    output_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config_L1
    bias_t = pad_by_zero(
        bias, device, tt_memory_config=interleaved_mem_config_DRAM, tt_dtype=ttl.tensor.DataType.BFLOAT8_B
    )[0]

    if in0_sharded:
        in0_t = ttl.tensor.interleaved_to_sharded(
            in0_t,
            grid_size,
            [M // grid_size[0], K // grid_size[1]],
            ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
            ttl.tensor.ShardOrientation.COL_MAJOR,
        )

    program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=out_block_h,
        per_core_N=out_block_w,
        transpose_mcast=True,
        fused_activation=activation,
        # mcast_use_same_noc=True,
        use_noc_vc=False,
    )

    compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=fidelity,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    if has_bias:
        output_t = ttl.operations.primary.matmul(
            in0_t,
            in1_t,
            bias=bias_t,
            program_config=program_config,
            output_mem_config=output_mem_config,
            compute_kernel_config=compute_kernel_config,
        )
    else:
        output_t = ttl.operations.primary.matmul(
            in0_t,
            in1_t,
            program_config=program_config,
            output_mem_config=output_mem_config,
            compute_kernel_config=compute_kernel_config,
        )

    if out_sharded:
        output_t = ttl.tensor.sharded_to_interleaved(output_t, interleaved_mem_config_L1)

    pt_out = in0 @ in1

    if has_bias:
        pt_out = pt_out + bias

    if activation != None:
        pt_out = torch.nn.functional.gelu(pt_out)
    tt_out = tt2torch_tensor(output_t)

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing


@skip_for_wormhole_b0()
@pytest.mark.skipif(is_grayskull(), reason="no need to test on GS")
@pytest.mark.parametrize(
    "enable_opt",
    [
        True,
    ],
    ids=["enable_opt"],
)
@pytest.mark.parametrize(
    "packer_l1_acc",
    [
        True,
    ],
    ids=["pack_l1"],
)
@pytest.mark.parametrize(
    "fp32_acc_mode",
    [
        False,
    ],
    ids=["no_fp32"],
)
@pytest.mark.parametrize(
    "fidelity",
    [
        ttl.tensor.MathFidelity.LoFi,
    ],
    ids=["LoFi"],
)
@pytest.mark.parametrize(
    "has_bias",
    [
        False,
    ],
    ids=["no_bias"],
)
@pytest.mark.parametrize(
    "in1_in_dram, out_sharded, in0_sharded, M, K, N, activation",
    [
        (False, True, True, 128, 1280, 2560, None),
        # (False, True, True, 256, 256, 256, None),
        # (False, True, True, 256, 256, 512, None),
        # (False, True, True, 256, 256, 1024, None),
        # (False, True, True, 256, 256, 2048, None),
        # (False, True, True, 256, 256, 4096, None),
        # (False, True, True, 256, 256, 8192, None),
        # (False, True, True, 256, 512, 256, None),
        # (False, True, True, 256, 512, 512, None),
        # (False, True, True, 256, 512, 1024, None),
        # (False, True, True, 256, 512, 2048, None),
        # (False, True, True, 256, 512, 4096, None),
        # (False, True, True, 256, 512, 8192, None),
        # (False, True, True, 256, 1024, 256, None),
        # (False, True, True, 256, 1024, 512, None),
        # (False, True, True, 256, 1024, 1024, None),
        # (False, True, True, 256, 1024, 2048, None),
        # (False, True, True, 256, 1024, 4096, None),
        # (False, True, True, 256, 1024, 8192, None),
        # (False, True, True, 256, 2048, 256, None),
        # (False, True, True, 256, 2048, 512, None),
        # (False, True, True, 256, 2048, 1024, None),
        # (False, True, True, 256, 2048, 2048, None),
        # (False, True, True, 256, 2048, 4096, None),
        # (False, True, True, 256, 2048, 8192, None),
        # (False, True, True, 256, 4096, 256, None),
        # (False, True, True, 256, 4096, 512, None),
        # (False, True, True, 256, 4096, 1024, None),
        # (False, True, True, 256, 4096, 2048, None),
        # (False, True, True, 256, 4096, 4096, None),
        # (False, True, True, 256, 4096, 8192, None),
        # (False, True, True, 256, 8192, 256, None),
        # (False, True, True, 256, 8192, 512, None),
        # (False, True, True, 256, 8192, 1024, None),
        # (False, True, True, 256, 8192, 2048, None),
        # (False, True, True, 256, 8192, 4096, None),
        # (False, True, True, 512, 256, 256, None),
        # (False, True, True, 512, 256, 512, None),
        # (False, True, True, 512, 256, 1024, None),
        # (False, True, True, 512, 256, 2048, None),
        # (False, True, True, 512, 256, 4096, None),
        # (False, True, True, 512, 256, 8192, None),
        # (False, True, True, 512, 512, 256, None),
        # (False, True, True, 512, 512, 512, None),
        # (False, True, True, 512, 512, 1024, None),
        # (False, True, True, 512, 512, 2048, None),
        # (False, True, True, 512, 512, 4096, None),
        # (False, True, True, 512, 512, 8192, None),
        # (False, True, True, 512, 1024, 256, None),
        # (False, True, True, 512, 1024, 512, None),
        # (False, True, True, 512, 1024, 1024, None),
        # (False, True, True, 512, 1024, 2048, None),
        # (False, True, True, 512, 1024, 4096, None),
        # (False, True, True, 512, 1024, 8192, None),
        # (False, True, True, 512, 2048, 256, None),
        # (False, True, True, 512, 2048, 512, None),
        # (False, True, True, 512, 2048, 1024, None),
        # (False, True, True, 512, 2048, 2048, None),
        # (False, True, True, 512, 2048, 4096, None),
        # (False, True, True, 512, 2048, 8192, None),
        # (False, True, True, 512, 4096, 256, None),
        # (False, True, True, 512, 4096, 512, None),
        # (False, True, True, 512, 4096, 1024, None),
        # (False, True, True, 512, 4096, 2048, None),
        # (False, True, True, 512, 4096, 4096, None),
        # (False, True, True, 512, 8192, 256, None),
        # (False, True, True, 512, 8192, 512, None),
        # (False, True, True, 512, 8192, 1024, None),
        # (False, True, True, 512, 8192, 2048, None),
        # (False, True, True, 1024, 256, 256, None),
        # (False, True, True, 1024, 256, 512, None),
        # (False, True, True, 1024, 256, 1024, None),
        # (False, True, True, 1024, 256, 2048, None),
        # (False, True, True, 1024, 256, 4096, None),
        # (False, True, True, 1024, 256, 8192, None),
        # (False, True, True, 1024, 512, 256, None),
        # (False, True, True, 1024, 512, 512, None),
        # (False, True, True, 1024, 512, 1024, None),
        # (False, True, True, 1024, 512, 2048, None),
        # (False, True, True, 1024, 512, 4096, None),
        # (False, True, True, 1024, 512, 8192, None),
        # (False, True, True, 1024, 1024, 256, None),
        # (False, True, True, 1024, 1024, 512, None),
        # (False, True, True, 1024, 1024, 1024, None),
        # (False, True, True, 1024, 1024, 2048, None),
        # (False, True, True, 1024, 1024, 4096, None),
        # (False, True, True, 1024, 1024, 8192, None),
        # (False, True, True, 1024, 2048, 256, None),
        # (False, True, True, 1024, 2048, 512, None),
        # (False, True, True, 1024, 2048, 1024, None),
        # (False, True, True, 1024, 2048, 2048, None),
        # (False, True, True, 1024, 2048, 4096, None),
        # (False, True, True, 1024, 2048, 8192, None),
        # (False, True, True, 1024, 4096, 256, None),
        # (False, True, True, 1024, 4096, 512, None),
        # (False, True, True, 1024, 4096, 1024, None),
        # (False, True, True, 1024, 4096, 2048, None),
        # (False, True, True, 1024, 4096, 4096, None),
        # (False, True, True, 1024, 8192, 256, None),
        # (False, True, True, 1024, 8192, 512, None),
        # (False, True, True, 1024, 8192, 1024, None),
        # (False, True, True, 1024, 8192, 2048, None),
        # (False, True, True, 256, 4096, 4096, None),
        # (False, True, True, 512, 1280, 5120, None),
        # (False, True, True, 512, 5120, 1280, None),
        # (False, True, True, 512, 1280, 1280, None),
        # (False, True, True, 512, 1280, 3840, None),
        # (False, True, True, 512, 2560, 1280, None),
    ],
)
def test_matmul_opt_threshold(
    device,
    in0_sharded,
    out_sharded,
    in1_in_dram,
    M,
    K,
    N,
    fidelity,
    has_bias,
    activation,
    packer_l1_acc,
    fp32_acc_mode,
    enable_opt,
    function_level_defaults,
):
    for i in range(1):
        logger.info(i)
        run_bert_linear_batch8(
            device,
            in0_sharded,
            out_sharded,
            in1_in_dram,
            M,
            K,
            N,
            fidelity,
            has_bias,
            activation,
            packer_l1_acc,
            fp32_acc_mode,
            enable_opt,
            function_level_defaults,
        )
