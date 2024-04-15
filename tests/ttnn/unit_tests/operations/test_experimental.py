# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from models.utility_functions import skip_for_wormhole_b0, torch_random, is_wormhole_b0
from tests.ttnn.utils_for_testing import assert_with_pcc


@skip_for_wormhole_b0()
@pytest.mark.parametrize("height", [32])
@pytest.mark.parametrize("width", [32])
def test_ttnn_experimental_tensor_exp(device, height, width):
    torch.manual_seed(0)

    torch_input_tensor = torch_random((1, 1, height, width), -1, 1, dtype=torch.bfloat16)
    torch_output_tensor = ttnn.experimental.tensor.exp.golden_function(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, device=device)
    output_tensor = ttnn.experimental.tensor.exp(input_tensor)

    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("m_size", [32])
@pytest.mark.parametrize("k_size", [32])
@pytest.mark.parametrize("n_size", [32])
def test_ttnn_experimental_operations_primary_matmul(device, m_size, k_size, n_size):
    torch.manual_seed(0)

    torch_input_tensor_a = torch_random((1, 1, m_size, k_size), -1, 1, dtype=torch.bfloat16)
    torch_input_tensor_b = torch_random((1, 1, k_size, n_size), -1, 1, dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a @ torch_input_tensor_b

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, device=device, layout=ttnn.TILE_LAYOUT)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, device=device, layout=ttnn.TILE_LAYOUT)
    output_tensor = ttnn.experimental.operations.primary.matmul(input_tensor_a, input_tensor_b)

    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("m_size", [32])
@pytest.mark.parametrize("k_size", [32])
@pytest.mark.parametrize("n_size", [32])
def test_ttnn_experimental_operations_primary_moreh_matmul(device, m_size, k_size, n_size):
    torch.manual_seed(0)

    torch_input_tensor_a = torch_random((1, 1, m_size, k_size), -1, 1, dtype=torch.bfloat16)
    torch_input_tensor_b = torch_random((1, 1, k_size, n_size), -1, 1, dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a @ torch_input_tensor_b

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, device=device, layout=ttnn.TILE_LAYOUT)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, device=device, layout=ttnn.TILE_LAYOUT)
    output_tensor = ttnn.experimental.operations.primary.moreh_matmul(input_tensor_a, input_tensor_b)

    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("input_a_is_sharded", [True, False])
@pytest.mark.parametrize("output_is_sharded", [True, False])
@pytest.mark.parametrize("m_size, num_cores", [[25088, 98]])
@pytest.mark.parametrize("k_size, n_size", [[64, 64], [64, 256]])
@pytest.mark.parametrize("input_a_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("input_b_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_ttnn_experimental_operations_primary_matmul_1d(
    device, input_a_is_sharded, output_is_sharded, m_size, k_size, n_size, num_cores, input_a_dtype, input_b_dtype
):
    grid_size = device.compute_with_storage_grid_size()
    compute_grid_size = device.compute_with_storage_grid_size()
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")
    if input_a_dtype != input_b_dtype and is_wormhole_b0():
        pytest.skip("WH does not work with mixed precision")

    input_shape_a = [1, 1, m_size, k_size]
    input_shape_b = [1, 1, k_size, n_size]
    bias_shape = [1, 1, 1, n_size]

    interleaved_memory_config = ttnn.experimental.tensor.MemoryConfig(
        memory_layout=ttnn.experimental.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.experimental.tensor.BufferType.DRAM,
    )
    sharded_memory_config = ttnn.experimental.tensor.MemoryConfig(
        memory_layout=ttnn.experimental.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttnn.experimental.tensor.BufferType.L1,
    )

    output_memory_config = sharded_memory_config if output_is_sharded else interleaved_memory_config

    program_config = ttnn.experimental.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(12, 9),
        in0_block_w=k_size // 32,
        out_subblock_h=8 // (n_size // 32),
        out_subblock_w=n_size // 32,
        per_core_M=m_size // 32 // num_cores,
        per_core_N=n_size // 32,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
    )

    with ttnn.tracer.trace():
        torch_input_tensor_a = torch.randn(input_shape_a).bfloat16().float()
        torch_input_tensor_b = torch.randn(input_shape_b).bfloat16().float()
        torch_bias = torch.randn(bias_shape).bfloat16().float()
        torch_output_tensor = torch_input_tensor_a @ torch_input_tensor_b + torch_bias

        input_tensor_a = ttnn.from_torch(
            torch_input_tensor_a,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=interleaved_memory_config,
            dtype=input_a_dtype,
        )
        input_tensor_b = ttnn.from_torch(
            torch_input_tensor_b,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=interleaved_memory_config,
            dtype=input_b_dtype,
        )

        bias = ttnn.from_torch(
            torch_bias,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=interleaved_memory_config,
            dtype=input_b_dtype,
        )
        if input_a_is_sharded:
            input_tensor_a = ttnn.experimental.tensor.interleaved_to_sharded(
                input_tensor_a,
                grid_size,
                [m_size // num_cores, k_size],
                ttnn.experimental.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.experimental.tensor.ShardOrientation.ROW_MAJOR,
            )

        output_tensor = ttnn.experimental.operations.primary.matmul_1d(
            input_tensor_a,
            input_tensor_b,
            bias=bias,
            program_config=program_config,
            output_mem_config=output_memory_config,
            output_dtype=input_a_dtype,
        )
        if output_is_sharded:
            output_tensor = ttnn.experimental.tensor.sharded_to_interleaved(output_tensor, interleaved_memory_config)

        output_tensor = ttnn.to_torch(output_tensor)
        ttnn.tracer.visualize(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9996)


@pytest.mark.parametrize("H, num_cores", [[64, 64]])
@pytest.mark.parametrize("num_slices", [2])
def test_sharded_partial_op(
    device,
    H,
    num_cores,
    num_slices,
):
    compute_grid_size = device.compute_with_storage_grid_size()
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")
    grid_size = (8, 8)
    in0_shape = [1, 1, H, 64]
    W = in0_shape[-1]

    interleaved_mem_config = ttnn.experimental.tensor.MemoryConfig(
        memory_layout=ttnn.experimental.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.experimental.tensor.BufferType.L1,
    )
    sharded_mem_config = ttnn.experimental.tensor.MemoryConfig(
        memory_layout=ttnn.experimental.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttnn.experimental.tensor.BufferType.L1,
    )

    in0 = torch.ones(in0_shape).bfloat16().float()
    out_initial = torch.randn(in0_shape).bfloat16().float()

    in0_t = ttnn.from_torch(
        in0,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=interleaved_mem_config,
    )
    out_tt_tensor = ttnn.from_torch(
        out_initial,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=interleaved_mem_config,
    )

    height_shard_spec = [H // 2, W]

    for slice_index in range(num_slices):
        in0_t_slice = ttnn.experimental.tensor.interleaved_to_sharded_partial(
            in0_t,
            grid_size,
            height_shard_spec,
            num_slices,
            slice_index,
            ttnn.experimental.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.experimental.tensor.ShardOrientation.ROW_MAJOR,
        )
        assert in0_t_slice.is_sharded()

        ttnn.experimental.tensor.sharded_to_interleaved_partial(
            in0_t_slice,
            out_tt_tensor,
            num_slices,
            slice_index,
            interleaved_mem_config,
        )

    pt_out = in0

    tt_out = ttnn.to_torch(out_tt_tensor)

    assert_with_pcc(pt_out, tt_out)
