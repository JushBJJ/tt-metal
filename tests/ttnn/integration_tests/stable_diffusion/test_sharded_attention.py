# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import math
import pytest

import tt_lib as ttl
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import comp_pcc, tt2torch_tensor, torch2tt_tensor, skip_for_wormhole_b0


# Test matmul attention sequence with InterleavedToShardedPartialOp
@pytest.mark.parametrize("seq_len", [4096])
@pytest.mark.parametrize("kv_len", [4096])
@pytest.mark.parametrize("num_slices", [16])
@pytest.mark.parametrize("num_cores", [64])
@pytest.mark.parametrize("num_heads", [16])
@pytest.mark.parametrize("data_format", [ttl.tensor.DataType.BFLOAT8_B])
def test_time_sharded_attnention(
    device,
    seq_len,
    kv_len,
    num_slices,
    num_cores,
    num_heads,
    data_format,
    function_level_defaults,
):
    compute_grid_size = device.compute_with_storage_grid_size()
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")
    grid_size = (8, 8)

    query_layer_shape = [1, num_heads, seq_len, 64]
    key_layer_transposed_shape = [1, num_heads, 64, kv_len]
    value_layer_shape = [1, num_heads, kv_len, 64]
    output_shape = [1, num_heads, seq_len, 64]

    torch_query_layer = torch.randn(query_layer_shape).bfloat16().float()
    torch_key_layer_transposed = torch.randn(key_layer_transposed_shape).bfloat16().float()
    torch_value_layer = torch.randn(value_layer_shape).bfloat16().float()
    torch_output = torch.randn(output_shape).bfloat16().float()

    dram_interleaved_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.DRAM,
    )

    height_sharded_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, buffer_type=ttl.tensor.BufferType.L1
    )

    # compare output to regular case
    reference_query_layer = torch2tt_tensor(
        torch_query_layer,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=data_format,
    )
    reference_key_layer_transposed = torch2tt_tensor(
        torch_key_layer_transposed,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=data_format,
    )
    reference_value_layer = torch2tt_tensor(
        torch_value_layer,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=data_format,
    )

    compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttl.tensor.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    passing = True
    output = None

    mm_out = torch2tt_tensor(
        torch_output,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=data_format,
    )
    tiles_per_shard = math.ceil((((num_heads * seq_len) / num_cores) / num_slices) / 32)
    mm_activations_height_shard_spec = [tiles_per_shard * 32, 2 * 32]
    mm_output_height_shard_spec = [tiles_per_shard * 32, seq_len]

    heads_per_slice = num_heads // num_slices
    for i in range(num_slices):
        slice = ttl.tensor.interleaved_to_sharded_partial(
            reference_query_layer,
            grid_size,
            mm_activations_height_shard_spec,
            num_slices,  # num_slices
            i,  # slice_index
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            ttl.tensor.ShardOrientation.ROW_MAJOR,
        )
        program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=2,
            per_core_M=tiles_per_shard,
            per_core_N=kv_len // 32,
            out_subblock_h=1,
            out_subblock_w=1,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        )

        k_slice = ttl.tensor.unpad(
            reference_key_layer_transposed,
            (0, (i * heads_per_slice), 0, 0),
            (0, (i * heads_per_slice) + (heads_per_slice - 1), 63, kv_len - 1),
            output_mem_config=dram_interleaved_memory_config,
        )
        mm_slice = ttl.operations.primary.matmul(
            slice,
            k_slice,
            program_config=program_config,
            output_mem_config=height_sharded_memory_config,
            output_dtype=data_format,
            compute_kernel_config=compute_kernel_config,
        )
        k_slice.deallocate()
        slice.deallocate()

        softmax_program_config = ttl.operations.primary.transformers.SoftmaxShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=grid_size,
            subblock_w=1,
            block_h=mm_output_height_shard_spec[0] // 32,
            block_w=mm_output_height_shard_spec[1] // 32,
            math_fidelity=ttl.tensor.MathFidelity.LoFi,
            im_data_format=ttl.tensor.DataType.BFLOAT16,
        )

        mm_slice = ttl.operations.primary.softmax_in_place(mm_slice, program_config=softmax_program_config)

        program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=kv_len // 32,
            per_core_M=tiles_per_shard,
            per_core_N=2,
            out_subblock_h=1,
            out_subblock_w=1,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        )
        v_slice = ttl.tensor.unpad(
            reference_value_layer,
            (0, (i * heads_per_slice), 0, 0),
            (0, (i * heads_per_slice) + (heads_per_slice - 1), kv_len - 1, 63),
            output_mem_config=dram_interleaved_memory_config,
        )
        mm_slice = ttl.operations.primary.matmul(
            mm_slice,
            v_slice,
            program_config=program_config,
            output_mem_config=height_sharded_memory_config,
            output_dtype=data_format,
            compute_kernel_config=compute_kernel_config,
        )
        v_slice.deallocate()

        ttl.tensor.sharded_to_interleaved_partial(
            mm_slice,
            mm_out,
            num_slices,
            i,
            dram_interleaved_memory_config,
        )

        mm_slice.deallocate()

    mm_out_torch = tt2torch_tensor(mm_out)

    attn_weights = ttl.tensor.bmm(
        reference_query_layer, reference_key_layer_transposed, output_mem_config=dram_interleaved_memory_config
    )
    attn_weights = ttl.operations.primary.softmax_in_place(attn_weights)
    attn_weights = ttl.tensor.bmm(attn_weights, reference_value_layer, output_mem_config=dram_interleaved_memory_config)

    attn_weights_torch = tt2torch_tensor(attn_weights)
    passing, output = comp_pcc(mm_out_torch, attn_weights_torch)

    print(output)
    assert passing


# Test matmul attention sequence with InterleavedToShardedPartialOp
@pytest.mark.parametrize("seq_len", [4096])
@pytest.mark.parametrize("kv_len", [96])
# @pytest.mark.parametrize("num_cores", [64])
@pytest.mark.parametrize("num_heads", [16])
@pytest.mark.parametrize("data_format", [ttl.tensor.DataType.BFLOAT8_B])
def test_attnention(
    device,
    seq_len,
    kv_len,
    # num_cores,
    num_heads,
    data_format,
    function_level_defaults,
):
    compute_grid_size = device.compute_with_storage_grid_size()
    grid_size = (8, 2)
    num_cores = grid_size[0] * grid_size[1]
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")

    query_layer_shape = [1, num_heads, seq_len, 64]
    key_layer_transposed_shape = [1, num_heads, 64, kv_len]
    value_layer_shape = [1, num_heads, kv_len, 64]
    output_shape = [1, num_heads, seq_len, 64]

    torch_query_layer = torch.randn(query_layer_shape).bfloat16().float()
    torch_key_layer_transposed = torch.randn(key_layer_transposed_shape).bfloat16().float()
    torch_value_layer = torch.randn(value_layer_shape).bfloat16().float()
    torch_output = torch.randn(output_shape).bfloat16().float()

    dram_interleaved_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.DRAM,
    )

    height_sharded_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, buffer_type=ttl.tensor.BufferType.L1
    )

    # compare output to regular case
    reference_query_layer = torch2tt_tensor(
        torch_query_layer,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=data_format,
    )
    reference_key_layer_transposed = torch2tt_tensor(
        torch_key_layer_transposed,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=data_format,
    )
    reference_value_layer = torch2tt_tensor(
        torch_value_layer,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=data_format,
    )

    compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttl.tensor.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    passing = True
    output = None

    q_sharded = ttl.tensor.interleaved_to_sharded(
        reference_query_layer,
        grid_size,
        [num_heads * seq_len // num_cores, 64],
        ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
        ttl.tensor.ShardOrientation.COL_MAJOR,
    )

    # k_sharded = ttl.tensor.interleaved_to_sharded(
    #     reference_key_layer_transposed,
    #     grid_size,
    #     [num_heads * 64 // num_cores, kv_len],
    #     ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
    #     ttl.tensor.ShardOrientation.COL_MAJOR,
    # )

    program_config = ttl.operations.primary.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=128,  # num_heads * seq_len // num_cores // 32,
        per_core_N=3,
    )

    compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttl.tensor.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    mm_slice = ttl.operations.primary.matmul(
        q_sharded,
        reference_key_layer_transposed,
        program_config=program_config,
        output_mem_config=height_sharded_memory_config,
        output_dtype=data_format,
        compute_kernel_config=compute_kernel_config,
    )
    q_sharded.deallocate()

    # breakpoint()
    # compute_grid = ttl.tensor.CoreCoord(8, 8)
    # output_shard_grid = ttl.tensor.CoreRangeSet({ttl.tensor.CoreRange(ttl.tensor.CoreCoord(0, 0), compute_grid)})
    # output_shard_spec = ttl.tensor.ShardSpec(output_shard_grid, [1024, 96], ttl.tensor.ShardOrientation.ROW_MAJOR, False)
    # output_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, ttl.tensor.BufferType.L1, output_shard_spec)
    # mm_slice = ttl.tensor.reshard(
    #     mm_slice,
    #     output_mem_config,
    # )
    softmax_program_config = ttl.operations.primary.transformers.SoftmaxShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=grid_size,
        subblock_w=1,
        block_h=128,
        block_w=3,
        math_fidelity=ttl.tensor.MathFidelity.LoFi,
        im_data_format=ttl.tensor.DataType.BFLOAT16,
    )

    mm_slice = ttl.operations.primary.softmax_in_place(mm_slice, program_config=softmax_program_config)

    v_sharded = ttl.tensor.interleaved_to_sharded(
        reference_value_layer,
        grid_size,
        [num_heads * kv_len // num_cores, 64],
        ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
        ttl.tensor.ShardOrientation.COL_MAJOR,
    )
    compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttl.tensor.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    program_config = ttl.operations.primary.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=3,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=128,  # num_heads * seq_len // num_cores // 32,
        per_core_N=2,
    )
    mm_slice = ttl.operations.primary.matmul(
        mm_slice,
        v_sharded,
        program_config=program_config,
        output_mem_config=height_sharded_memory_config,
        output_dtype=data_format,
        compute_kernel_config=compute_kernel_config,
    )
    v_sharded.deallocate()

    mm_out_torch = tt2torch_tensor(mm_slice)

    attn_weights = ttl.tensor.bmm(
        reference_query_layer, reference_key_layer_transposed, output_mem_config=dram_interleaved_memory_config
    )
    attn_weights = ttl.operations.primary.softmax_in_place(attn_weights)
    attn_weights = ttl.tensor.bmm(attn_weights, reference_value_layer, output_mem_config=dram_interleaved_memory_config)

    attn_weights_torch = tt2torch_tensor(attn_weights)
    passing, output = comp_pcc(mm_out_torch, attn_weights_torch)

    print(output)
    assert passing
