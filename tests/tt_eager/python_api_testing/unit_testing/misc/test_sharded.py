# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import math

import tt_lib as ttl
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_equal,
    comp_pcc,
)
from models.utility_functions import is_wormhole_b0, skip_for_wormhole_b0
from loguru import logger
from models.utility_functions import torch2tt_tensor, tt2torch_tensor, pad_by_zero, roundup32


@pytest.mark.parametrize(
    "input_shape, shard_scheme, shard_size",
    [
        ([1, 1, 100352, 64], ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, (1024, 64)),
        ([1, 1, 128, 50176], ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED, (128, 512)),
        ([1, 1, 100352, 64], ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED, (2048, 32)),
    ],
)
@pytest.mark.parametrize(
    "shard_orientation", [ttl.tensor.ShardOrientation.ROW_MAJOR, ttl.tensor.ShardOrientation.COL_MAJOR]
)
@pytest.mark.parametrize("input_dtype", [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B])
@pytest.mark.parametrize("output_dtype", [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B])
def test_sharded_tile(
    device, input_shape, shard_size, shard_scheme, shard_orientation, input_dtype, output_dtype, function_level_defaults
):
    grid_size = device.compute_with_storage_grid_size()
    input_size = torch.Size(input_shape)
    num_cores = 98
    compute_grid_size = device.compute_with_storage_grid_size()
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")

    x = torch.arange(input_size.numel()).reshape(input_size).bfloat16().float()

    xt = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            input_dtype,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(
            device,
            ttl.tensor.MemoryConfig(
                memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
                buffer_type=ttl.tensor.BufferType.L1,
            ),
        )
    )

    yt = ttl.tensor.interleaved_to_sharded(
        xt, grid_size, shard_size, shard_scheme, shard_orientation, output_dtype=output_dtype
    )

    zt = ttl.tensor.sharded_to_interleaved(
        yt,
        ttl.tensor.MemoryConfig(
            memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttl.tensor.BufferType.L1,
        ),
        output_dtype=input_dtype,
    )

    tt_og = xt.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    tt_got_back = zt.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    if input_dtype == output_dtype:
        passing, output = comp_equal(tt_og, tt_got_back)
    else:
        passing, output = comp_pcc(tt_og, tt_got_back, 0.999)
    logger.info(output)

    assert passing


@pytest.mark.parametrize(
    "input_shape, shard_scheme, shard_size",
    [
        ([1, 1, 100352, 64], ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, (1024, 64)),
        ([1, 1, 128, 50176], ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED, (128, 512)),
        ([1, 1, 100352, 64], ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED, (2048, 32)),
    ],
)
@pytest.mark.parametrize(
    "shard_orientation",
    [ttl.tensor.ShardOrientation.ROW_MAJOR, ttl.tensor.ShardOrientation.COL_MAJOR],
)
def test_sharded_rm(device, input_shape, shard_size, shard_scheme, shard_orientation, function_level_defaults):
    grid_size = device.compute_with_storage_grid_size()
    input_size = torch.Size(input_shape)
    num_cores = 98
    compute_grid_size = device.compute_with_storage_grid_size()
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")
    x = torch.arange(input_size.numel()).reshape(input_size).bfloat16().float()

    xt = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    ).to(
        device,
        ttl.tensor.MemoryConfig(
            memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttl.tensor.BufferType.L1,
        ),
    )

    yt = ttl.tensor.interleaved_to_sharded(xt, grid_size, shard_size, shard_scheme, shard_orientation)

    zt = ttl.tensor.sharded_to_interleaved(
        yt,
        ttl.tensor.MemoryConfig(
            memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttl.tensor.BufferType.L1,
        ),
    )

    tt_og = xt.cpu().to_torch()

    tt_got_back = zt.cpu().to_torch()

    passing, output = comp_equal(tt_og, tt_got_back)
    logger.info(output)

    assert passing


@pytest.mark.parametrize("H, num_cores", [[100352, 98], [25088, 98]])
@pytest.mark.parametrize("in_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
@pytest.mark.parametrize("dtype", [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B])
def test_sharded_untilize(H, num_cores, in_sharded, out_sharded, dtype, device, function_level_defaults):
    grid_size = device.compute_with_storage_grid_size()
    compute_grid_size = device.compute_with_storage_grid_size()
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")

    N = 1
    C = 1
    W = 64
    if out_sharded and not in_sharded and H == 100352:
        pytest.skip("Unsupported config for sharding")

    interleaved_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.L1,
    )
    sharded_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    out_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config

    x = torch.randn((N, C, H, W)).bfloat16()

    xt = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            dtype,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(
            device,
            interleaved_mem_config,
        )
    )

    if in_sharded:
        xt = ttl.tensor.interleaved_to_sharded(
            xt,
            grid_size,
            [H // num_cores, W],
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            ttl.tensor.ShardOrientation.ROW_MAJOR,
        )

    yt = ttl.tensor.untilize(
        xt,
        output_mem_config=out_mem_config,
        use_multicore=True,
    )

    if out_sharded:
        yt = ttl.tensor.sharded_to_interleaved(
            yt,
            interleaved_mem_config,
        )

    tt_got_back = yt.cpu().to_torch()

    if dtype == ttl.tensor.DataType.BFLOAT16:
        passing, output = comp_equal(x, tt_got_back)
    else:
        passing, output = comp_pcc(x, tt_got_back, 0.999)
    logger.info(output)

    assert passing


@pytest.mark.parametrize("H, num_cores", [[25088, 98]])
@pytest.mark.parametrize("output_dtype", [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B])
def test_sharded_tilize(H, num_cores, output_dtype, device, function_level_defaults):
    grid_size = device.compute_with_storage_grid_size()
    compute_grid_size = device.compute_with_storage_grid_size()
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")

    N = 1
    C = 1
    W = 64

    x = torch.arange(N * C * H * W).reshape((N, C, H, W)).bfloat16()

    xt = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    ).to(
        device,
        ttl.tensor.MemoryConfig(
            memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttl.tensor.BufferType.L1,
        ),
    )

    yt = ttl.tensor.interleaved_to_sharded(
        xt,
        grid_size,
        [H // num_cores, W],
        ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
        ttl.tensor.ShardOrientation.ROW_MAJOR,
    )

    yt_tilized = ttl.tensor.tilize(
        yt,
        output_mem_config=ttl.tensor.MemoryConfig(
            memory_layout=ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            buffer_type=ttl.tensor.BufferType.L1,
        ),
        use_multicore=True,
        output_dtype=output_dtype,
    )

    zt = ttl.tensor.sharded_to_interleaved(
        yt_tilized,
        ttl.tensor.MemoryConfig(
            memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttl.tensor.BufferType.L1,
        ),
    )

    tt_got_back = zt.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    if output_dtype == ttl.tensor.DataType.BFLOAT16:
        passing, output = comp_equal(x, tt_got_back)
    else:
        passing, output = comp_pcc(x, tt_got_back, 0.999)
    logger.info(output)

    assert passing


@pytest.mark.parametrize("M", [127 * 32])
@pytest.mark.parametrize("K", [1 * 32])
@pytest.mark.parametrize("N", [1 * 32])
@pytest.mark.parametrize("num_cores", [64])
def test_height_sharded_matmul_1d_padding(device, M, K, N, num_cores):
    grid_size = device.compute_with_storage_grid_size()
    if num_cores > (grid_size.x * grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {grid_size}")
    in0_shape = [1, 1, M, K]
    in1_shape = [1, 1, K, N]
    height_shard_spec = [2 * 32, 32]  # [2, 1] in tiles

    interleaved_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.DRAM,
    )

    sharded_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()

    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config, tt_dtype=ttl.tensor.DataType.BFLOAT16)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=interleaved_mem_config, tt_dtype=ttl.tensor.DataType.BFLOAT16)

    in0_t = ttl.tensor.interleaved_to_sharded(
        in0_t,
        grid_size,
        height_shard_spec,
        ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
        ttl.tensor.ShardOrientation.ROW_MAJOR,
    )

    program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=K // 32,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=2,
        per_core_N=1,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
    )

    output_t = ttl.operations.primary.matmul_1d(
        in0_t,
        in1_t,
        bias=None,
        program_config=program_config,
        output_mem_config=sharded_mem_config,
        output_dtype=ttl.tensor.DataType.BFLOAT16,
    )

    output_t = ttl.tensor.sharded_to_interleaved(output_t, interleaved_mem_config)

    pt_out = in0 @ in1
    tt_out = tt2torch_tensor(output_t)
    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing


@skip_for_wormhole_b0("WH ND hang, see issue #4392")
@pytest.mark.parametrize("in0_sharded", [True, False], ids=["in0_sharded", "in0_unsharded"])
@pytest.mark.parametrize("out_sharded", [True, False], ids=["out_sharded", "out_unsharded"])
@pytest.mark.parametrize("M, num_cores", [[25088, 98], [50176, 98]])
@pytest.mark.parametrize("K, N", [[64, 64], [64, 256], [256, 64], [256, 128]])
@pytest.mark.parametrize("activations_dtype", [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B])
@pytest.mark.parametrize("weights_dtype", [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B])
def test_sharded_matmul_1d_in1(
    device, in0_sharded, out_sharded, M, K, N, num_cores, activations_dtype, weights_dtype, function_level_defaults
):
    grid_size = device.compute_with_storage_grid_size()
    compute_grid_size = device.compute_with_storage_grid_size()
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")
    if activations_dtype != weights_dtype and is_wormhole_b0():
        pytest.skip("WH does not work with mixed precision")
    in0_shape = [1, 1, M, K]
    in1_shape = [1, 1, K, N]
    bias_shape = [1, 1, 1, N]

    interleaved_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.DRAM,
    )
    sharded_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()
    bias = torch.randn(bias_shape).bfloat16().float()

    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=interleaved_mem_config, tt_dtype=weights_dtype)
    bias_t = pad_by_zero(bias, device, tt_memory_config=interleaved_mem_config, tt_dtype=weights_dtype)[0]

    output_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config

    if in0_sharded:
        in0_t = ttl.tensor.interleaved_to_sharded(
            in0_t,
            grid_size,
            [M // num_cores, K],
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            ttl.tensor.ShardOrientation.ROW_MAJOR,
        )

    program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(12, 9),
        in0_block_w=K // 32,
        out_subblock_h=8 // (N // 32),
        out_subblock_w=N // 32,
        per_core_M=M // 32 // num_cores,
        per_core_N=N // 32,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
    )
    output_t = ttl.operations.primary.matmul_1d(
        in0_t,
        in1_t,
        bias=bias_t,
        program_config=program_config,
        output_mem_config=output_mem_config,
        output_dtype=activations_dtype,
    )
    if out_sharded:
        output_t = ttl.tensor.sharded_to_interleaved(output_t, interleaved_mem_config)
    pt_out = in0 @ in1 + bias

    tt_out = tt2torch_tensor(output_t)

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing


@pytest.mark.parametrize("H, num_cores", [[64, 64]])
@pytest.mark.parametrize(
    "activations_dtype",
    [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B],
    ids=["inputs_BFLOAT16", "inputs_BFLOAT8_B"],
)
@pytest.mark.parametrize(
    "output_dtype", [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B], ids=["out_BFLOAT16", "out_BFLOAT8_B"]
)
def test_sharded_partial_spill_reload(
    device,
    H,
    num_cores,
    activations_dtype,
    output_dtype,
    function_level_defaults,
):
    grid_size = device.compute_with_storage_grid_size()
    compute_grid_size = device.compute_with_storage_grid_size()
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")
    in0_shape = [1, 1, H, 64]
    W = in0_shape[-1]

    interleaved_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.L1,
    )
    sharded_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    in0 = torch.ones(in0_shape).bfloat16().float()
    out_initial = torch.randn(in0_shape).bfloat16().float()
    mid_index = in0_shape[2] // 2  # H dimension
    in0[:, :, mid_index:] = 2

    # torch.set_printoptions(profile="full", sci_mode=False, linewidth=220)

    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype)
    out_tt_tensor = torch2tt_tensor(
        out_initial, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype
    )

    height_shard_spec = [H // 2, W]
    num_slices = 2

    for slice_index in range(num_slices):
        print("Running slice: ", slice_index)
        in0_t_slice = ttl.tensor.interleaved_to_sharded_partial(
            in0_t,
            grid_size,
            height_shard_spec,
            ttl.tensor.ShardedOpSplitDim.DimRow,
            num_slices,  # num_slices
            slice_index,  # slice_index
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            ttl.tensor.ShardOrientation.ROW_MAJOR,
        )

        ttl.tensor.sharded_to_interleaved_partial(
            in0_t_slice,
            out_tt_tensor,
            ttl.tensor.ShardedOpSplitDim.DimRow,
            num_slices,
            slice_index,
            interleaved_mem_config,
        )  # produces dim of total tensor!
        out_temp = tt2torch_tensor(out_tt_tensor)
        print(f"Printing output tensor after slice {slice_index}")
        print(out_temp)

    pt_out = in0

    tt_out = tt2torch_tensor(out_tt_tensor)

    print("TT output")
    print(tt_out)

    print("Pytorch output")
    print(pt_out)

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing


@pytest.mark.parametrize("H, W, num_cores", [[32 * 32, 16 * 32, 64]])
@pytest.mark.parametrize(
    "activations_dtype",
    [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B],
    ids=["inputs_BFLOAT16", "inputs_BFLOAT8_B"],
)
@pytest.mark.parametrize(
    "output_dtype", [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B], ids=["out_BFLOAT16", "out_BFLOAT8_B"]
)
def test_block_sharded_partial_spill_reload(
    device,
    H,
    W,
    num_cores,
    activations_dtype,
    output_dtype,
    function_level_defaults,
):
    grid_size = device.compute_with_storage_grid_size()
    compute_grid_size = device.compute_with_storage_grid_size()
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")
    in0_shape = [1, 1, H, W]
    W = in0_shape[-1]

    interleaved_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.L1,
    )
    sharded_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()
    out_initial = torch.randn(in0_shape).bfloat16().float()
    # mid_index = in0_shape[2] // 2  # H dimension
    # in0[:, :, mid_index:] = 2

    # torch.set_printoptions(profile="full", sci_mode=False, linewidth=220)

    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype)
    out_tt_tensor = torch2tt_tensor(
        out_initial, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype
    )

    block_shard_spec = [2 * 32, 2 * 32]
    num_slices = 2

    for slice_index in range(num_slices):
        print("Running slice: ", slice_index)
        in0_t_slice = ttl.tensor.interleaved_to_sharded_partial(
            in0_t,
            grid_size,
            block_shard_spec,
            ttl.tensor.ShardedOpSplitDim.DimRow,
            num_slices,  # num_slices
            slice_index,  # slice_index
            ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
            ttl.tensor.ShardOrientation.ROW_MAJOR,
        )

        ttl.tensor.sharded_to_interleaved_partial(
            in0_t_slice,
            out_tt_tensor,
            ttl.tensor.ShardedOpSplitDim.DimRow,
            num_slices,
            slice_index,
            interleaved_mem_config,
        )  # produces dim of total tensor!
        # out_temp = tt2torch_tensor(out_tt_tensor)
        # print(f"Printing output tensor after slice {slice_index}")
        # print(out_temp)

    pt_out = in0

    tt_out = tt2torch_tensor(out_tt_tensor)

    print("TT output")
    print(tt_out)

    print("Pytorch output")
    print(pt_out)

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing


@pytest.mark.parametrize("seq_len", [1024, 2048], ids=["seq_len_1024", "seq_len_2048"])
# @pytest.mark.parametrize("slice", [False, True], ids=["one_slice", "two_slices"])
@pytest.mark.parametrize("num_cores", [64])
def test_falcon7b_attnention_reference_matmuls(
    device,
    seq_len,
    num_cores,
    function_level_defaults,
):
    # query layer
    query_layer_shape = [1, 71, seq_len, 64]
    key_layer_transposed_shape = [1, 1, 64, seq_len]
    scalar_shape = [1, 1, 32, 32]
    attention_mask_shape = [1, 71, seq_len, seq_len]

    torch_query_layer = torch.randn(query_layer_shape).bfloat16().float()
    torch_key_layer_transposed = torch.randn(key_layer_transposed_shape).bfloat16().float()
    torch_scalar = torch.randn(scalar_shape).bfloat16().float()
    torch_attention_mask = torch.randn(attention_mask_shape).bfloat16().float()

    dram_interleaved_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.DRAM,
    )

    # compare output to regular case
    reference_query_layer = torch2tt_tensor(
        torch_query_layer,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=ttl.tensor.DataType.BFLOAT16,
    )
    reference_key_layer_transposed = torch2tt_tensor(
        torch_key_layer_transposed,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=ttl.tensor.DataType.BFLOAT16,
    )

    # matmul
    # optimised version
    if seq_len == 1024:
        compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
            math_fidelity=ttl.tensor.MathFidelity.HiFi4,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        # program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        #     compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        #     in0_block_w=1,
        #     per_core_M=36,
        #     per_core_N=32,
        #     out_subblock_h=1,
        #     out_subblock_w=1,
        #     fuse_batch = True,
        #     fused_activation=None,
        #     mcast_in0 = False
        # )
        # attn_weights = ttl.operations.primary.matmul(
        #     reference_query_layer,
        #     reference_key_layer_transposed,
        #     program_config=program_config,
        #     output_mem_config=dram_interleaved_memory_config,
        #     output_dtype=ttl.tensor.DataType.BFLOAT16,
        #     compute_kernel_config=compute_kernel_config,
        # )
        attn_weights = ttl.tensor.matmul(
            reference_query_layer, reference_key_layer_transposed, output_mem_config=dram_interleaved_memory_config
        )
    else:
        attn_weights = ttl.tensor.matmul(
            reference_query_layer, reference_key_layer_transposed, output_mem_config=dram_interleaved_memory_config
        )

    # broadcast
    reference_scalar = torch2tt_tensor(
        torch_scalar, device, tt_memory_config=dram_interleaved_memory_config, tt_dtype=ttl.tensor.DataType.BFLOAT16
    )

    attn_weights = ttl.tensor.bcast(
        attn_weights,
        reference_scalar,
        ttl.tensor.BcastOpMath.MUL,
        ttl.tensor.BcastOpDim.HW,
        output_mem_config=dram_interleaved_memory_config,
    )

    attention_mask = torch2tt_tensor(
        torch_attention_mask,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=ttl.tensor.DataType.BFLOAT16,
    )

    attn_weights = ttl.tensor.add(
        attn_weights,
        attention_mask,
        output_mem_config=dram_interleaved_memory_config,
    )

    # softmax
    attn_weights = ttl.operations.primary.softmax_in_place(attn_weights)

    passing = True
    assert passing


@pytest.mark.parametrize("seq_len", [1024, 2048], ids=["seq_len_1024", "seq_len_2048"])
@pytest.mark.parametrize("num_slices", [1, 2, 4], ids=["one_slice", "two_slices", "four_slices"])
@pytest.mark.parametrize("num_cores", [64])
def test_falcon7b_attnention_slice_matmuls(
    device,
    seq_len,
    num_slices,
    num_cores,
    function_level_defaults,
):
    # query layer
    query_layer_shape = [1, 71, seq_len, 64]
    key_layer_transposed_shape = [1, 1, 64, seq_len]
    scalar_shape = [1, 1, 32, 32]
    attention_mask_shape = [1, 71, seq_len, seq_len]

    torch_query_layer = torch.randn(query_layer_shape).bfloat16().float()
    torch_key_layer_transposed = torch.randn(key_layer_transposed_shape).bfloat16().float()
    torch_scalar = torch.randn(scalar_shape).bfloat16().float()
    torch_attention_mask = torch.randn(attention_mask_shape).bfloat16().float()

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
        tt_dtype=ttl.tensor.DataType.BFLOAT16,
    )
    reference_key_layer_transposed = torch2tt_tensor(
        torch_key_layer_transposed,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=ttl.tensor.DataType.BFLOAT16,
    )

    # broadcast
    reference_scalar = torch2tt_tensor(
        torch_scalar, device, tt_memory_config=dram_interleaved_memory_config, tt_dtype=ttl.tensor.DataType.BFLOAT16
    )

    # attention_mask_slices = []
    # slice_width = seq_len // num_slices
    # for i in range(num_slices):
    #     attention_mask_slices.append(
    #         torch2tt_tensor(
    #             torch_attention_mask[:, :, (i) * slice_width : (i + 1) * slice_width, :],
    #             device,
    #             tt_memory_config=dram_interleaved_memory_config,
    #             tt_dtype=ttl.tensor.DataType.BFLOAT16,
    #         )
    #     )

    attention_mask = torch2tt_tensor(
        torch_attention_mask,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=ttl.tensor.DataType.BFLOAT16,
    )

    compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttl.tensor.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    # matmul
    # optimised version
    passing = True
    output = None
    if seq_len == 1024:
        # prepare interleaved output, temp
        mm_out = torch2tt_tensor(
            torch_attention_mask,
            device,
            tt_memory_config=dram_interleaved_memory_config,
            tt_dtype=ttl.tensor.DataType.BFLOAT16,
        )
        # tiles_per_shard = (math.ceil((71 * seq_len) // num_cores) // num_slices)
        tiles_per_shard = math.ceil((((71 * seq_len) / num_cores) / num_slices) / 32)
        print("Tiles per shard is: ", tiles_per_shard)
        mm_activations_height_shard_spec = [tiles_per_shard * 32, 2 * 32]
        mm_output_height_shard_spec = [tiles_per_shard * 32, seq_len]

        for i in range(num_slices):
            print("Running slice: ", i)
            slice = ttl.tensor.interleaved_to_sharded_partial(
                reference_query_layer,
                device.compute_with_storage_grid_size(),
                mm_activations_height_shard_spec,
                ttl.tensor.ShardedOpSplitDim.DimRow,
                num_slices,  # num_slices
                i,  # slice_index
                ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
                ttl.tensor.ShardOrientation.ROW_MAJOR,
            )

            # Let's do a partial matmul :)
            program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
                in0_block_w=2,
                per_core_M=tiles_per_shard,
                per_core_N=32,
                out_subblock_h=1,
                out_subblock_w=1,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=False,
            )

            # [1, 1, 71, 32, 2] * [2, 32]
            mm_slice = ttl.operations.primary.matmul(
                slice,
                reference_key_layer_transposed,
                program_config=program_config,
                output_mem_config=height_sharded_memory_config,
                output_dtype=ttl.tensor.DataType.BFLOAT16,
                compute_kernel_config=compute_kernel_config,
            )

            # Perform broadcast
            # print("Running bcast")
            # mm_slice = ttl.tensor.bcast(
            #     mm_slice,
            #     reference_scalar,
            #     ttl.tensor.BcastOpMath.MUL,
            #     ttl.tensor.BcastOpDim.HW,
            #     output_mem_config=height_sharded_memory_config,
            # )

            # Slice attention mask
            # [1, 1, 71, 1024, 1024]
            # [1, 1, 71, 256, 2024]
            # This really should be [1, 1, 1024, 32!!!] tiles
            # Sharded add bcast
            attn_mask_slice = ttl.tensor.interleaved_to_sharded_partial(
                attention_mask,
                device.compute_with_storage_grid_size(),
                mm_output_height_shard_spec,
                ttl.tensor.ShardedOpSplitDim.DimRow,
                num_slices,  # num_slices
                i,  # slice_index
                ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
                ttl.tensor.ShardOrientation.ROW_MAJOR,
            )

            # tensor.add(mm_slice, att_mask[0:1024],)
            mm_slice = ttl.tensor.add_without_autoformat(
                mm_slice, attn_mask_slice, output_mem_config=height_sharded_memory_config, in_place=True
            )

            attn_mask_slice.deallocate()

            softmax_program_config = ttl.operations.primary.transformers.SoftmaxShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
                subblock_w=1,
                block_h=mm_output_height_shard_spec[0] // 32,
                block_w=mm_output_height_shard_spec[1] // 32,
                math_fidelity=ttl.tensor.MathFidelity.HiFi4,
                im_data_format=ttl.tensor.DataType.BFLOAT16,
            )

            mm_slice = ttl.operations.primary.softmax_in_place(mm_slice, program_config=softmax_program_config)

            ttl.tensor.sharded_to_interleaved_partial(
                mm_slice,
                mm_out,
                ttl.tensor.ShardedOpSplitDim.DimRow,
                num_slices,
                i,
                dram_interleaved_memory_config,
            )  # produces dim of total tensor!

            mm_slice.deallocate()
            # attn_mask_slice.deallocate()

        # mm_out = ttl.operations.primary.softmax_in_place(mm_out)
        mm_out_torch = tt2torch_tensor(mm_out)

        attn_weights = ttl.tensor.matmul(
            reference_query_layer, reference_key_layer_transposed, output_mem_config=dram_interleaved_memory_config
        )

        # attn_weights = ttl.tensor.bcast(
        #     attn_weights,
        #     reference_scalar,
        #     ttl.tensor.BcastOpMath.MUL,
        #     ttl.tensor.BcastOpDim.HW,
        #     output_mem_config=dram_interleaved_memory_config,
        # )

        attn_weights = ttl.tensor.add(attn_weights, attention_mask, output_mem_config=dram_interleaved_memory_config)
        attn_weights = ttl.operations.primary.softmax_in_place(attn_weights)

        attn_weights_torch = tt2torch_tensor(attn_weights)
        passing, output = comp_pcc(mm_out_torch, attn_weights_torch)
    else:
        attn_weights = ttl.tensor.matmul(
            reference_query_layer, reference_key_layer_transposed, output_mem_config=dram_interleaved_memory_config
        )

    attn_weights = ttl.tensor.bcast(
        attn_weights,
        reference_scalar,
        ttl.tensor.BcastOpMath.MUL,
        ttl.tensor.BcastOpDim.HW,
        output_mem_config=dram_interleaved_memory_config,
    )

    attention_mask = torch2tt_tensor(
        torch_attention_mask,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=ttl.tensor.DataType.BFLOAT16,
    )

    attn_weights = ttl.tensor.add(
        attn_weights,
        attention_mask,
        output_mem_config=dram_interleaved_memory_config,
    )

    # softmax
    attn_weights = ttl.operations.primary.softmax_in_place(attn_weights)

    if not passing:
        print(output)

    assert passing


@pytest.mark.parametrize("H, W, num_cores", [[32 * 32, 16 * 32, 64]])
@pytest.mark.parametrize(
    "activations_dtype",
    [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B],
    ids=["inputs_BFLOAT16", "inputs_BFLOAT8_B"],
)
@pytest.mark.parametrize(
    "output_dtype", [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B], ids=["out_BFLOAT16", "out_BFLOAT8_B"]
)
def test_block_sharded_binary(
    device,
    H,
    W,
    num_cores,
    activations_dtype,
    output_dtype,
    function_level_defaults,
):
    grid_size = device.compute_with_storage_grid_size()
    compute_grid_size = device.compute_with_storage_grid_size()
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")
    in0_shape = [1, 1, H, W]
    in1_shape = in0_shape
    W = in0_shape[-1]

    interleaved_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.L1,
    )
    sharded_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    # in0 = torch.randn(in0_shape).bfloat16().float()
    in0 = torch.randn(in0_shape).bfloat16().float()
    # mid_index = in0_shape[2] // 2  # H dimension
    # in0[:, :, mid_index:] = 2

    in1 = torch.randn(in1_shape).bfloat16().float()
    # in1[:, :, mid_index:] = 2

    out_values = torch.randn(in0_shape).bfloat16().float()

    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype)
    out_tt_tensor = torch2tt_tensor(
        out_values, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype
    )

    output_mem_config = sharded_mem_config
    block_shard_spec = [2 * 32, 2 * 32]
    num_slices = 2

    for slice_index in range(num_slices):
        print("Running slice: ", slice_index)
        in0_t_slice = ttl.tensor.interleaved_to_sharded_partial(
            in0_t,
            grid_size,
            block_shard_spec,
            ttl.tensor.ShardedOpSplitDim.DimRow,
            num_slices,  # num_slices
            slice_index,  # slice_index
            ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
            ttl.tensor.ShardOrientation.ROW_MAJOR,
        )

        in1_t_slice = ttl.tensor.interleaved_to_sharded_partial(
            in1_t,
            grid_size,
            block_shard_spec,
            ttl.tensor.ShardedOpSplitDim.DimRow,
            num_slices,  # num_slices
            slice_index,  # slice_index
            ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
            ttl.tensor.ShardOrientation.ROW_MAJOR,
        )

        sliced_tensor = ttl.tensor.add(
            in0_t_slice, in1_t_slice, output_mem_config=output_mem_config, output_dtype=output_dtype
        )  # produces a dim of slice, rather than the entire tensor, outputs on device
        ttl.tensor.sharded_to_interleaved_partial(
            sliced_tensor,
            out_tt_tensor,
            ttl.tensor.ShardedOpSplitDim.DimRow,
            num_slices,
            slice_index,
            interleaved_mem_config,
        )  # produces dim of total tensor!

    pt_out = in0 + in1

    tt_out = tt2torch_tensor(out_tt_tensor)

    print("TT output")
    print(tt_out)

    print("Pytorch output")
    print(pt_out)

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing


@pytest.mark.parametrize("in0_sharded", [True])
@pytest.mark.parametrize("in1_sharded", [True])
@pytest.mark.parametrize("out_sharded", [True])
@pytest.mark.parametrize("H, num_cores", [[128 * 32, 64]])
@pytest.mark.parametrize(
    "activations_dtype",
    [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B],
    ids=["inputs_BFLOAT16", "inputs_BFLOAT8_B"],
)
@pytest.mark.parametrize(
    "output_dtype", [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B], ids=["out_BFLOAT16", "out_BFLOAT8_B"]
)
def test_my_sharded_binary(
    device,
    in0_sharded,
    in1_sharded,
    out_sharded,
    H,
    num_cores,
    activations_dtype,
    output_dtype,
    function_level_defaults,
):
    grid_size = device.compute_with_storage_grid_size()
    compute_grid_size = device.compute_with_storage_grid_size()
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")
    in0_shape = [1, 1, H, 96]
    in1_shape = in0_shape
    W = in0_shape[-1]

    if out_sharded and not in0_sharded and not in1_sharded and H == 64:
        pytest.skip("Unsupported sharding config")

    interleaved_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.L1,
    )
    sharded_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    # in0 = torch.randn(in0_shape).bfloat16().float()
    in0 = torch.randn(in0_shape).bfloat16().float()
    # mid_index = in0_shape[2] // 2  # H dimension
    # in0[:, :, mid_index:] = 2

    in1 = torch.randn(in1_shape).bfloat16().float()
    # in1[:, :, mid_index:] = 2

    out_values = torch.randn(in0_shape).bfloat16().float()

    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype)
    out_tt_tensor = torch2tt_tensor(
        out_values, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype
    )

    output_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config
    height_shard_spec = [32, W]  # Total H dim is 128 * 32
    num_slices = 2

    for slice_index in range(num_slices):
        print("Running slice: ", slice_index)
        in0_t_slice = ttl.tensor.interleaved_to_sharded_partial(
            in0_t,
            grid_size,
            height_shard_spec,
            ttl.tensor.ShardedOpSplitDim.DimRow,
            num_slices,  # num_slices
            slice_index,  # slice_index
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            ttl.tensor.ShardOrientation.ROW_MAJOR,
        )

        in1_t_slice = ttl.tensor.interleaved_to_sharded_partial(
            in1_t,
            grid_size,
            height_shard_spec,
            ttl.tensor.ShardedOpSplitDim.DimRow,
            num_slices,  # num_slices
            slice_index,  # slice_index
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            ttl.tensor.ShardOrientation.ROW_MAJOR,
        )

        sliced_tensor = ttl.tensor.add(
            in0_t_slice, in1_t_slice, output_mem_config=output_mem_config, output_dtype=output_dtype
        )  # produces a dim of slice, rather than the entire tensor, outputs on device
        ttl.tensor.sharded_to_interleaved_partial(
            sliced_tensor,
            out_tt_tensor,
            ttl.tensor.ShardedOpSplitDim.DimRow,
            num_slices,
            slice_index,
            interleaved_mem_config,
        )  # produces dim of total tensor!

    pt_out = in0 + in1

    tt_out = tt2torch_tensor(out_tt_tensor)

    print("TT output")
    print(tt_out)

    print("Pytorch output")
    print(pt_out)

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing


@pytest.mark.parametrize("in0_sharded", [True, False], ids=["in0_sharded", "in0_unsharded"])
@pytest.mark.parametrize("in1_sharded", [True, False], ids=["in1_sharded", "in1_unsharded"])
@pytest.mark.parametrize("out_sharded", [True, False], ids=["out_sharded", "out_unsharded"])
@pytest.mark.parametrize("H, num_cores", [[128, 64]])
@pytest.mark.parametrize(
    "activations_dtype",
    [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B],
    ids=["inputs_BFLOAT16", "inputs_BFLOAT8_B"],
)
@pytest.mark.parametrize(
    "output_dtype", [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B], ids=["out_BFLOAT16", "out_BFLOAT8_B"]
)
def test_sharded_binary(
    device,
    in0_sharded,
    in1_sharded,
    out_sharded,
    H,
    num_cores,
    activations_dtype,
    output_dtype,
    function_level_defaults,
):
    grid_size = device.compute_with_storage_grid_size()
    compute_grid_size = device.compute_with_storage_grid_size()
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")
    in0_shape = [1, 1, H, 64]
    in1_shape = in0_shape
    W = in0_shape[-1]

    if out_sharded and not in0_sharded and not in1_sharded and H == 25088:
        pytest.skip("Unsupported sharding config")

    interleaved_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.L1,
    )
    sharded_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()

    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype)

    output_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config

    if in0_sharded:
        in0_t = ttl.tensor.interleaved_to_sharded(
            in0_t,
            grid_size,
            [H // 2, W],
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            ttl.tensor.ShardOrientation.ROW_MAJOR,
        )

    if in1_sharded:
        in1_t = ttl.tensor.interleaved_to_sharded(
            in1_t,
            grid_size,
            [H // 2, W],
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            ttl.tensor.ShardOrientation.ROW_MAJOR,
        )

    output_t = ttl.tensor.add(in0_t, in1_t, output_mem_config=output_mem_config, output_dtype=output_dtype)
    if out_sharded:
        output_t = ttl.tensor.sharded_to_interleaved(output_t, interleaved_mem_config)
    pt_out = in0 + in1

    tt_out = tt2torch_tensor(output_t)

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing


def test_sharded_program_cache(device, use_program_cache, function_level_defaults):
    grid_size = device.compute_with_storage_grid_size()
    num_cores = 98
    compute_grid_size = device.compute_with_storage_grid_size()
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")
    N = 1
    C = 1
    H = 25088
    W = 64
    x = torch.ones((N, C, H, W)).bfloat16().float()
    x2 = torch.zeros((N, C, H, W)).bfloat16().float()

    xt = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(
            device,
            ttl.tensor.MemoryConfig(
                memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
                buffer_type=ttl.tensor.BufferType.L1,
            ),
        )
    )

    yt = ttl.tensor.interleaved_to_sharded(
        xt,
        grid_size,
        [H // num_cores, W],
        ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
        ttl.tensor.ShardOrientation.ROW_MAJOR,
    )

    zt = ttl.tensor.sharded_to_interleaved(
        yt,
        ttl.tensor.MemoryConfig(
            memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttl.tensor.BufferType.L1,
        ),
    )

    xt2 = (
        ttl.tensor.Tensor(
            x2.reshape(-1).tolist(),
            x2.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(
            device,
            ttl.tensor.MemoryConfig(
                memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
                buffer_type=ttl.tensor.BufferType.L1,
            ),
        )
    )

    yt2 = ttl.tensor.interleaved_to_sharded(
        xt2,
        grid_size,
        [H // num_cores, W],
        ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
        ttl.tensor.ShardOrientation.ROW_MAJOR,
    )

    zt2 = ttl.tensor.sharded_to_interleaved(
        yt2,
        ttl.tensor.MemoryConfig(
            memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttl.tensor.BufferType.L1,
        ),
    )
    zt = ttl.tensor.sharded_to_interleaved(
        yt,
        ttl.tensor.MemoryConfig(
            memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttl.tensor.BufferType.L1,
        ),
    )

    tt_og = xt.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    tt_og2 = xt2.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    tt_got_back = zt.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    tt_got_back2 = zt2.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    eq = torch.equal(tt_og, tt_got_back)
    assert eq
    eq = torch.equal(tt_og2, tt_got_back2)
    assert eq


@pytest.mark.parametrize("in0_sharded", [True, False], ids=["in0_sharded", "in0_unsharded"])
@pytest.mark.parametrize("out_sharded", [True, False], ids=["out_sharded", "out_unsharded"])
@pytest.mark.parametrize("M", [1600])
@pytest.mark.parametrize("N", [1024])
@pytest.mark.parametrize("activations_dtype", [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B])
@pytest.mark.parametrize("weights_dtype", [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B])
def test_sharded_matmul_2d(
    device, in0_sharded, out_sharded, M, N, activations_dtype, weights_dtype, function_level_defaults
):
    K = 256
    in0_shape = [1, 1, M, K]
    in1_shape = [1, 1, K, N]
    bias_shape = [1, 1, 1, N]

    grid_size = (8, 5)
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")
    if activations_dtype != weights_dtype and is_wormhole_b0():
        pytest.skip("WH does not work with mixed precision")

    interleaved_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.L1,
    )
    sharded_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()
    bias = torch.randn(bias_shape).bfloat16().float()

    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=interleaved_mem_config, tt_dtype=weights_dtype)
    bias_t = pad_by_zero(bias, device, tt_memory_config=interleaved_mem_config, tt_dtype=weights_dtype)[0]

    output_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config

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
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=4,
        per_core_M=10,
        per_core_N=4,
        transpose_mcast=False,
        fused_activation=None,
    )
    output_t = ttl.operations.primary.matmul(
        in0_t,
        in1_t,
        bias=bias_t,
        program_config=program_config,
        output_mem_config=output_mem_config,
        output_dtype=activations_dtype,
    )
    if out_sharded:
        output_t = ttl.tensor.sharded_to_interleaved(output_t, interleaved_mem_config)
    pt_out = in0 @ in1 + bias

    tt_out = tt2torch_tensor(output_t)

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing


@pytest.mark.parametrize("in0_sharded", [True, False], ids=["in0_sharded", "in0_unsharded"])
@pytest.mark.parametrize("out_sharded", [True, False], ids=["out_sharded", "out_unsharded"])
@pytest.mark.parametrize("M", [1600])
@pytest.mark.parametrize("N", [1024])
@pytest.mark.parametrize("activations_dtype", [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B])
@pytest.mark.parametrize("weights_dtype", [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B])
def test_sharded_matmul_2d_transposed(
    device, in0_sharded, out_sharded, M, N, activations_dtype, weights_dtype, function_level_defaults
):
    K = 256
    in0_shape = [1, 1, M, K]
    in1_shape = [1, 1, K, N]
    bias_shape = [1, 1, 1, N]

    grid_size = (10, 8)
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")
    if activations_dtype != weights_dtype and is_wormhole_b0():
        pytest.skip("WH does not work with mixed precision")

    interleaved_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.L1,
    )
    sharded_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()
    bias = torch.randn(bias_shape).bfloat16().float()

    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=interleaved_mem_config, tt_dtype=weights_dtype)
    bias_t = pad_by_zero(bias, device, tt_memory_config=interleaved_mem_config, tt_dtype=weights_dtype)[0]

    output_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config

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
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=4,
        per_core_M=5,
        per_core_N=4,
        transpose_mcast=True,
        fused_activation=None,
    )
    output_t = ttl.operations.primary.matmul(
        in0_t,
        in1_t,
        bias=bias_t,
        program_config=program_config,
        output_mem_config=output_mem_config,
        output_dtype=activations_dtype,
    )
    if out_sharded:
        output_t = ttl.tensor.sharded_to_interleaved(output_t, interleaved_mem_config)
    pt_out = in0 @ in1 + bias

    tt_out = tt2torch_tensor(output_t)

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing


def test_resharded_binary_to_matmul(device, function_level_defaults):
    grid_size_binary = device.compute_with_storage_grid_size()
    num_cores_binary = 98
    compute_grid_size = device.compute_with_storage_grid_size()
    if num_cores_binary > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores_binary} cores to run this test but core grid is {compute_grid_size}")
    grid_size_matmul = (10, 8)
    if grid_size_matmul[0] > compute_grid_size.x or grid_size_matmul[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size_matmul} grid size to run this test but core grid is {compute_grid_size}")
    in0_shape = [1, 1, 6272, 512]
    in1_shape = in0_shape
    weight_shape = [1, 1, 512, 256]
    bias_shape = [1, 1, 1, 256]
    H = in0_shape[-2]
    W = in0_shape[-1]

    interleaved_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.L1,
    )
    height_sharded_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttl.tensor.BufferType.L1,
    )
    block_sharded_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()
    weight = torch.randn(weight_shape).bfloat16().float()
    bias = torch.randn(bias_shape).bfloat16().float()

    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=interleaved_mem_config)
    weight_t = torch2tt_tensor(weight, device, tt_memory_config=interleaved_mem_config)
    bias_t = pad_by_zero(bias, device, tt_memory_config=interleaved_mem_config)[0]

    in0_t = ttl.tensor.interleaved_to_sharded(
        in0_t,
        grid_size_binary,
        [H // num_cores_binary, W],
        ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
        ttl.tensor.ShardOrientation.ROW_MAJOR,
    )

    in1_t = ttl.tensor.interleaved_to_sharded(
        in1_t,
        grid_size_binary,
        [H // num_cores_binary, W],
        ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
        ttl.tensor.ShardOrientation.ROW_MAJOR,
    )

    output_binary_t = ttl.tensor.add(in0_t, in1_t, output_mem_config=interleaved_mem_config)
    output_binary_t = ttl.tensor.interleaved_to_sharded(
        output_binary_t,
        grid_size_matmul,
        [math.ceil((H // 32) / grid_size_matmul[0]) * 32, W // grid_size_matmul[1]],
        ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        ttl.tensor.ShardOrientation.COL_MAJOR,
    )
    program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size_matmul,
        in0_block_w=2,
        out_subblock_h=5,
        out_subblock_w=1,
        per_core_M=20,
        per_core_N=1,
        transpose_mcast=True,
        fused_activation=None,
    )
    output_matmul_t = ttl.operations.primary.matmul(
        output_binary_t,
        weight_t,
        bias=bias_t,
        program_config=program_config,
        output_mem_config=block_sharded_mem_config,
    )
    output_matmul_t = ttl.tensor.sharded_to_interleaved(output_matmul_t, interleaved_mem_config)

    tt_out = tt2torch_tensor(output_matmul_t)

    pt_out = (in0 + in1) @ weight

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing


@pytest.mark.parametrize("in_sharded", [True, False], ids=["in0_sharded", "in0_unsharded"])
@pytest.mark.parametrize("out_sharded", [False], ids=["out_unsharded"])
@pytest.mark.parametrize("dtype", [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B])
def test_sharded_untilize_padded_shard(in_sharded, out_sharded, dtype, device, function_level_defaults):
    grid_size = (10, 8)
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")
    N = 1
    C = 1
    H = 6272
    W = 256

    interleaved_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.L1,
    )
    sharded_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    out_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config

    x = torch.arange(N * C * H * W).reshape((N, C, H, W)).bfloat16()

    xt = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            dtype,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(
            device,
            interleaved_mem_config,
        )
    )

    if in_sharded:
        xt = ttl.tensor.interleaved_to_sharded(
            xt,
            grid_size,
            [
                math.ceil((xt.get_legacy_shape()[-2] // 32) / grid_size[0]) * 32,
                xt.get_legacy_shape()[-1] // grid_size[1],
            ],
            ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
            ttl.tensor.ShardOrientation.COL_MAJOR,
        )

    yt = ttl.tensor.untilize(
        xt,
        output_mem_config=out_mem_config,
        use_multicore=True,
    )

    if out_sharded:
        yt = ttl.tensor.sharded_to_interleaved(
            yt,
            interleaved_mem_config,
        )

    tt_got_back = yt.cpu().to_torch()

    if dtype == ttl.tensor.DataType.BFLOAT16:
        passing, output = comp_equal(x, tt_got_back)
    else:
        passing, output = comp_pcc(x, tt_got_back, 0.999)
    logger.info(output)

    assert passing


@pytest.mark.parametrize("in_sharded", [True, False], ids=["in0_sharded", "in0_unsharded"])
@pytest.mark.parametrize("out_sharded", [False], ids=["out_unsharded"])
@pytest.mark.parametrize("activations_dtype", [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B])
@pytest.mark.parametrize("output_dtype", [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B])
def test_sharded_binary_padded_shard(
    in_sharded, out_sharded, activations_dtype, output_dtype, device, function_level_defaults
):
    grid_size = (10, 8)
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")
    N = 1
    C = 1
    H = 1568
    W = 1024

    interleaved_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.L1,
    )
    sharded_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    out_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config

    x = torch.ones((N, C, H, W)).bfloat16()
    y = torch.ones((N, C, H, W)).bfloat16() * 2

    xt = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            activations_dtype,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(
            device,
            interleaved_mem_config,
        )
    )

    yt = (
        ttl.tensor.Tensor(
            y.reshape(-1).tolist(),
            y.shape,
            activations_dtype,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(
            device,
            interleaved_mem_config,
        )
    )

    if in_sharded:
        xt = ttl.tensor.interleaved_to_sharded(
            xt,
            grid_size,
            [
                math.ceil((xt.get_legacy_shape()[-2] // 32) / grid_size[0]) * 32,
                xt.get_legacy_shape()[-1] // grid_size[1],
            ],
            ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
            ttl.tensor.ShardOrientation.COL_MAJOR,
        )
        yt = ttl.tensor.interleaved_to_sharded(
            yt,
            grid_size,
            [
                math.ceil((xt.get_legacy_shape()[-2] // 32) / grid_size[0]) * 32,
                xt.get_legacy_shape()[-1] // grid_size[1],
            ],
            ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
            ttl.tensor.ShardOrientation.COL_MAJOR,
        )

    zt = ttl.tensor.add(xt, yt, output_mem_config=out_mem_config, output_dtype=output_dtype)

    if out_sharded:
        zt = ttl.tensor.sharded_to_interleaved(
            zt,
            interleaved_mem_config,
        )

    tt_got_back = zt.cpu().to_torch()

    passing, output = comp_equal(x + y, tt_got_back)
    logger.info(output)

    assert passing


@pytest.mark.parametrize("in_sharded", [True, False], ids=["in0_sharded", "in0_unsharded"])
@pytest.mark.parametrize("out_sharded", [False], ids=["out_unsharded"])
@pytest.mark.parametrize("dtype", [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B])
def test_block_sharded_untilize_with_unpadding(in_sharded, out_sharded, dtype, device, function_level_defaults):
    grid_size = (7, 8)
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")
    N = 1
    C = 1
    H = 416
    W = 512

    interleaved_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.L1,
    )
    sharded_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    out_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config

    x = torch.randn((N, C, H, W)).bfloat16()

    xt = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            dtype,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(
            device,
            interleaved_mem_config,
        )
    )

    if in_sharded:
        xt = ttl.tensor.interleaved_to_sharded(
            xt,
            grid_size,
            [
                math.ceil((xt.get_legacy_shape()[-2] // 32) / grid_size[0]) * 32,
                xt.get_legacy_shape()[-1] // grid_size[1],
            ],
            ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
            ttl.tensor.ShardOrientation.COL_MAJOR,
        )

    yt = ttl.tensor.untilize_with_unpadding(
        xt,
        ttl.tensor.Shape([0, 0, 0, 0]),
        ttl.tensor.Shape([0, 0, 391, 511]),
        output_mem_config=out_mem_config,
    )

    if out_sharded:
        yt = ttl.tensor.sharded_to_interleaved(
            yt,
            interleaved_mem_config,
        )

    tt_got_back = yt.cpu().to_torch()

    y = x[..., :392, :512]

    if dtype == ttl.tensor.DataType.BFLOAT16:
        passing, output = comp_equal(y, tt_got_back)
    else:
        passing, output = comp_pcc(y, tt_got_back, 0.999)

    logger.info(output)

    assert passing


@pytest.mark.parametrize("in_sharded", [True], ids=["in0_sharded"])
@pytest.mark.parametrize(
    "shape, output_H, out_sharded",
    [
        [(8, 1, 32, 2048), 1, True],
        [(1, 1, 32, 1024), 8, False],
        [(16, 1, 32, 2048), 1, True],
        [(1, 1, 32, 1024), 16, False],
    ],
    ids=[
        "batched_8_shape_out_sharded",
        "unbatched_8_shape_out_interleaved",
        "batched_16_shape_out_sharded",
        "unbatched_16_shape_out_interleaved",
    ],
)
@pytest.mark.parametrize("dtype", [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B])
def test_width_sharded_untilize_with_unpadding(
    shape, output_H, in_sharded, out_sharded, dtype, device, function_level_defaults
):
    grid_size = (8, 4)
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")
    N, C, H, W = shape

    interleaved_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.L1,
    )
    sharded_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    out_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config

    x = torch.randn((N, C, H, W)).bfloat16()

    xt = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            dtype,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(
            device,
            interleaved_mem_config,
        )
    )

    if in_sharded:
        xt = ttl.tensor.interleaved_to_sharded(
            xt,
            grid_size,
            [N * C * H, W // (grid_size[0] * grid_size[1])],
            ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED,
            ttl.tensor.ShardOrientation.COL_MAJOR,
        )

    yt = ttl.tensor.untilize_with_unpadding(
        xt,
        ttl.tensor.Shape([0, 0, 0, 0]),
        ttl.tensor.Shape([N - 1, C - 1, output_H - 1, W - 1]),
        output_mem_config=out_mem_config,
    )

    if out_sharded:
        yt = ttl.tensor.sharded_to_interleaved(
            yt,
            interleaved_mem_config,
        )

    tt_got_back = yt.cpu().to_torch()

    y = x[..., :output_H, :]
    if dtype == ttl.tensor.DataType.BFLOAT16:
        passing, output = comp_equal(y, tt_got_back)
    else:
        passing, output = comp_pcc(y, tt_got_back, 0.999)
    logger.info(output)

    assert passing


@pytest.mark.parametrize("input_shape", [[8, 1, 49, 2048], [1, 1, 8, 2048], [16, 1, 49, 2048], [1, 1, 16, 2048]])
@pytest.mark.parametrize("in_sharded", [True], ids=["in0_sharded"])
@pytest.mark.parametrize("out_sharded", [True], ids=["out_sharded"])
@pytest.mark.parametrize("output_dtype", [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B])
def test_sharded_tilize_with_val_padding(
    input_shape, in_sharded, out_sharded, output_dtype, device, function_level_defaults
):
    grid_size = (8, 4)
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")
    N, C, H, W = input_shape

    interleaved_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.L1,
    )
    sharded_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    out_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config

    x = torch.arange(N * C * H * W).reshape((N, C, H, W)).bfloat16()

    xt = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    ).to(
        device,
        interleaved_mem_config,
    )

    if in_sharded:
        xt = ttl.tensor.interleaved_to_sharded(
            xt,
            grid_size,
            [N * C * H, W // (grid_size[0] * grid_size[1])],
            ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED,
            ttl.tensor.ShardOrientation.COL_MAJOR,
        )

    yt = ttl.tensor.tilize_with_val_padding(
        xt,
        ttl.tensor.Shape([N, C, roundup32(H), W]),
        ttl.tensor.Shape([0, 0, 0, 0]),
        1.0,
        output_mem_config=out_mem_config,
        output_dtype=output_dtype,
    )

    if out_sharded:
        yt = ttl.tensor.sharded_to_interleaved(
            yt,
            interleaved_mem_config,
        )

    tt_got_back = yt.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    y = torch.nn.functional.pad(x, [0, 0, 0, roundup32(H) - H], "constant", 1.0)

    if output_dtype == ttl.tensor.DataType.BFLOAT16:
        passing, output = comp_equal(y, tt_got_back)
    else:
        passing, output = comp_pcc(y, tt_got_back, 0.999)
    logger.info(output)

    assert passing


@pytest.mark.parametrize("N", [8, 16])
@pytest.mark.parametrize("in_sharded", [True], ids=["in0_sharded"])
@pytest.mark.parametrize("out_sharded", [True], ids=["out_sharded"])
@pytest.mark.parametrize("dtype", [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B])
def test_sharded_reduce_h(N, in_sharded, out_sharded, dtype, device, function_level_defaults):
    grid_size = (8, 4)
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")
    C = 1
    H = 64
    W = 2048

    interleaved_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.L1,
    )
    sharded_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    out_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config

    x = torch.randn((N, C, H, W)).bfloat16()

    xt = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            dtype,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(
            device,
            interleaved_mem_config,
        )
    )

    if in_sharded:
        xt = ttl.tensor.interleaved_to_sharded(
            xt,
            grid_size,
            [N * C * H, W // (grid_size[0] * grid_size[1])],
            ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED,
            ttl.tensor.ShardOrientation.COL_MAJOR,
        )

    yt = ttl.tensor.reduce(
        xt,
        ttl.tensor.ReduceOpMath.MAX,
        ttl.tensor.ReduceOpDim.H,
        1.0,
        output_mem_config=out_mem_config,
    )

    if out_sharded:
        yt = ttl.tensor.sharded_to_interleaved(
            yt,
            interleaved_mem_config,
        )

    tt_got_back = yt.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()[:, :, :1, :]

    y = torch.max(x, 2, True)[0]

    if dtype == ttl.tensor.DataType.BFLOAT16:
        passing, output = comp_equal(y, tt_got_back)
    else:
        passing, output = comp_pcc(y, tt_got_back, 0.999)
    logger.info(output)

    assert passing


@pytest.mark.parametrize("in0_sharded", [True, False], ids=["in0_sharded", "in0_unsharded"])
@pytest.mark.parametrize("out_sharded", [True, False], ids=["out_sharded", "out_unsharded"])
@pytest.mark.parametrize("M", [32])
@pytest.mark.parametrize("N", [1024])
@pytest.mark.parametrize("activations_dtype", [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B])
@pytest.mark.parametrize("weights_dtype", [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B])
def test_sharded_matmul_1d_in0(
    device, in0_sharded, out_sharded, M, N, activations_dtype, weights_dtype, function_level_defaults
):
    grid_size = (8, 4)
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")
    if activations_dtype != weights_dtype and is_wormhole_b0():
        pytest.skip("WH does not work with mixed precision")
    num_cores = grid_size[0] * grid_size[1]
    K = 2048
    in0_shape = [1, 1, M, K]
    in1_shape = [1, 1, K, N]
    bias_shape = [1, 1, 1, N]

    interleaved_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.L1,
    )
    sharded_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()
    bias = torch.randn(bias_shape).bfloat16().float()

    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=interleaved_mem_config, tt_dtype=weights_dtype)
    bias_t = pad_by_zero(bias, device, tt_memory_config=interleaved_mem_config, tt_dtype=weights_dtype)[0]

    output_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config

    if in0_sharded:
        in0_t = ttl.tensor.interleaved_to_sharded(
            in0_t,
            grid_size,
            [M, K // num_cores],
            ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED,
            ttl.tensor.ShardOrientation.ROW_MAJOR,
        )

    program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=1,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )
    output_t = ttl.operations.primary.matmul_1d(
        in0_t,
        in1_t,
        bias=bias_t,
        program_config=program_config,
        output_mem_config=output_mem_config,
        output_dtype=activations_dtype,
    )
    if out_sharded:
        output_t = ttl.tensor.sharded_to_interleaved(output_t, interleaved_mem_config)
    pt_out = in0 @ in1 + bias

    tt_out = tt2torch_tensor(output_t)

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing


@pytest.mark.parametrize("in0_sharded", [True, False], ids=["in0_sharded", "in0_unsharded"])
@pytest.mark.parametrize("in1_sharded", [True, False], ids=["in1_sharded", "in1_unsharded"])
@pytest.mark.parametrize("out_sharded", [True, False], ids=["out_sharded", "out_unsharded"])
@pytest.mark.parametrize(
    "B, H, M, K, N, out_subblock_h, out_subblock_w", [[12, 16, 384, 64, 384, 1, 6], [12, 16, 384, 384, 64, 4, 2]]
)
@pytest.mark.parametrize("activations_dtype", [ttl.tensor.DataType.BFLOAT8_B])
def test_sharded_matmul_no_mcast(
    device,
    in0_sharded,
    in1_sharded,
    out_sharded,
    B,
    H,
    M,
    K,
    N,
    out_subblock_h,
    out_subblock_w,
    activations_dtype,
    function_level_defaults,
):
    grid_size = (12, 8)
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")
    num_cores = grid_size[0] * grid_size[1]
    in0_shape = [B, H, M, K]
    in1_shape = [B, H, K, N]

    interleaved_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.DRAM,
    )
    sharded_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()

    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype)

    output_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config

    if in0_sharded:
        in0_t = ttl.tensor.interleaved_to_sharded(
            in0_t,
            grid_size,
            [B * H * M // num_cores, K],
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            ttl.tensor.ShardOrientation.COL_MAJOR,
        )
    if in1_sharded:
        in1_t = ttl.tensor.interleaved_to_sharded(
            in1_t,
            grid_size,
            [B * H * K // num_cores, N],
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            ttl.tensor.ShardOrientation.COL_MAJOR,
        )

    program_config = ttl.operations.primary.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=K // 32,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=B * H * M // num_cores // 32,
        per_core_N=N // 32,
    )

    output_t = ttl.operations.primary.matmul(
        in0_t,
        in1_t,
        program_config=program_config,
        output_mem_config=output_mem_config,
        output_dtype=activations_dtype,
    )
    if out_sharded:
        output_t = ttl.tensor.sharded_to_interleaved(output_t, interleaved_mem_config)

    pt_out = in0 @ in1

    tt_out = tt2torch_tensor(output_t)

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing


@pytest.mark.parametrize("in0_shape, grid_size", [([12, 16, 384, 64], (12, 8)), ([1, 32, 32, 64], (8, 4))])
@pytest.mark.parametrize("in0_sharded, out_sharded", [[True, True], [False, False]], ids=["sharded", "unsharded"])
@pytest.mark.parametrize("activations_dtype", [ttl.tensor.DataType.BFLOAT8_B])
def test_sharded_concat_heads(
    device,
    in0_shape,
    grid_size,
    in0_sharded,
    out_sharded,
    activations_dtype,
    function_level_defaults,
):
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")
    num_cores = grid_size[0] * grid_size[1]
    B = in0_shape[0]
    num_heads = in0_shape[1]
    seq_len = in0_shape[2]
    head_dim = in0_shape[3]

    in0_shape = [B, num_heads, seq_len, head_dim]

    interleaved_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.DRAM,
    )
    sharded_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()

    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype)

    output_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config

    if in0_sharded:
        in0_t = ttl.tensor.interleaved_to_sharded(
            in0_t,
            grid_size,
            [B * num_heads * seq_len // num_cores, head_dim],
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            ttl.tensor.ShardOrientation.COL_MAJOR,
        )

    output_t = ttl.tensor.nlp_concat_heads(
        in0_t,
        output_mem_config=output_mem_config,
    )
    if out_sharded:
        output_t = ttl.tensor.sharded_to_interleaved(output_t, interleaved_mem_config)

    pt_out = torch.transpose(in0, -3, -2).reshape([B, 1, seq_len, num_heads * head_dim])

    tt_out = tt2torch_tensor(output_t)

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing


@skip_for_wormhole_b0("WH ND hang, see issue #4392")
@pytest.mark.parametrize(
    "input_shape, input_layout, input_shard_grid,  input_shard_shape, input_shard_orientation, input_sharding_scheme, output_shard_grid, output_shard_shape, output_shard_orientation, output_sharding_scheme ",
    [
        (
            [1, 1, 64, 64],
            ttl.tensor.Layout.TILE,
            (0, 1),
            (64, 32),
            ttl.tensor.ShardOrientation.ROW_MAJOR,
            ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED,
            (0, 1),
            (32, 64),
            ttl.tensor.ShardOrientation.ROW_MAJOR,
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
        ),
        (
            [1, 1, 128, 64],
            ttl.tensor.Layout.TILE,
            (0, 1),
            (64, 64),
            ttl.tensor.ShardOrientation.ROW_MAJOR,
            ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
            (0, 7),
            (32, 32),
            ttl.tensor.ShardOrientation.ROW_MAJOR,
            ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        ),
        (
            [1, 1, 32, 128],
            ttl.tensor.Layout.TILE,
            (0, 3),
            (32, 32),
            ttl.tensor.ShardOrientation.ROW_MAJOR,
            ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
            (0, 1),
            (32, 64),
            ttl.tensor.ShardOrientation.ROW_MAJOR,
            ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        ),
        (
            [1, 1, 32, 2304],
            ttl.tensor.Layout.TILE,
            (0, 7),
            (32, 288),
            ttl.tensor.ShardOrientation.ROW_MAJOR,
            ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
            (0, 1),
            (32, 1152),
            ttl.tensor.ShardOrientation.ROW_MAJOR,
            ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        ),
        (
            [1, 1, 32, 16],
            ttl.tensor.Layout.ROW_MAJOR,
            (0, 0),
            (32, 16),
            ttl.tensor.ShardOrientation.ROW_MAJOR,
            ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
            (0, 1),
            (16, 16),
            ttl.tensor.ShardOrientation.ROW_MAJOR,
            ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        ),
        (
            [1, 1, 32, 8192],
            ttl.tensor.Layout.TILE,
            (7, 7),
            (32, 128),
            ttl.tensor.ShardOrientation.ROW_MAJOR,
            ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
            (0, 7),
            (32, 1024),
            ttl.tensor.ShardOrientation.ROW_MAJOR,
            ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        ),
    ],
)
def test_reshard(
    device,
    input_shape,
    input_layout,
    input_shard_grid,
    input_shard_shape,
    input_shard_orientation,
    input_sharding_scheme,
    output_shard_grid,
    output_shard_shape,
    output_shard_orientation,
    output_sharding_scheme,
):
    compute_grid = ttl.tensor.CoreCoord(input_shard_grid[0], input_shard_grid[1])
    input_shard_grid = ttl.tensor.CoreRangeSet({ttl.tensor.CoreRange(ttl.tensor.CoreCoord(0, 0), compute_grid)})

    compute_grid = ttl.tensor.CoreCoord(output_shard_grid[0], output_shard_grid[1])
    output_shard_grid = ttl.tensor.CoreRangeSet({ttl.tensor.CoreRange(ttl.tensor.CoreCoord(0, 0), compute_grid)})
    output_shard_spec = ttl.tensor.ShardSpec(output_shard_grid, output_shard_shape, output_shard_orientation, False)
    output_mem_config = ttl.tensor.MemoryConfig(output_sharding_scheme, ttl.tensor.BufferType.L1, output_shard_spec)
    tt_dtype = ttl.tensor.DataType.BFLOAT16

    dram_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.DRAM,
    )
    torch_tensor = torch.randn(input_shape).bfloat16()
    tt_tensor_sharded = ttl.tensor.Tensor(torch_tensor, tt_dtype).to(input_layout)
    tt_tensor_sharded = tt_tensor_sharded.to(device, dram_memory_config)
    tt_tensor_sharded = ttl.tensor.interleaved_to_sharded(
        tt_tensor_sharded,
        input_shard_grid,
        input_shard_shape,
        input_sharding_scheme,
        input_shard_orientation,
        output_dtype=tt_dtype,
    )

    tt_tensor_reshard = ttl.tensor.reshard(tt_tensor_sharded, output_mem_config)

    tt_tensor_interleaved = ttl.tensor.sharded_to_interleaved(
        tt_tensor_reshard,
        dram_memory_config,
    )

    tt_tensor_interleaved = tt_tensor_interleaved.cpu().to(ttl.tensor.Layout.ROW_MAJOR)
    torch_tensor_after_round_trip = tt_tensor_interleaved.to_torch()

    assert torch_tensor.dtype == torch_tensor_after_round_trip.dtype
    assert torch_tensor.shape == torch_tensor_after_round_trip.shape
    passing = torch.allclose(torch_tensor, torch_tensor_after_round_trip)
    if not passing:
        print(torch_tensor)
        print(torch_tensor_after_round_trip)
    assert passing


@pytest.mark.parametrize(
    "input_shape, shard_scheme",
    [
        ([1, 1, 128, 256], ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED),
        ([1, 1, 128, 256], ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED),
        ([1, 1, 128, 256], ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED),
    ],
)
@pytest.mark.parametrize(
    "shard_orientation", [ttl.tensor.ShardOrientation.ROW_MAJOR, ttl.tensor.ShardOrientation.COL_MAJOR]
)
def test_sharded_to_from_l1(device, input_shape, shard_scheme, shard_orientation):
    input_dtype = ttl.tensor.DataType.BFLOAT16
    output_dtype = ttl.tensor.DataType.BFLOAT16

    assert input_shape[-2] % 32 == 0
    assert input_shape[-1] % 32 == 0
    if shard_scheme == ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED:
        grid_x = input_shape[-2] // 32
        grid_y = 1
        shard_shape = [input_shape[-2] // grid_x, input_shape[-1] // grid_y]
    elif shard_scheme == ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED:
        grid_x = input_shape[-1] // 32
        grid_y = 1
        shard_shape = [input_shape[-2] // grid_y, input_shape[-1] // grid_x]
    elif shard_scheme == ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED:
        grid_x = input_shape[-1] // 32
        grid_y = input_shape[-2] // 32
        shard_shape = [input_shape[-2] // grid_y, input_shape[-1] // grid_x]
    else:
        assert False, f"Unsupported {shard_scheme}"

    shard_grid = ttl.tensor.CoreRangeSet(
        {ttl.tensor.CoreRange(ttl.tensor.CoreCoord(0, 0), ttl.tensor.CoreCoord(grid_x - 1, grid_y - 1))}
    )
    shard_halo = False
    shard_spec = ttl.tensor.ShardSpec(shard_grid, shard_shape, shard_orientation, shard_halo)
    mem_config = ttl.tensor.MemoryConfig(shard_scheme, ttl.tensor.BufferType.L1, shard_spec)

    volume = input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]
    golden = torch.arange(volume).reshape(input_shape).bfloat16()
    ttl_golden = ttl.tensor.Tensor(golden.reshape(-1).tolist(), golden.shape, input_dtype, ttl.tensor.Layout.ROW_MAJOR)

    ## TEST to/from ##
    ttl_device = ttl_golden.to(device, mem_config)
    result = ttl_device.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    passing, output = comp_equal(result, golden)
    assert passing
