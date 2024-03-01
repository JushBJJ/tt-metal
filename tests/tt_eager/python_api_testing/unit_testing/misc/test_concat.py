# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
from pathlib import Path
import sys

import torch

import tt_lib as ttl
from models.utility_functions import print_diff_argmax
import pytest
from loguru import logger

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_pcc,
    comp_equal,
)


@pytest.mark.parametrize(
    "memcfg",
    (
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize("dtype", ((ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B)))
@pytest.mark.parametrize("nChannels", ((2, 3, 4)))
def test_tile_simple_concat(memcfg, dtype, nChannels, device, function_level_defaults):
    input_shape = torch.Size([nChannels, nChannels, 32, 32])
    x = torch.arange(0, input_shape.numel()).reshape(input_shape).bfloat16()

    y = (1 + torch.arange(0, input_shape.numel()).reshape(input_shape)).bfloat16()

    xtt = (
        ttl.tensor.Tensor(x, dtype).to(ttl.tensor.Layout.TILE).to(device, memcfg),
        ttl.tensor.Tensor(y, dtype).to(ttl.tensor.Layout.TILE).to(device, memcfg),
    )

    dim = 3
    output_shape = list(x.shape)
    output_shape[3] = y.shape[3] + x.shape[3]
    tt_cpu = torch.concat([x, y], dim)
    assert tt_cpu.shape == torch.Size(output_shape)

    tt = ttl.tensor.concat(xtt, dim)
    assert list(tt.shape()) == output_shape
    xtt_data = tt.cpu().to(ttl.tensor.Layout.ROW_MAJOR)
    tt_dev = xtt_data.to_torch()

    if dtype == ttl.tensor.DataType.BFLOAT8_B:
        passing, output = comp_pcc(tt_cpu, tt_dev)
    else:
        passing, output = comp_equal(tt_cpu, tt_dev)
    logger.info(output)
    assert passing


# @pytest.mark.skip(reason="For Stable Diffusion Sizes only")
@pytest.mark.parametrize(
    "shape_a, shape_b, dim",
    (
        ((1, 1, 32, 32), (1, 1, 32, 32), 3),
        ((1, 1, 32, 64), (1, 1, 32, 128), 3),
        ((1, 1, 32, 128), (1, 1, 32, 64), 3),
        ((1, 1, 64, 128), (1, 1, 64, 256), 3),
        ((1, 32, 32, 32), (1, 32, 32, 32), 2),
        ((1, 1, 32, 32), (1, 1, 32, 32), 3),
        ((2, 4, 32, 1280), (2, 4, 32, 1280), 3),
        # SD Shapes
        ((2, 1280, 4, 4), (2, 1280, 4, 4), 1),
        ((2, 640, 32, 32), (2, 320, 32, 32), 1),
        ((2, 1280, 8, 8), (2, 1280, 8, 8), 1),
        ((2, 640, 16, 16), (2, 640, 16, 16), 1),
        ((2, 320, 32, 32), (2, 320, 32, 32), 1),
    ),
)
def test_tile_concat(shape_a, shape_b, dim, device, function_level_defaults):
    shape_a = torch.Size(shape_a)

    x = torch.arange(0, shape_a.numel()).reshape(shape_a).to(torch.bfloat16)

    shape_b = torch.Size(shape_b)
    y = torch.arange(0, shape_b.numel()).reshape(shape_b).to(torch.bfloat16)

    xtt = (
        ttl.tensor.Tensor(
            x,
            ttl.tensor.DataType.BFLOAT16,
        ).to(device),
        ttl.tensor.Tensor(
            y,
            ttl.tensor.DataType.BFLOAT16,
        ).to(device),
    )

    output_shape = list(x.shape)
    output_shape[dim] = y.shape[dim] + x.shape[dim]
    tt_cpu = torch.concat([x, y], dim)
    assert tt_cpu.shape == torch.Size(output_shape)

    tt = ttl.tensor.concat([xtt[0], xtt[1]], dim)
    assert list(tt.shape()) == output_shape
    tt_dev = tt.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch().to(torch.bfloat16)

    passing, output = comp_equal(tt_cpu, tt_dev)
    logger.info(output)
    assert passing


@pytest.mark.parametrize(
    "shapes, dim",
    (
        (((1, 2, 64, 64),), -1),
        (((1, 1, 64, 64), (1, 1, 128, 64)), -2),
        (((1, 1, 32, 128), (1, 1, 32, 64), (1, 1, 32, 256)), -1),
        (((2, 4, 32, 1280), (2, 3, 32, 1280), (2, 5, 32, 1280), (2, 8, 32, 1280)), 1),
    ),
)
def test_multi_input_concat(shapes, dim, device, function_level_defaults):
    inputs = []
    tt_inputs = []
    for i in range(len(shapes)):
        shape = torch.Size(shapes[i])
        inputs.append(i + torch.arange(0, shape.numel()).reshape(shape).to(torch.bfloat16))
        tt_inputs.append(
            ttl.tensor.Tensor(
                inputs[i],
                ttl.tensor.DataType.BFLOAT16,
            ).to(device)
        )

    tt_cpu = torch.concat(inputs, dim)

    tt = ttl.tensor.concat(tt_inputs, dim)

    tt_dev = tt.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch().to(torch.bfloat16)

    passing, output = comp_equal(tt_cpu, tt_dev)
    logger.info(output)
    assert passing


@pytest.mark.parametrize(
    "input_shape, shard_shape, output_shard_shape, shard_grid",
    (
        (
            (1, 1, 16, 16),
            (8, 16),
            (8, 32),
            ttl.tensor.CoreRangeSet({ttl.tensor.CoreRange(ttl.tensor.CoreCoord(0, 0), ttl.tensor.CoreCoord(0, 1))}),
        ),
        (
            (1, 1, 160, 32),
            (80, 32),
            (80, 64),
            ttl.tensor.CoreRangeSet({ttl.tensor.CoreRange(ttl.tensor.CoreCoord(0, 0), ttl.tensor.CoreCoord(0, 1))}),
        ),
        (
            (1, 1, 5280, 32),
            (80, 32),
            (80, 64),
            ttl.tensor.CoreRangeSet(
                {
                    ttl.tensor.CoreRange(ttl.tensor.CoreCoord(0, 0), ttl.tensor.CoreCoord(11, 4)),
                    ttl.tensor.CoreRange(ttl.tensor.CoreCoord(0, 5), ttl.tensor.CoreCoord(5, 5)),
                }
            ),
        ),
    ),
)
def test_sharded_concat(input_shape, shard_shape, output_shard_shape, shard_grid, device):
    num_inputs = 2
    inputs = []
    tt_inputs = []
    input_shard_scheme = ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED
    input_shard_orientation = ttl.tensor.ShardOrientation.ROW_MAJOR
    shard_spec = ttl.tensor.ShardSpec(shard_grid, shard_shape, input_shard_orientation, False)
    sharded_mem_config = ttl.tensor.MemoryConfig(input_shard_scheme, ttl.tensor.BufferType.L1, shard_spec)

    output_shard_spec = ttl.tensor.ShardSpec(shard_grid, output_shard_shape, input_shard_orientation, False)
    output_sharded_mem_config = ttl.tensor.MemoryConfig(input_shard_scheme, ttl.tensor.BufferType.L1, output_shard_spec)

    total_elements = input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]
    for i in range(num_inputs):
        shape = torch.Size(input_shape)
        inputs.append(i * 1000 * total_elements + torch.arange(0, shape.numel()).reshape(shape).to(torch.bfloat16))
        tt_inputs.append(
            ttl.tensor.Tensor(
                inputs[i],
                ttl.tensor.DataType.BFLOAT16,
            )
            .to(ttl.tensor.Layout.ROW_MAJOR)
            .to(device, sharded_mem_config)
        )

    dram_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.DRAM,
    )

    dim = 3
    tt_output_interleaved = ttl.tensor.concat(tt_inputs, dim, output_sharded_mem_config)

    tt_dev = tt_output_interleaved.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch().to(torch.bfloat16)
    tt_cpu = torch.concat(inputs, dim)

    passing, output = comp_equal(tt_cpu, tt_dev)
    logger.info(output)
    assert passing
