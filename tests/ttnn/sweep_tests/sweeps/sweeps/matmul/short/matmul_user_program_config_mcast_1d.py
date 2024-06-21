# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from loguru import logger
import enum

import torch

import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc
from models.utility_functions import torch_random


class TensorMemoryConfigs(enum.Enum):
    CUSTOM_MEMORY_CONFIG = enum.auto()


core_grid = ttnn.CoreCoord(8, 7)
parameters = {
    "matmul_specs": [
        ########################################
        # TESTS: in0 mcast grid != output grid #
        ########################################
        # Matmul 1D mcast in0: in0 grid == output grid
        (
            (1,),
            (64, 32 * 96, 32 * 96),
            ttnn.experimental.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(8, 4),
                in0_block_w=1,
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=2,
                per_core_N=3,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            ),
            ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                buffer_type=ttnn.BufferType.L1,
                shard_spec=ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
                    (64, 96),
                    ttnn.ShardOrientation.ROW_MAJOR,
                    False,
                ),
            ),
        ),
        # Matmul 1D mcast in0: in0 grid < output grid
        (
            (1,),
            (64, 28 * 96, 35 * 96),
            ttnn.experimental.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(8, 5),
                in0_block_w=1,
                out_subblock_h=1,
                out_subblock_w=3,
                per_core_M=2,
                per_core_N=3,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            ),
            ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                buffer_type=ttnn.BufferType.L1,
                shard_spec=ttnn.ShardSpec(
                    ttnn.experimental.tensor.num_cores_to_core_range_set(28, core_grid, row_wise=True),
                    (64, 96),
                    ttnn.ShardOrientation.ROW_MAJOR,
                    False,
                ),
            ),
        ),
        # Matmul 1D mcast in0: in0 grid > output grid
        (
            (1,),
            (64, 35 * 96, 28 * 96),
            ttnn.experimental.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(8, 5),
                in0_block_w=1,
                out_subblock_h=1,
                out_subblock_w=3,
                per_core_M=2,
                per_core_N=3,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            ),
            ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                buffer_type=ttnn.BufferType.L1,
                shard_spec=ttnn.ShardSpec(
                    ttnn.experimental.tensor.num_cores_to_core_range_set(35, core_grid, row_wise=True),
                    (64, 96),
                    ttnn.ShardOrientation.ROW_MAJOR,
                    False,
                ),
            ),
        ),
        # Matmul 1D mcast in0: in0 grid.y == output grid.y but in0 grid.x < output grid.x and output grid.x isn't full row; tests mcast logic for num_active_cores
        (
            (1,),
            (64, 28 * 96, 30 * 96),
            ttnn.experimental.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(8, 4),
                in0_block_w=1,
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=2,
                per_core_N=3,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            ),
            ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                buffer_type=ttnn.BufferType.L1,
                shard_spec=ttnn.ShardSpec(
                    ttnn.experimental.tensor.num_cores_to_core_range_set(28, core_grid, row_wise=True),
                    (64, 96),
                    ttnn.ShardOrientation.ROW_MAJOR,
                    False,
                ),
            ),
        ),
        # Matmul 1D mcast in0: in0 grid.y == output grid.y but in0 grid.x > output grid.x and in0 grid.x isn't full row; tests mcast logic for num_active_cores
        (
            (1,),
            (64, 30 * 96, 28 * 96),
            ttnn.experimental.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(8, 4),
                in0_block_w=1,
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=2,
                per_core_N=3,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            ),
            ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                buffer_type=ttnn.BufferType.L1,
                shard_spec=ttnn.ShardSpec(
                    ttnn.experimental.tensor.num_cores_to_core_range_set(30, core_grid, row_wise=True),
                    (64, 96),
                    ttnn.ShardOrientation.ROW_MAJOR,
                    False,
                ),
            ),
        ),
        #############################################
        # TESTS: Single core output grid edge cases #
        #############################################
        # Matmul 1D mcast in0: single-core in0 grid and output grid
        (
            (1,),
            (64, 96, 128),
            ttnn.experimental.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(1, 1),
                in0_block_w=1,
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=2,
                per_core_N=4,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            ),
            ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                buffer_type=ttnn.BufferType.L1,
                shard_spec=ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
                    (64, 96),
                    ttnn.ShardOrientation.ROW_MAJOR,
                    False,
                ),
            ),
        ),
        # Matmul 1D mcast in0: multi-core in0 grid and single-core output grid
        (
            (1,),
            (64, 5 * 96, 128),
            ttnn.experimental.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(5, 1),
                in0_block_w=1,
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=2,
                per_core_N=4,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            ),
            ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                buffer_type=ttnn.BufferType.L1,
                shard_spec=ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(4, 0))}),
                    (64, 96),
                    ttnn.ShardOrientation.ROW_MAJOR,
                    False,
                ),
            ),
        ),
        # Matmul 1D mcast in1: single-core in1 grid and output grid
        (
            (1,),
            (64, 96, 128),
            ttnn.experimental.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(1, 1),
                in0_block_w=1,
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=2,
                per_core_N=4,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=False,
            ),
            ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                buffer_type=ttnn.BufferType.L1,
                shard_spec=ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
                    (64, 96),
                    ttnn.ShardOrientation.ROW_MAJOR,
                    False,
                ),
            ),
        ),
    ],
    "batch_matrix_multiply": [False],
    "in0_block_w": [
        1,
        3,
    ],  # Used to override in0_block_w in program config (1: loop along in0 shard width; 2: no looping along in0 shard width)
    "input_a_memory_config": [TensorMemoryConfigs.CUSTOM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    "output_memory_config": [TensorMemoryConfigs.CUSTOM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG],
    "input_a_dtype": [ttnn.bfloat16],
    "input_b_dtype": [ttnn.bfloat8_b],
    "output_dtype": [ttnn.bfloat16],
    "input_layout": [ttnn.TILE_LAYOUT],
    "compute_kernel_config": [None],
}


def skip(**_) -> Tuple[bool, Optional[str]]:
    return False, None


def is_expected_to_fail(**_) -> Tuple[bool, Optional[str]]:
    return False, None


def run(
    matmul_specs,
    batch_matrix_multiply,
    in0_block_w,
    input_a_memory_config,
    input_b_memory_config,
    output_memory_config,
    input_a_dtype,
    input_b_dtype,
    output_dtype,
    input_layout,
    compute_kernel_config,
    *,
    device,
) -> Tuple[bool, Optional[str]]:
    (
        batch_sizes,
        input_shapes,
        program_config,
        input_a_custom_memory_config,
    ) = matmul_specs

    program_config.in0_block_w = in0_block_w

    if input_a_memory_config == TensorMemoryConfigs.CUSTOM_MEMORY_CONFIG:
        input_a_memory_config = input_a_custom_memory_config

    if output_memory_config == TensorMemoryConfigs.CUSTOM_MEMORY_CONFIG:
        output_memory_config = (
            ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG if program_config.mcast_in0 else ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
        )

    (m_size, k_size, n_size) = input_shapes
    input_shape_a = (*batch_sizes, m_size, k_size)
    input_shape_b = (k_size, n_size)
    if batch_matrix_multiply:
        input_shape_b = (*batch_sizes, k_size, n_size)

    input_a_layout = input_layout
    input_b_layout = input_layout

    torch_input_tensor_a = torch_random(input_shape_a, -0.1, 0.1, dtype=torch.float32)
    torch_input_tensor_b = torch_random(input_shape_b, -0.1, 0.1, dtype=torch.float32)
    torch_output_tensor = torch.matmul(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        layout=input_a_layout,
        dtype=input_a_dtype,
        device=device,
        memory_config=input_a_memory_config,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        layout=input_b_layout,
        dtype=input_b_dtype,
        device=device,
        memory_config=input_b_memory_config,
    )

    output_tensor = ttnn.matmul(
        input_tensor_a,
        input_tensor_b,
        memory_config=output_memory_config,
        dtype=output_dtype,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )
    output_tensor = ttnn.to_torch(output_tensor)

    expected_pcc = 0.99
    return check_with_pcc(torch_output_tensor, output_tensor, expected_pcc)
