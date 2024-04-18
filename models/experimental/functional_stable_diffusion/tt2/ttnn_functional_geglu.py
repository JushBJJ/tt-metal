# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import math
import tt_lib as ttl
from models.experimental.functional_stable_diffusion.tt2.ttnn_functional_utility_functions import (
    find_max_subblock,
    determine_largest_subblock_size,
)


def ttnn_to_torch(input):
    input = ttnn.to_layout(input, ttnn.ROW_MAJOR_LAYOUT)
    input = ttnn.from_device(input)
    input = ttnn.to_torch(input)
    return input


def split_linear_params(params):
    dim = -1
    device = params.proj.weight.device()
    memory_config = ttnn.DRAM_MEMORY_CONFIG

    weight = ttnn_to_torch(params.proj.weight)
    bias = ttnn_to_torch(params.proj.bias)

    proj_weight, gate_weight = torch.split(weight, weight.shape[dim] // 2, dim=dim)
    proj_bias, gate_bias = torch.split(bias, bias.shape[dim] // 2, dim=dim)

    while len(proj_weight.shape) < 4:
        proj_weight = proj_weight.unsqueeze(0)
    proj_weight = ttnn.from_torch(proj_weight, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    proj_weight = ttnn.to_device(proj_weight, device, memory_config=memory_config)

    while len(gate_weight.shape) < 4:
        gate_weight = gate_weight.unsqueeze(0)
    gate_weight = ttnn.from_torch(gate_weight, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    gate_weight = ttnn.to_device(gate_weight, device, memory_config=memory_config)

    while len(proj_bias.shape) < 4:
        proj_bias = proj_bias.unsqueeze(0)
    proj_bias = ttnn.from_torch(proj_bias, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    proj_bias = ttnn.to_device(proj_bias, device, memory_config=memory_config)

    while len(gate_bias.shape) < 4:
        gate_bias = gate_bias.unsqueeze(0)
    gate_bias = ttnn.from_torch(gate_bias, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    gate_bias = ttnn.to_device(gate_bias, device, memory_config=memory_config)

    params.proj.proj_weight = proj_weight
    params.proj.gate_weight = gate_weight
    params.proj.proj_bias = proj_bias
    params.proj.gate_bias = gate_bias

    del params.proj.weight
    del params.proj.bias
    return params


class geglu:
    def __init__(self, device, parameters):
        self.device = device
        parameters = split_linear_params(parameters)
        self.parameters = parameters
        self.grid_sizes = {8192: (8, 5), 2048: (8, 5), 512: (8, 8), 128: (4, 8)}
        self.out_subblock_hs = {8192: 8, 2048: 8, 512: 2, 128: 1}

        self.l1_interleaved_memory_config = ttnn.experimental.tensor.MemoryConfig(
            memory_layout=ttnn.experimental.tensor.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttnn.experimental.tensor.BufferType.L1,
        )
        self.block_sharded_memory_config = ttnn.experimental.tensor.MemoryConfig(
            memory_layout=ttnn.experimental.tensor.TensorMemoryLayout.BLOCK_SHARDED,
            buffer_type=ttnn.experimental.tensor.BufferType.L1,
        )
        self.compute_kernel_config = ttnn.experimental.tensor.WormholeComputeKernelConfig(
            math_fidelity=ttnn.experimental.tensor.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

    def __call__(self, config, hidden_states):
        # TODO: Output sharded once https://github.com/tenstorrent/tt-metal/issues/6775 is fixed
        interleaved_output = False
        size = hidden_states.shape[-2]
        grid_size = self.grid_sizes[size]
        M, K, N = hidden_states.shape[-2], hidden_states.shape[-1], self.parameters.proj.proj_weight.shape[-1]
        if not hidden_states.is_sharded():
            hidden_states = ttnn.experimental.tensor.interleaved_to_sharded(
                hidden_states,
                grid_size,
                [M // grid_size[0], K // grid_size[1]],
                ttnn.experimental.tensor.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.experimental.tensor.ShardOrientation.COL_MAJOR,
            )
        in0_block_h = M // grid_size[0] // 32
        in0_block_w = K // grid_size[1] // 32
        out_block_h = math.ceil(M / grid_size[0] / 32)
        out_block_w = math.ceil(N / grid_size[1] / 32)
        out_subblock_h, out_subblock_w = determine_largest_subblock_size(out_block_h, out_block_w)
        program_config = ttnn.experimental.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=in0_block_w,
            out_subblock_h=out_subblock_h,
            out_subblock_w=out_subblock_w,
            per_core_M=out_block_h,
            per_core_N=out_block_w,
            transpose_mcast=True,
            fused_activation=None,
        )
        proj = ttnn.experimental.operations.primary.matmul(
            hidden_states,
            self.parameters.proj.proj_weight,
            bias=self.parameters.proj.proj_bias,
            program_config=program_config,
            output_mem_config=self.l1_interleaved_memory_config
            if interleaved_output
            else self.block_sharded_memory_config,
            output_dtype=ttnn.experimental.tensor.DataType.BFLOAT8_B,
            compute_kernel_config=self.compute_kernel_config,
        )
        if interleaved_output:
            proj = ttnn.experimental.tensor.interleaved_to_sharded(
                proj,
                grid_size,
                [proj.shape[-2] // grid_size[0], proj.shape[-1] // grid_size[1]],
                ttnn.experimental.tensor.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.experimental.tensor.ShardOrientation.COL_MAJOR,
            )
        if hidden_states.shape[-2] == 8192:
            proj = ttnn.reallocate(proj)

        program_config = ttnn.experimental.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=in0_block_w,
            out_subblock_h=out_subblock_h,
            out_subblock_w=out_subblock_w,
            per_core_M=out_block_h,
            per_core_N=out_block_w,
            transpose_mcast=True,
            fused_activation=[ttnn.experimental.tensor.FusibleActivation.GELU, True],
        )
        gate = ttnn.experimental.operations.primary.matmul(
            hidden_states,
            self.parameters.proj.gate_weight,
            bias=self.parameters.proj.gate_bias,
            program_config=program_config,
            output_mem_config=self.l1_interleaved_memory_config
            if interleaved_output
            else self.block_sharded_memory_config,
            output_dtype=ttnn.experimental.tensor.DataType.BFLOAT8_B,
            compute_kernel_config=self.compute_kernel_config,
        )
        if interleaved_output:
            gate = ttnn.experimental.tensor.interleaved_to_sharded(
                gate,
                grid_size,
                [gate.shape[-2] // grid_size[0], gate.shape[-1] // grid_size[1]],
                ttnn.experimental.tensor.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.experimental.tensor.ShardOrientation.COL_MAJOR,
            )
        if hidden_states.shape[-2] == 8192:
            gate = ttnn.reallocate(gate)
        ret = ttnn.mul(proj, gate, memory_config=gate.memory_config())
        ttnn.deallocate(proj)
        ttnn.deallocate(gate)
        return ret
