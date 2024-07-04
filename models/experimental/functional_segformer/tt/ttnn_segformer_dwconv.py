# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch


def torch_to_ttnn(input, device, layout=ttnn.TILE_LAYOUT):
    input = ttnn.from_torch(input, ttnn.bfloat16)
    input = ttnn.to_layout(input, layout)
    input = ttnn.to_device(input, device)
    return input


def ttnn_to_torch(input):
    input = ttnn.to_layout(input, ttnn.ROW_MAJOR_LAYOUT)
    input = ttnn.from_device(input)
    input = ttnn.to_torch(input)
    return input


class TtSegformerDWConv:
    def __init__(self, parameters, model):
        super().__init__()
        self.dwconv = model.dwconv

    def __call__(self, hidden_states: ttnn.Tensor, height: int, width: int, device, dim, parameters):
        batch_size, seq_len, num_channels = hidden_states.shape
        hidden_states = ttnn.permute(hidden_states, (0, 2, 1))
        hidden_states = ttnn.from_device(hidden_states)
        hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
        hidden_states = ttnn.reshape(hidden_states, (batch_size, hidden_states.shape[0], hidden_states.shape[1]))
        hidden_states = ttnn.reshape(hidden_states, (batch_size, num_channels, height, width))
        hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)
        hidden_states = ttnn.to_device(hidden_states, device=device)

        hidden_states = ttnn.permute(hidden_states, (0, 2, 3, 1))
        hidden_states, out_height, out_width, weight, bias = ttnn.conv2d(
            input_tensor=hidden_states,
            weight_tensor=parameters.dwconv.weight,
            in_channels=dim,
            out_channels=dim,
            device=device,
            bias_tensor=parameters.dwconv.bias,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            batch_size=1,
            groups=dim,
            input_height=height,
            input_width=width,
            conv_config=ttnn.Conv2dConfig(
                dtype=ttnn.bfloat16,
                weights_dtype=ttnn.bfloat16,
                math_fidelity=ttnn.MathFidelity.LoFi,
                height_sharding=True,
                input_channels_alignment=(16 if False else 32),
                deallocate_activation=False,
                fp32_dest_acc_enabled=False,
                packer_l1_accum_enabled=False,
            ),
            conv_op_cache={},
        )
        hidden_states = ttnn.from_device(hidden_states)
        hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.ROW_MAJOR_LAYOUT)

        hidden_states = ttnn.reshape(hidden_states, (batch_size, out_height, out_width, num_channels))
        del out_height, out_width, weight, bias
        hidden_states = ttnn.from_device(hidden_states)
        hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.TILE_LAYOUT)
        hidden_states = ttnn.to_device(hidden_states, device=device)

        hidden_states = ttnn.permute(hidden_states, (0, 3, 1, 2))

        hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.ROW_MAJOR_LAYOUT)
        hidden_states = ttnn.from_device(hidden_states)

        hidden_states = ttnn.reshape(
            hidden_states,
            (hidden_states.shape[0], hidden_states.shape[1], hidden_states.shape[2] * hidden_states.shape[3]),
        )
        hidden_states = ttnn.to_device(hidden_states, device)
        hidden_states = ttnn.from_device(hidden_states)
        hidden_states = ttnn.to_device(hidden_states, device)
        hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.TILE_LAYOUT)
        # hidden_states=ttnn.to_memory_config(hidden_states,memory_config=ttnn.DRAM_MEMORY_CONFIG)

        hidden_states = ttnn.permute(hidden_states, (0, 2, 1))
        hidden_states = ttnn.reshape(hidden_states, (batch_size, hidden_states.shape[0], hidden_states.shape[1]))
        hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)
        return hidden_states
