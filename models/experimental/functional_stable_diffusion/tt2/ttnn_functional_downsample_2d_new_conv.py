# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import ttnn
import torch
from typing import Optional
import torch.nn as nn
from tt_lib.fallback_ops import fallback_ops
from models.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor
from models.experimental.functional_stable_diffusion.tt2.ttnn_functional_utility_functions import (
    run_ttnn_conv_with_pre_and_post_tensor_formatting,
)
import math


def permute_conv_parameters(weight, bias):
    weight = ttnn.to_torch(weight)
    weight = torch.permute(weight, (2, 3, 0, 1))
    bias = ttnn.to_torch(bias)
    return weight, bias


config_override = {
    (320, 320, 64, 64): {"act_block_h": 64},
    (640, 640, 32, 32): {"act_block_h": 64},
    (640, 1920, 32, 32): {"act_block_h": 32},
    (640, 1280, 32, 32): {"act_block_h": 32},
    (1280, 1920, 16, 16): {"act_block_h": 32},
    (1280, 1280, 32, 32): {"act_block_h": 32},
    (320, 960, 64, 64): {"act_block_h": 32},
    (640, 960, 32, 32): {"act_block_h": 32},
    (320, 640, 64, 64): {"act_block_h": 32},
    (640, 640, 64, 64): {"act_block_h": 64},
    (640, 320, 64, 64): {"act_block_h": 64},
}


class downsample_2d:
    def __init__(
        self,
        device,
        parameters,
        reader_patterns_cache,
        batch_size,
        input_height,
        input_width,
        compute_kernel_config,
    ):
        self.device = device
        self.parameters = parameters
        parameters.conv.weight, parameters.conv.bias = permute_conv_parameters(
            parameters.conv.weight, parameters.conv.bias
        )
        parameters.conv.bias = torch.reshape(parameters.conv.bias, (1, 1, 1, parameters.conv.bias.shape[-1]))

        self.conv_weights = ttnn.from_torch(parameters.conv.weight, ttnn.float32)
        self.conv_bias = ttnn.from_torch(parameters.conv.bias, ttnn.float32)
        self.input_height = input_height
        self.input_width = input_width
        self.batch_size = batch_size
        self.out_channels = parameters.conv.weight.shape[0]
        self.in_channels = parameters.conv.weight.shape[1]
        self.conv_config_override = {}
        if (self.out_channels, self.in_channels, input_height, input_width) in config_override:
            self.conv_config_override = config_override[
                (self.out_channels, self.in_channels, input_height, input_width)
            ]

        self.stride = 2
        self.conv = ttnn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=(3, 3),
            stride=(self.stride, self.stride),
            padding=(1, 1),
            dtype=ttnn.bfloat8_b,
            device=device,
            use_1d_systolic_array=True if self.in_channels < 320 else False,
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            reader_patterns_cache=reader_patterns_cache,
            weight=self.conv_weights,
            bias=self.conv_bias,
            math_fidelity=ttnn.MathFidelity.LoFi,
            weights_dtype=ttnn.bfloat8_b,
            conv_blocking_and_parallelization_config_override=self.conv_config_override,
            use_shallow_conv_variant=False,
            # enable_auto_formatting=True,
            compute_kernel_config=compute_kernel_config,
        )

        self.output_height = self.conv.output_height
        self.output_width = self.conv.output_width
        print(f"Downsample Input = {input_height}x{input_width} Output = {self.output_height}x{self.output_width}")

    def __call__(
        self,
        in_channels,
        hidden_states,
        use_conv=False,
        out_channels=None,
        padding=1,
        dtype: Optional[ttnn.DataType] = None,
    ):
        stride = 2
        assert padding == 1

        assert hidden_states.shape[-1] == in_channels

        if use_conv and padding == 0:
            pad = (0, 1, 0, 1)
            hidden_states = ttnn.pad(hidden_states, pad, value=0)

        if ttnn.get_memory_config(hidden_states) != self.conv.conv.input_sharded_memory_config:
            hidden_states = ttnn.to_memory_config(hidden_states, self.conv.conv.input_sharded_memory_config)
        # hidden_states = self.conv(hidden_states)
        conv_config = ttnn.ConvConfig(
            dtype=ttnn.bfloat8_b,
            weights_dtype=ttnn.bfloat8_b,
            math_fidelity=ttnn.MathFidelity.LoFi,
            activation=None,
            height_sharding=True if self.in_channels < 320 else False,
            input_channels_alignment=32,
        )
        if self.conv_config_override and "act_block_h" in self.conv_config_override:
            print("Setting Act Block H to ", self.conv_config_override["act_block_h"])
            conv_config.act_block_h = self.conv_config_override["act_block_h"]

        [hidden_states, _out_height, _out_width, _dev_weights, _dev_bias] = ttnn.conv2d(
            input_tensor=hidden_states,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=(3, 3),
            stride=(self.stride, self.stride),
            padding=(1, 1),
            device=self.device,
            batch_size=self.batch_size,
            input_height=self.input_height,
            input_width=self.input_width,
            weight_tensor=self.conv_weights,
            bias_tensor=self.conv_bias,
            conv_config=conv_config,
        )
        # hidden_states = run_ttnn_conv_with_pre_and_post_tensor_formatting(
        #     self.device,
        #     self.conv,
        #     hidden_states,
        #     self.conv.batch_size,
        #     self.conv.output_height,
        #     self.conv.output_width,
        #     self.conv.out_channels,
        # )

        return hidden_states
