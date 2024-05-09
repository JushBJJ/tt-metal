# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from models.utility_functions import (
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
)

from models.experimental.functional_stable_diffusion.tt2.ttnn_functional_upsample_nearest_2d import upsample_nearest2d
from tt_lib.fallback_ops import fallback_ops
from models.experimental.functional_stable_diffusion.tt2.ttnn_functional_utility_functions import (
    run_ttnn_conv_with_pre_and_post_tensor_formatting,
)
from models.experimental.functional_stable_diffusion.tt2.ttnn_functional_utility_functions import (
    permute_conv_parameters,
)

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
}


class upsample2d:
    def __init__(
        self, device, parameters, reader_patterns_cache, batch_size, input_height, input_width, compute_kernel_config
    ):
        self.input_height = input_height
        self.input_width = input_width
        self.batch_size = batch_size
        self.device = device
        self.parameters = parameters
        parameters.conv.weight, parameters.conv.bias = permute_conv_parameters(
            parameters.conv.weight, parameters.conv.bias
        )

        self.scale_factor = 2
        input_height = input_height * self.scale_factor
        input_width = input_width * self.scale_factor

        out_channels = parameters.conv.weight.shape[0]
        in_channels = parameters.conv.weight.shape[1]
        # breakpoint()
        parameters.conv.bias = torch.reshape(parameters.conv.bias, (1, 1, 1, out_channels))
        self.conv_weights = ttnn.from_torch(parameters.conv.weight, ttnn.float32)
        self.conv_bias = ttnn.from_torch(parameters.conv.bias, ttnn.float32)
        self.conv_config_override = {}
        if (out_channels, in_channels, input_height, input_width) in config_override:
            self.conv_config_override = config_override[(out_channels, in_channels, input_height, input_width)]

        self.conv = ttnn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            dtype=ttnn.bfloat8_b,
            device=device,
            use_1d_systolic_array=False,
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
            deallocate_activation=True,
            compute_kernel_config=compute_kernel_config,
        )
        self.output_height = self.conv.output_height
        self.output_width = self.conv.output_width
        self.in_channels = in_channels
        self.out_channels = out_channels
        print(f"Upsample Input = {input_height}x{input_width} Output = {self.output_height}x{self.output_width}")

    def __call__(self, input, in_channels, out_channels):
        if input.layout == ttnn.TILE_LAYOUT:
            input = ttnn.to_layout(input, ttnn.ROW_MAJOR_LAYOUT)
        # # slice out batch
        input = ttnn.reshape(input, (2, self.input_height, self.input_width, input.shape[3]))
        print(f"Upsample Input = {input.shape}")
        tt_out = upsample_nearest2d(input, self.scale_factor)
        del input
        tt_out = ttnn.reshape(tt_out, (1, 1, tt_out.shape[0] * tt_out.shape[1] * tt_out.shape[2], tt_out.shape[3]))
        if ttnn.get_memory_config(tt_out) != self.conv.conv.input_sharded_memory_config:
            tt_out = ttnn.to_memory_config(tt_out, self.conv.conv.input_sharded_memory_config)
        tt_out = self.conv(tt_out)
        conv_config = ttnn.ConvConfig(
            dtype=ttnn.bfloat8_b,
            weights_dtype=ttnn.bfloat8_b,
            math_fidelity=ttnn.MathFidelity.LoFi,
            activation=None,
            height_sharding=False,
            input_channels_alignment=32,
        )
        if self.conv_config_override and "act_block_h" in self.conv_config_override:
            print("Setting Act Block H to ", self.conv_config_override["act_block_h"])
            conv_config.act_block_h = self.conv_config_override["act_block_h"]

        [tt_out, _out_height, _out_width, _dev_weights, _dev_bias] = ttnn.conv2d(
            input_tensor=tt_out,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            weight_tensor=self.conv_weights,
            bias_tensor=self.conv_bias,
            device=self.device,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            batch_size=self.batch_size,
            input_height=self.input_height * self.scale_factor,
            input_width=self.input_width * self.scale_factor,
            conv_config=conv_config,
            reshard_if_not_optimal=True,
        )
        # tt_out = run_ttnn_conv_with_pre_and_post_tensor_formatting(
        #     self.device,
        #     self.conv,
        #     tt_out,
        #     self.conv.batch_size,
        #     self.conv.input_height,
        #     self.conv.input_width,
        #     self.conv.out_channels,
        # )
        print(f"Upsample Output = {tt_out.shape}")
        return tt_out
