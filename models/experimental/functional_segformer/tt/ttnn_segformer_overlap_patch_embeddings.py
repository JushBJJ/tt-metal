# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import tt_lib


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


class TtSegformerOverlapPatchEmbeddings:
    """Construct the overlapping patch embeddings."""

    def __init__(self, parameters, model):
        super().__init__()
        if model.proj.stride[0] == 4:
            self.proj = model.proj  # parameters.proj
        else:
            self.proj = parameters.proj

    def __call__(self, pixel_values: ttnn.Tensor, parameters, model, stride, patch_size, num_channels, hidden_size):
        device = pixel_values.device()
        batch_size, _, input_height, input_width = pixel_values.shape
        pixel_values = ttnn.permute(pixel_values, (0, 2, 3, 1))

        embeddings, out_height, out_width, weight, bias = ttnn.conv2d(
            input_tensor=pixel_values,
            weight_tensor=parameters.proj.weight,
            in_channels=num_channels,
            out_channels=hidden_size,
            device=device,
            bias_tensor=parameters.proj.bias,
            kernel_size=(patch_size, patch_size),
            stride=(stride, stride),
            padding=(patch_size // 2, patch_size // 2),
            batch_size=batch_size,
            groups=1,
            input_height=input_height,
            input_width=input_width,
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
        embeddings = ttnn.from_device(embeddings)
        embeddings = ttnn.to_layout(embeddings, layout=ttnn.ROW_MAJOR_LAYOUT)

        embeddings = ttnn.reshape(embeddings, (batch_size, out_height, out_width, hidden_size))
        del out_height, out_width, weight, bias
        embeddings = ttnn.from_device(embeddings)
        embeddings = ttnn.to_layout(embeddings, layout=ttnn.TILE_LAYOUT)
        embeddings = ttnn.to_device(embeddings, device=device)

        embeddings = ttnn.permute(embeddings, (0, 3, 1, 2))

        ttnn.deallocate(pixel_values)
        embeddings = ttnn.from_device(embeddings)
        embeddings = ttnn.to_layout(embeddings, layout=ttnn.ROW_MAJOR_LAYOUT)

        embeddings_hw = embeddings  # used instead of shape

        embeddings = ttnn.reshape(
            embeddings, (embeddings.shape[0], embeddings.shape[1], embeddings.shape[2] * embeddings.shape[3])
        )
        embeddings = ttnn.to_layout(embeddings, layout=ttnn.TILE_LAYOUT)
        embeddings = ttnn.to_device(embeddings, device)
        embeddings = ttnn.permute(embeddings, (0, 2, 1))
        if len(embeddings.shape) == 2:
            embeddings = ttnn.reshape(embeddings, (1, embeddings.shape[0], embeddings.shape[1]))

        parameters.layer_norm.weight = ttnn.to_device(parameters.layer_norm.weight, device=device)
        parameters.layer_norm.bias = ttnn.to_device(parameters.layer_norm.bias, device=device)
        embeddings = ttnn.layer_norm(embeddings, weight=parameters.layer_norm.weight, bias=parameters.layer_norm.bias)
        return embeddings, embeddings_hw
