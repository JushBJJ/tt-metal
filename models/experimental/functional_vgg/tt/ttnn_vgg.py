# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import torch

from torchvision import models
from typing import List, Union, Dict, cast

import tt_lib
import ttnn

from tt_lib.fallback_ops import fallback_ops
from models.helper_funcs import Linear as TtLinear
from models.utility_functions import (
    is_conv_supported_on_device,
    run_conv_on_device_wrapper,
)
from models.experimental.vgg.vgg_utils import format_tensor

from models.utility_functions import (
    is_grayskull,
    is_wormhole_b0,
    pad_and_fold_conv_activation_for_unity_stride,
)


def custom_preprocessor(model, name):
    parameters = {}
    if isinstance(model, nn.Conv2d):
        parameters[f"weight"] = model.weight
        parameters[f"bias"] = model.bias
    return parameters


cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
}
conv_ttnn_params = [
    [3, 64, 224, 224],
    [64, 64, 224, 224],
    [64, 128, 112, 112],
    [128, 128, 112, 112],
    [128, 256, 56, 56],
    [256, 256, 56, 56],
    [256, 256, 56, 56],
    [256, 512, 28, 28],
    [512, 512, 28, 28],
    [512, 512, 28, 28],
    [512, 512, 14, 14],
    [512, 512, 14, 14],
    [512, 512, 14, 14],
]
conv_feature_ids = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
classifier_ids = [0, 3, 6]


def ttnn_vgg16(
    device,
    tt_x,
    parameters,
    batch_size,
    model_config,
    conv_config,
):
    i = 0
    for v in cfgs["D"]:
        if v == "M":
            # Call MaxPool
            tt_x = fallback_ops.MaxPool2d(kernel_size=2, stride=2)(tt_x)
            tt_x = ttnn.to_torch(tt_x)
        else:
            h_sharding = True
            if conv_ttnn_params[i][0] > 256 or conv_ttnn_params[i][1] > 256:
                h_sharding = False

            conv_config = ttnn.Conv2dConfig(
                dtype=ttnn.bfloat16,
                weights_dtype=ttnn.bfloat16,
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode_enabled=True,
                fp32_dest_acc_enabled=True,
                activation="relu",
                deallocate_activation=False,
                input_channels_alignment=32,
                reallocate_halo_output=False,
                act_block_h_override=0,
                transpose_shards=True,
                height_sharding=h_sharding,
            )

            # Prepare ttnn conv
            # Prepare weights and bias
            weight = parameters["features"][conv_feature_ids[i]]["weight"]
            tt_weight = ttnn.from_torch(weight, ttnn.bfloat16)
            bias = parameters["features"][conv_feature_ids[i]]["bias"]
            bias = ((bias.unsqueeze(0)).unsqueeze(0)).unsqueeze(0)
            tt_bias = ttnn.from_torch(bias, ttnn.bfloat16)

            # Prepare input
            if isinstance(tt_x, torch.Tensor):
                tt_x = tt_x.permute(0, 2, 3, 1)
                tt_x = ttnn.from_torch(tt_x, ttnn.bfloat16)

            # Call ttnn.conv
            [tt_output_tensor_on_device, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
                input_tensor=tt_x,
                weight_tensor=tt_weight,
                in_channels=conv_ttnn_params[i][0],
                out_channels=conv_ttnn_params[i][1],
                device=device,
                bias_tensor=tt_bias,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                batch_size=batch_size,
                input_height=conv_ttnn_params[i][2],
                input_width=conv_ttnn_params[i][3],
                conv_config=conv_config,
                conv_op_cache={},
            )

            tt_output_tensor = ttnn.from_device(tt_output_tensor_on_device)
            ttnn.deallocate(tt_output_tensor_on_device)
            torch_output_tensor = ttnn.to_torch(tt_output_tensor)

            torch_output_tensor = torch.permute(torch_output_tensor, (0, 3, 1, 2))
            tt_x = torch_output_tensor.reshape(batch_size, conv_ttnn_params[i][1], out_height, out_width)

            i += 1

    # Adaptive Pooling
    tt_x = nn.AdaptiveAvgPool2d((7, 7))(tt_x)

    tt_x = tt_x.reshape(1, 1, 1, -1)
    tt_x = ttnn.from_torch(tt_x, ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    # Linear 1
    tt_x = tt_x @ parameters["classifier"][classifier_ids[0]]["weight"]
    tt_x = tt_x + parameters["classifier"][classifier_ids[0]]["bias"]
    tt_x = ttnn.relu(tt_x)
    # Linear 2
    tt_x = tt_x @ parameters["classifier"][classifier_ids[1]]["weight"]
    tt_x = tt_x + parameters["classifier"][classifier_ids[1]]["bias"]
    tt_x = ttnn.relu(tt_x)
    # Linear 3
    tt_x = tt_x @ parameters["classifier"][classifier_ids[2]]["weight"]
    tt_x = tt_x + parameters["classifier"][classifier_ids[2]]["bias"]
    return tt_x


conv_feature_ids_2 = [0, 3, 6, 8, 11, 13, 16, 18]
conv_ttnn_params_2 = [
    [3, 64, 224, 224],
    [64, 128, 112, 112],
    [128, 256, 56, 56],
    [256, 256, 56, 56],
    [256, 512, 28, 28],
    [512, 512, 28, 28],
    [512, 512, 14, 14],
    [512, 512, 14, 14],
]


def ttnn_vgg11(
    device,
    tt_x,
    parameters,
    batch_size,
    model_config,
    conv_config,
):
    i = 0
    for v in cfgs["A"]:
        if v == "M":
            # Call MaxPool
            tt_x = fallback_ops.MaxPool2d(kernel_size=2, stride=2)(tt_x)
            tt_x = ttnn.to_torch(tt_x)
        else:
            h_sharding = True
            if conv_ttnn_params_2[i][0] > 256 or conv_ttnn_params_2[i][1] > 256:
                h_sharding = False

            conv_config = ttnn.Conv2dConfig(
                dtype=ttnn.bfloat16,
                weights_dtype=ttnn.bfloat16,
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode_enabled=True,
                fp32_dest_acc_enabled=True,
                activation="relu",
                deallocate_activation=False,
                input_channels_alignment=32,
                reallocate_halo_output=False,
                act_block_h_override=0,
                transpose_shards=True,
                height_sharding=h_sharding,
            )

            # Prepare ttnn conv
            # Prepare weights and bias
            weight = parameters["features"][conv_feature_ids_2[i]]["weight"]
            tt_weight = ttnn.from_torch(weight, ttnn.bfloat16)
            bias = parameters["features"][conv_feature_ids_2[i]]["bias"]
            bias = ((bias.unsqueeze(0)).unsqueeze(0)).unsqueeze(0)
            tt_bias = ttnn.from_torch(bias, ttnn.bfloat16)

            # Prepare input
            if isinstance(tt_x, torch.Tensor):
                tt_x = tt_x.permute(0, 2, 3, 1)
                tt_x = ttnn.from_torch(tt_x, ttnn.bfloat16)

            # Call ttnn.conv
            [tt_output_tensor_on_device, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
                input_tensor=tt_x,
                weight_tensor=tt_weight,
                in_channels=conv_ttnn_params_2[i][0],
                out_channels=conv_ttnn_params_2[i][1],
                device=device,
                bias_tensor=tt_bias,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                batch_size=batch_size,
                input_height=conv_ttnn_params_2[i][2],
                input_width=conv_ttnn_params_2[i][3],
                conv_config=conv_config,
                conv_op_cache={},
            )

            tt_output_tensor = ttnn.from_device(tt_output_tensor_on_device)
            ttnn.deallocate(tt_output_tensor_on_device)
            torch_output_tensor = ttnn.to_torch(tt_output_tensor)

            torch_output_tensor = torch.permute(torch_output_tensor, (0, 3, 1, 2))
            tt_x = torch_output_tensor.reshape(batch_size, conv_ttnn_params_2[i][1], out_height, out_width)

            i += 1

    # Adaptive Pooling
    tt_x = nn.AdaptiveAvgPool2d((7, 7))(tt_x)

    tt_x = tt_x.reshape(1, 1, 1, -1)
    tt_x = ttnn.from_torch(tt_x, ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    # Linear 1
    tt_x = tt_x @ parameters["classifier"][classifier_ids[0]]["weight"]
    tt_x = tt_x + parameters["classifier"][classifier_ids[0]]["bias"]
    tt_x = ttnn.relu(tt_x)
    # Linear 2
    tt_x = tt_x @ parameters["classifier"][classifier_ids[1]]["weight"]
    tt_x = tt_x + parameters["classifier"][classifier_ids[1]]["bias"]
    tt_x = ttnn.relu(tt_x)
    # Linear 3
    tt_x = tt_x @ parameters["classifier"][classifier_ids[2]]["weight"]
    tt_x = tt_x + parameters["classifier"][classifier_ids[2]]["bias"]
    return tt_x
