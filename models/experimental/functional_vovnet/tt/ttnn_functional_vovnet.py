# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import tt_lib
import torch.nn as nn


def fold_bn_to_conv_weights_bias(model, path):
    bn_weight = model[path + ".bn.weight"].unsqueeze(1).unsqueeze(1).unsqueeze(1)
    bn_running_var = model[path + ".bn.running_var"].unsqueeze(1).unsqueeze(1).unsqueeze(1)

    weight = model[path + ".conv.weight"]
    weight = (weight / torch.sqrt(bn_running_var)) * bn_weight

    bn_running_mean = model[path + ".bn.running_mean"].unsqueeze(1).unsqueeze(1).unsqueeze(1)
    bn_bias = model[path + ".bn.bias"].unsqueeze(1).unsqueeze(1).unsqueeze(1)

    bias = -(bn_weight) * (bn_running_mean / torch.sqrt(bn_running_var)) + bn_bias

    bias = bias.reshape(1, 1, 1, -1)
    return (
        ttnn.from_torch(
            weight,
        ),
        ttnn.from_torch(bias),
    )


def conv(
    device,
    input_tensor,
    model,
    path,
    input_params,
    conv_params,
    *,
    act_block_h=None,
    reshard=False,
    deallocate=True,
    height_sharding=True,
    activation="relu",
    fused_op=True,
):
    if fused_op:
        weights, bias = fold_bn_to_conv_weights_bias(model, path)
    else:
        weight = model[path + ".weight"]
        bias = model[path + ".bias"]
        weights = ttnn.from_torch(weight)
        bias = bias.reshape(1, 1, 1, -1)
        bias = ttnn.from_torch(bias)
    input_params = input_params
    kernel_size = (weights.shape[2], weights.shape[3])
    conv_params = conv_params
    out_channels = weights.shape[0]
    act_block_h = act_block_h
    reshard = reshard
    height_sharding = height_sharding
    deallocate = deallocate
    activation = activation

    conv_config = ttnn.Conv2dConfig(
        dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat8_b,
        math_fidelity=ttnn.MathFidelity.LoFi,
        activation=activation,
        height_sharding=height_sharding,
        math_approx_mode_enabled=True,
        input_channels_alignment=16 if input_params[1] < 16 else 32,
        transpose_shards=False,
        reshard_if_not_optimal=reshard,
        deallocate_activation=deallocate,
        reallocate_halo_output=False,
    )
    if act_block_h is not None:
        conv_config.act_block_h_override = act_block_h

    [output_tensor, _out_height, _out_width, weights, bias] = ttnn.conv2d(
        input_tensor=input_tensor,
        weight_tensor=weights,
        bias_tensor=bias,
        in_channels=input_params[3],
        out_channels=out_channels,
        device=device,
        kernel_size=kernel_size,
        stride=(conv_params[0], conv_params[1]),
        padding=(conv_params[2], conv_params[3]),
        batch_size=input_params[0],
        input_height=input_params[1],
        input_width=input_params[2],
        conv_config=conv_config,
    )

    return output_tensor, _out_height, _out_width


def effective_se_module(
    device,
    torch_model,
    path,
    input_tensor,
    input_params,
    conv_params,
):
    x, _x_h, _x_w = conv(
        device,
        input_tensor,
        torch_model,
        path,
        input_params,
        conv_params,
        act_block_h=None,
        reshard=False,
        deallocate=True,
        height_sharding=True,
        activation="",
        fused_op=False,
    )

    x = ttnn.hardsigmoid(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)
    x = ttnn.reshape(x, (1, _x_h, _x_w, 256))

    return x


def preprocess_conv_parameter(parameter, *, dtype):
    parameter = ttnn.from_torch(parameter, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    return parameter


def custom_preprocessor(model, name):
    parameters = {}
    if isinstance(model, nn.Conv2d):
        # weight = torch.permute(model.weight, (2, 3, 0, 1))
        weight = model.weight
        bias = model.bias
        # while weight.dim() < 4:
        #     weight = weight.unsqueeze(0)
        # while bias.dim() < 4:
        #     bias = bias.unsqueeze(0)
        parameters["weight"] = preprocess_conv_parameter(weight, dtype=ttnn.bfloat16)
        # parameters["bias"] = preprocess_conv_parameter(bias, dtype=ttnn.bfloat16)
    return parameters
