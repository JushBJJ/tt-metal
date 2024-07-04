# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import cv2
import ttnn
from models.utility_functions import skip_for_wormhole_b0, skip_for_grayskull, is_grayskull


def blazeblock(
    x,
    in_channel,
    out_channel,
    kernel_size,
    stride,
    padding,
    skip_proj,
    parameters,
    i,
    conv_config,
    device,
    out_height,
    out_width,
    itr=1,
):
    print("ITeration count = ", i, " ", x.shape)
    channel_pad = out_channel - in_channel
    if stride == 2:
        if kernel_size == 3:
            h = ttnn.to_torch(x)
            # h = ttnn.pad(x, ((0, 2), (0, 2)), value=0)
            h = F.pad(h, (0, 2, 0, 2), "constant", 0)
            h = ttnn.from_torch(h, dtype=ttnn.bfloat16)
        else:
            # print("Shape before padding :", x.shape)
            h = ttnn.to_torch(x)
            h = torch.permute(h, (0, 3, 1, 2))
            h = F.pad(h, (1, 2, 1, 2), "constant", 0)
            h = torch.permute(h, (0, 2, 3, 1))
            h = ttnn.from_torch(h, dtype=ttnn.bfloat16)
            out_height = h.shape[-2]
            out_width = h.shape[-1]
            # print("Shape after padding:", h.shape)
        # print("Maxpool input shape :", x.shape)
        max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)
        x = ttnn.to_torch(x)
        x = torch.permute(x, (0, 3, 1, 2))
        x = max_pool(x)
        x = torch.permute(x, (0, 2, 3, 1))
        x = ttnn.from_torch(x, dtype=ttnn.bfloat16)
        # print("Maxpool Output shape :", x.shape)
    else:
        h = x

    if skip_proj:
        if i == 5:
            print("PyTorchConv")
            x = ttnn.to_torch(x).to(torch.float)
            x = torch.reshape(x, (1, 48, 64, 64))
            skip_conv = nn.Conv2d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            skip_conv.weight = parameters[i].skip_proj.weight
            skip_conv.bias = parameters[i].skip_proj.bias
            x = skip_conv(x)
            out_height = x.shape[-2]
            out_width = x.shape[-1]
            x = ttnn.from_torch(x, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
            x = ttnn.permute(x, (0, 2, 3, 1))

        else:
            print("TTConv")
            weight = ttnn.from_torch(
                parameters[i].skip_proj.weight, dtype=ttnn.bfloat16
            )  # , memory_config = ttnn.L1_MEMORY_CONFIG)

            bias = ttnn.from_torch(
                parameters[i].skip_proj.bias.unsqueeze(0).unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16
            )  # , memory_config = ttnn.L1_MEMORY_CONFIG)
            print("skip proj input shape :", x.shape)
            print(
                (
                    "Iteration :",
                    i,
                    x.shape[0],
                    in_channel,
                    out_channel,
                    x.shape[-2],
                    x.shape[-1],
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    1,
                    True,
                    None,
                    False,
                ),
                ",",
            )
            [x, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
                input_tensor=x,
                weight_tensor=weight,
                in_channels=in_channel,
                out_channels=out_channel,
                device=device,
                bias_tensor=bias,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                batch_size=x.shape[0],
                input_height=x.shape[-2],
                input_width=x.shape[-1],
                conv_config=conv_config,
                conv_op_cache={},
                debug=None,
                groups=1,
            )
            if i == 16:
                out_height = 16
                out_width = 16
            print("skip proj output shape :", x.shape, " ", out_height, " ", out_width)

    elif channel_pad > 0:
        x = ttnn.pad(x, (0, 0, 0, 0), value=0)

    weight = ttnn.from_torch(
        parameters[i].convs[0].weight, dtype=ttnn.bfloat16
    )  # , memory_config = ttnn.L1_MEMORY_CONFIG)

    bias = ttnn.from_torch(
        parameters[i].convs[0].bias.unsqueeze(0).unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16
    )  # , memory_config = ttnn.L1_MEMORY_CONFIG)

    conv_config = ttnn.Conv2dConfig(
        dtype=ttnn.bfloat16,
        # weights_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat16,
        math_fidelity=ttnn.MathFidelity.LoFi,
        activation="",
        height_sharding=True,
        math_approx_mode_enabled=True,
        fp32_dest_acc_enabled=False,
        packer_l1_accum_enabled=False,
        input_channels_alignment=32,  # 16 if h.shape[1] < 16 else 32,
        transpose_shards=False,
        reshard_if_not_optimal=True,
        deallocate_activation=True,
        reallocate_halo_output=True,
    )

    h = ttnn.to_layout(h, layout=ttnn.ROW_MAJOR_LAYOUT)
    if i == 5 or i == 9 or (itr == 2 and i == 1):
        # print("Pytorch conv1")
        print("PyTorchConv")
        h = ttnn.to_torch(h).to(torch.float)

        # h = torch.reshape(h, (1, h.shape[-1], out_height, out_width))
        h = torch.permute(h, (0, 3, 1, 2))
        conv5 = nn.Conv2d(
            in_channels=in_channel,
            out_channels=in_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channel,
        )
        conv5.weight = parameters[i].convs[0].weight
        conv5.bias = parameters[i].convs[0].bias
        h = conv5(h)
        h = ttnn.from_torch(h, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    else:
        # print("Input shape for conv1 :", h.shape)
        # if i == 16:
        #    print("Out height and width :", out_height," ",out_width)
        print("TTConv")
        print("Input shape for conv1 :", h.shape, " ", out_height, " ", out_width)
        print(
            (
                "Iteration :",
                i,
                h.shape[0],
                in_channel,
                in_channel,
                h.shape[-2],
                h.shape[-1],
                kernel_size,
                kernel_size,
                stride,
                stride,
                padding,
                padding,
                in_channel,
                True,
                None,
                False,
            ),
            ",",
        )
        if i == 16:
            out_height = 35
            out_width = 35
        if itr == 2 and i == 0:
            out_height = 19
            out_width = 19
        [h, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
            input_tensor=h,
            # weight_tensor=ttnn.from_device(parameters[i].convs[0].weight),
            weight_tensor=weight,
            in_channels=in_channel,
            out_channels=in_channel,
            device=device,
            # bias_tensor=ttnn.from_device(parameters[i].convs[0].bias),
            bias_tensor=bias,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(padding, padding),
            batch_size=1,
            input_height=out_height,
            input_width=out_width,
            conv_config=conv_config,
            conv_op_cache={},
            debug=False,
            groups=in_channel,
        )
        print(
            "Output shape for conv1 :",
            h.shape,
            " ",
            out_height,
            " ",
            out_width,
        )

    weight = ttnn.from_torch(
        parameters[i].convs[1].weight, dtype=ttnn.bfloat16
    )  # , device = device, layout = ttnn.TILE_LAYOUT, memory_config = ttnn.L1_MEMORY_CONFIG)
    # weight = ttnn.permute(ttnn.from_device(weight), (2, 3, 0, 1))
    bias = ttnn.from_torch(
        parameters[i].convs[1].bias.unsqueeze(0).unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16
    )  # , device = device, layout = ttnn.TILE_LAYOUT, memory_config = ttnn.L1_MEMORY_CONFIG)

    if i == 2 or i == 3 or i == 4 or i == 5 or i == 9:
        # print("Pytorch conv 2")
        h = ttnn.to_torch(h).to(torch.float)
        # if i == 9:
        #    print("Shape of h in 9:", h.shape)
        print("PyTorchConv")
        if i != 5 and i != 9:
            h = torch.reshape(h, (1, h.shape[-1], out_height, out_width))
        conv2 = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        conv2.weight = parameters[i].convs[1].weight
        conv2.bias = parameters[i].convs[1].bias
        h = conv2(h)
        h = ttnn.from_torch(h, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    else:
        print("Input shape for conv2 :", h.shape)
        print("TTConv")
        if itr == 2 and i == 1:
            # h = ttnn.to_torch(h)
            # h = ttnn.from_torch(h, dtype = ttnn.bfloat16)
            h = ttnn.permute(h, (0, 2, 3, 1))
            # h = ttnn.to_layout(h, layout = ttnn.ROW_MAJOR_LAYOUT)
            h = ttnn.to_torch(h)
            h = ttnn.from_torch(h, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        """
        print(
            (
                "Iteration :",
                i,
                h.shape[0],
                in_channel,
                out_channel,
                h.shape[-2],
                h.shape[-1],
                1,
                1,
                1,
                1,
                0,
                0,
                1,
                True,
                None,
                False,
            ),
            ",",
        )
        """
        print("Shape of input and thinbgs :", h.shape, " ", weight.shape, " ", bias.shape)
        [h, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
            input_tensor=h,
            # weight_tensor=ttnn.from_device(parameters[i].convs[1].weight),
            weight_tensor=weight,
            in_channels=in_channel,
            out_channels=out_channel,
            device=device,
            # bias_tensor=ttnn.from_device(parameters[i].convs[1].bias),
            bias_tensor=bias,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            batch_size=1,
            input_height=out_height,
            input_width=out_width,
            conv_config=conv_config,
            conv_op_cache={},
            debug=False,
            groups=1,
        )
        print("Output shape for conv2 :", h.shape)
        h = ttnn.to_layout(h, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        h = ttnn.reshape(h, (h.shape[0], out_height, out_width, h.shape[-1]))
        # h = ttnn.to_layout(h, layout = ttnn.TILE_LAYOUT)
        h = ttnn.to_torch(h)
        h = torch.permute(h, (0, 3, 1, 2))
        h = ttnn.from_torch(h, dtype=ttnn.bfloat16)

    # x = ttnn.to_layout(x, dtype = ttnn.bfloat16, layout = ttnn.TILE_LAYOUT)

    x = ttnn.to_torch(x)
    x = ttnn.from_torch(x, device=device)
    if i == 9:
        print("Shape of x before reshape :", x.shape)
    if i == 9:
        out_height = 32
        out_width = 32
        x = ttnn.reshape(ttnn.from_device(x), (1, out_height, out_width, x.shape[-1]))
    else:
        x = ttnn.reshape(ttnn.from_device(x), (1, out_height, out_width, x.shape[-1]))
    x = ttnn.permute(ttnn.to_device(x, device=device), (0, 3, 1, 2))  # n, c, h, w -> n, w, c, h -> 0, 2, 3, 1
    # x = ttnn.to_layout(x, layout=ttnn.TILE_LAYOUT)
    h = ttnn.to_torch(h)
    x = ttnn.to_torch(x)
    if i == 9:
        print("Shape of x and h", h.shape, " ", x.shape)
    temp = h + x
    temp = ttnn.from_torch(temp, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    return ttnn.permute(ttnn.relu(temp), (0, 2, 3, 1)), out_height, out_width
    # return ttnn.permute(ttnn.relu(temp), (0, 2, 3, 1)), out_height, out_width


def blazepose(x, parameters, device):
    detection2roi_method = "alignment"
    kp1 = 2
    kp2 = 3
    theta0 = 90 * np.pi / 180
    dscale = 1.5
    dy = 0.0
    b = x.shape[0]
    use_shallow_conv_variant = False
    reader_patterns_cache = {}

    conv_config = ttnn.Conv2dConfig(
        dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        math_fidelity=ttnn.MathFidelity.LoFi,
        height_sharding=True,
        input_channels_alignment=(16 if use_shallow_conv_variant else 32),
        deallocate_activation=False,
        fp32_dest_acc_enabled=False,
        packer_l1_accum_enabled=False,
        # act_block_h_override=64,
        activation="relu",
        reallocate_halo_output=True,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    weight = ttnn.from_torch(
        parameters.backbone1[0].weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
    )  # , memory_config = ttnn.L1_MEMORY_CONFIG)
    bias = ttnn.from_torch(
        parameters.backbone1[0].bias.unsqueeze(0).unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )  # , memory_config = ttnn.L1_MEMORY_CONFIG)
    # print("Shape of tensors :", x.shape, " ", weight.shape, " ", bias.shape)

    [x, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
        input_tensor=x,
        # weight_tensor=ttnn.from_device(parameters.backbone1[0].weight),
        weight_tensor=weight,
        in_channels=3,
        out_channels=48,
        device=device,
        # bias_tensor=ttnn.from_device(parameters.backbone1[0].bias),
        bias_tensor=bias,
        kernel_size=(5, 5),
        stride=(1, 1),
        padding=(2, 2),
        batch_size=1,
        input_height=128,
        input_width=128,
        conv_config=conv_config,
        conv_op_cache=reader_patterns_cache,
        debug=False,
        groups=1,
        # activation = "relu"
    )
    print("Shape of x in conv 1 with new config:", x.shape)
    ttnn.deallocate(weights_device)
    ttnn.deallocate(bias_device)
    # temp = torch.save(x, "first_conv.pt")
    # ttnn.deallocate(parameters.backbone1[0].weight)
    # ttnn.deallocate(parameters.backbone1[0].bias)
    # x = ttnn.relu(x)
    x = ttnn.to_torch(x)
    x = ttnn.from_torch(x, dtype=ttnn.bfloat16)
    x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # x = ttnn.reshape(x, (x.shape[0], x.shape[-1], out_height, out_width))

    in_channel = [48, 48, 48, 48, 64, 64, 64, 64, 96, 96, 96, 96, 96, 96, 96, 128, 128, 128, 128, 128, 128, 128]
    out_channel = [48, 48, 48, 64, 64, 64, 64, 96, 96, 96, 96, 96, 96, 96, 128, 128, 128, 128, 128, 128, 128, 128]

    i = 2
    for i in range(2, 24):
        if i > 1:
            if i == 5 or i == 9 or i == 16:
                x, out_height, out_width = blazeblock(
                    x,
                    in_channel[i - 2],
                    out_channel[i - 2],
                    5,
                    2,
                    0,
                    True,
                    parameters.backbone1,
                    i,
                    None,
                    device,
                    out_height,
                    out_width,
                    1,
                )
                print("SHape in loop :", i, " ", x.shape, " ", out_height, " ", out_width)
            else:
                x, out_height, out_width = blazeblock(
                    x,
                    in_channel[i - 2],
                    out_channel[i - 2],
                    5,
                    1,
                    2,
                    False,
                    parameters.backbone1,
                    i,
                    None,
                    device,
                    out_height,
                    out_width,
                    1,
                )
                print("SHape in loop :", i, " ", x.shape, " ", out_height, " ", out_width)
                c1_out_height = out_height
                c1_out_width = out_width
        i += 1

    i = 0

    for i in range(6):
        if i == 0:
            h, out_height, out_width = blazeblock(
                x, 128, 256, 5, 2, 0, True, parameters.backbone2, i, None, device, out_height, out_width, 2
            )
            print("SHape in loop :", i, " ", h.shape, " ", out_height, " ", out_width)
        else:
            h, out_height, out_width = blazeblock(
                h, 256, 256, 5, 1, 2, False, parameters.backbone2, i, None, device, out_height, out_width, 2
            )
            print("SHape in loop :", i, " ", h.shape, " ", out_height, " ", out_width)
        i += 1

    weight = ttnn.from_torch(
        parameters.classifier_8.weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
    )  # , memory_config = ttnn.L1_MEMORY_CONFIG)
    bias = ttnn.from_torch(
        parameters.classifier_8.bias.unsqueeze(0).unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    conv_config = ttnn.Conv2dConfig(
        dtype=ttnn.bfloat16,
        # weights_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat16,
        math_fidelity=ttnn.MathFidelity.LoFi,
        activation="",
        height_sharding=True,
        math_approx_mode_enabled=True,
        fp32_dest_acc_enabled=False,
        packer_l1_accum_enabled=False,
        input_channels_alignment=32,  # 16 if h.shape[1] < 16 else 32,
        transpose_shards=False,
        reshard_if_not_optimal=True,
        deallocate_activation=True,
        reallocate_halo_output=True,
    )
    print("Input shape of conv1 :", x.shape)
    """
    [c1, c1_out_height, c1_out_width, weights_device, bias_device]= ttnn.conv2d(
        input_tensor=x,
        weight_tensor=weight,
        in_channels=128,
        out_channels=128,
        device=device,
        bias_tensor=bias,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        batch_size=x.shape[0],
        input_height=c1_out_height,
        input_width=c1_out_width,
        conv_config=conv_config,
        conv_op_cache={},
        debug=False,
        groups=1,
    )
    print("Shape c1s :", c1.shape," ",c1_out_height," ",c1_out_width)
    #c1 = ttnn.reshape(ttnn.to_layout(c1, layout = ttnn.ROW_MAJOR_LAYOUT),(1, 128, 16, 16))
    c1 = ttnn.to_torch(c1)
    c1 = torch.reshape(c1, (1, 128, 16, 16))
    c1 = ttnn.from_torch(c1, dtype = ttnn.bfloat16, device = device)
    print("Shape of c1 after torch:", c1.shape)
    c1 = ttnn.permute(c1, (0, 2, 3, 1))
    print("Shape after permute :", c1.shape)
    c1 = ttnn.reshape(c1, (b, -1, 1))
    """
    # Channel problem
    class8 = nn.Conv2d(128, 2, 1)
    class8.weight = parameters.classifier_8.weight
    class8.bias = parameters.classifier_8.bias
    print("nput of  class8 :", x.shape)
    temp = ttnn.to_torch(x).to(torch.float)
    temp = torch.permute(temp, (0, 3, 1, 2))
    c1 = class8(temp)
    print("Shape of c1 after first conv :", c1.shape)
    c1 = c1.permute(0, 2, 3, 1)
    c1 = c1.reshape(b, -1, 1)
    # c1 = ttnn.from_torch(c1, dtype = ttnn.bfloat16)
    # Channel problem
    """
    weight = ttnn.from_torch(
        parameters.classifier_16.weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
    )  # , memory_config = ttnn.L1_MEMORY_CONFIG)
    bias = ttnn.from_torch(
        parameters.classifier_16.bias.unsqueeze(0).unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    print("Input shape for conv2 :", h.shape," ",out_height," ",out_width)
    [c2, c2_out_height, c2_out_width, weights_device, bias_device] = ttnn.conv2d(
        input_tensor=h,
        weight_tensor=weight,
        in_channels=256,
        out_channels=6,
        device=device,
        bias_tensor=bias,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        batch_size=h.shape[0],
        input_height=out_height,
        input_width=out_width,
        conv_config=conv_config,
        conv_op_cache={},
        debug=False,
        groups=1,
    )
    print("Shape c2s :", c2.shape," ",c2_out_height," ",c2_out_width)


    """
    class16 = nn.Conv2d(256, 6, 1)
    class16.weight = weight = parameters.classifier_16.weight
    class16.bias = parameters.classifier_16.bias
    print("Shape of h for final conv2 :", h.shape, " ", class16.weight.shape, " ", class16.bias.shape)
    temp_h = ttnn.to_torch(h).to(torch.float)
    temp_h = torch.permute(temp_h, (0, 3, 1, 2))
    c2 = class16(temp_h)
    c2 = c2.permute(0, 2, 3, 1)
    c2 = c2.reshape(b, -1, 1)
    # c2 = ttnn.from_torch(c2, dtype =  ttnn.bfloat16)
    # c2 = ttnn.permute(c2, (0, 2, 3, 1))
    # c2 = ttnn.reshape(c2, (b, -1, 1))
    print("Shape at concat :", c1.shape, " ", c2.shape)
    # c = ttnn.concat([c1, c2], dim=1)
    c = torch.cat((c1, c2), dim=1)
    c = ttnn.from_torch(c, dtype=ttnn.bfloat16)
    """
    r1 = ttnn.conv2d(
        input_tensor=x,
        weight_tensor=parameters.regressor_8.weight,
        in_channels=128,
        out_channels=24,
        device=device,
        bias_tensor=parameters.regressor_8.bias,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        batch_size=x.shape[0],
        input_height=x.shape[-2],
        input_width=x.shape[-1],
        conv_config=conv_config,
        conv_op_cache={},
        debug=False,
        groups=1,
    )

    r1 = ttnn.permute(r1, (0, 2, 3, 1))
    r1 = ttnn.reshape(r1, (b, -1, 12))

    r2 = ttnn.conv2d(
        input_tensor=h,
        weight_tensor=parameters.regressor_16.weight,
        in_channels=256,
        out_channels=72,
        device=device,
        bias_tensor=parameters.regressor_16.bias,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        batch_size=h.shape[0],
        input_height=h.shape[-2],
        input_width=h.shape[-1],
        conv_config=conv_config,
        conv_op_cache={},
        debug=False,
        groups=1,
    )

    r2 = ttnn.permute(r2, (0, 2, 3, 1))
    r2 = ttnn.reshape(r2, (b, -1, 12))
    """
    regressor_8 = nn.Conv2d(128, 24, 1)
    regressor_8.weight = parameters.regressor_8.weight
    regressor_8.bias = parameters.regressor_8.bias
    x = ttnn.to_torch(x).to(torch.float)
    x = torch.permute(x, (0, 3, 1, 2))
    r1 = regressor_8(x)
    r1 = r1.permute(0, 2, 3, 1)
    r1 = r1.reshape(b, -1, 12)
    r1 = ttnn.from_torch(r1, dtype=ttnn.bfloat16)  #

    regressor_16 = nn.Conv2d(256, 72, 1)
    regressor_16.weight = parameters.regressor_16.weight
    regressor_16.bias = parameters.regressor_16.bias
    h = ttnn.to_torch(h).to(torch.float)
    h = torch.permute(h, (0, 3, 1, 2))
    r2 = regressor_16(h)
    r2 = r2.permute(0, 2, 3, 1)
    r2 = r2.reshape(b, -1, 12)
    r2 = ttnn.from_torch(r2, dtype=ttnn.bfloat16)  #
    r = ttnn.concat([r1, r2], dim=1)
    return [r, c]
