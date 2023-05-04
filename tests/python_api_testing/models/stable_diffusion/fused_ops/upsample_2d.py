from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")

import torch
from torch import nn
from torch.nn import functional as F
from diffusers import StableDiffusionPipeline
import numpy as np

from libs import tt_lib as ttl
from libs.tt_lib.fallback_ops import fallback_ops
from utility_functions import print_diff_argmax, torch_to_tt_tensor, tt_to_torch_tensor, print_corr_coef
from python_api_testing.sweep_tests.comparison_funcs import comp_allclose_and_pcc

from upsample_nearest2d import TtUpsampleNearest2d


class TtUpsample2d(nn.Module):
    def __init__(self, channels, state_dict, use_conv=False, use_conv_transpose=False, out_channels=None, name="conv", device=None, host=None, base_address="up_blocks.1.upsamplers.0"):
        super().__init__()
        assert not use_conv_transpose, "StableDiffusion's Unet does not use convTranspose, so leaving it out"
        self.in_channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.name = name
        self.device = device
        self.host = host

        self.conv = None
        if self.use_conv:
            # self.conv = nn.Conv2d(self.channels, self.out_channels, 3, padding=1)
            self.conv_weight = state_dict[f"{base_address}.conv.weight"]
            self.conv_bias = state_dict[f"{base_address}.conv.bias"]
            self.conv = fallback_ops.Conv2d(self.conv_weight, self.conv_bias, self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, hidden_states, output_size=None):
        # conv Transpose is not our concern
        # TT's execution is done on bfloat16 - casting makes no sense
        assert hidden_states.shape()[1] == self.in_channels

        if output_size is None:
            hidden_states = TtUpsampleNearest2d(device=self.device)(hidden_states)
        else:
            assert False, "we are not expected to support upsample 2d with output_size yet"
            hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")


        if self.use_conv:
            # hidden_states = tt_to_torch_tensor(hidden_states, self.host)
            hidden_states = self.conv(hidden_states)
            # hidden_states = torch_to_tt_tensor(hidden_states, self.device)

        return hidden_states

class TorchUpsample2D(nn.Module):
    """
    An upsampling layer with an optional convolution.

    Parameters:
        channels: channels in the inputs and outputs.
        use_conv: a bool determining if a convolution is applied.
        use_conv_transpose:
        out_channels:
    """

    def __init__(self, channels, state_dict, use_conv=False, use_conv_transpose=False, out_channels=None, name="conv", base_address="up_blocks.1.upsamplers.0"):
        super().__init__()
        self.in_channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name

        self.conv = None
        if use_conv_transpose:
            # self.conv = nn.ConvTranspose2d(channels, self.out_channels, 4, 2, 1)
            assert use_conv_transpose == True, 'conv_transpose is used!'
        elif use_conv:
            self.conv_weight = state_dict[f"{base_address}.conv.weight"]
            self.conv_bias = state_dict[f"{base_address}.conv.bias"]
            self.conv = fallback_ops.Conv2d(self.conv_weight, self.conv_bias, self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
            # self.conv = nn.Conv2d(self.channels, self.out_channels, 3, padding=1)


    def forward(self, hidden_states, output_size=None):
        assert hidden_states.shape[1] == self.in_channels

        if self.use_conv_transpose:
            return self.conv(hidden_states)

        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        dtype = hidden_states.dtype
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.float32)

        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        # if `output_size` is passed we force the interpolation output
        # size and do not make use of `scale_factor=2`
        if output_size is None:
            hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        else:
            hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")

        # If the input is bfloat16, we cast back to bfloat16
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype)

        if self.use_conv:
            hidden_states = self.conv(hidden_states)

        return hidden_states

def run_upsample2d_inference(device, host):

    pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', torch_dtype=torch.float32)

    # model = pipe.unet
    # model.eval()
    # state_dict = model.state_dict()

    # config = model.config.text_config

    unet = pipe.unet
    unet.eval()
    state_dict = unet.state_dict()
    unet_upblock = pipe.unet.up_blocks[0]
    resnet_upsampler = unet_upblock.upsamplers[0]

    input_shape =  [1, 1280, 32, 32]
    input = torch.randn(input_shape)
    in_channels = 1280
    out_channels = 1280
    # torch_up = TorchUpsample2D(channels=channels, out_channels=out_channels, use_conv=True, state_dict=state_dict)
    # torch_out = torch_up(input)
    torch_out = resnet_upsampler(input)
    print('torch_out:', torch_out[0][0][0][:12])

    tt_input = torch_to_tt_tensor(input, device,)

    tt_up = TtUpsample2d(channels=in_channels, out_channels=out_channels, use_conv=True, state_dict=state_dict, device=device, host=host)
    tt_out = tt_up(tt_input)
    tt_out = tt_to_torch_tensor(tt_out, host)
    print('tt_out:', tt_out[0][0][0][:12])

    print(comp_allclose_and_pcc(tt_out, torch_out))


if __name__ == "__main__":
    # Initialize the device
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()
    run_upsample2d_inference(device, host)
    ttl.device.CloseDevice(device)
