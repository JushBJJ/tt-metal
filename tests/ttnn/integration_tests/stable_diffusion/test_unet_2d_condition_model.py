# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from diffusers import StableDiffusionPipeline
import pytest
from torch import nn

from tests.ttnn.utils_for_testing import assert_with_pcc

import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters
from models.experimental.functional_stable_diffusion.custom_preprocessing import custom_preprocessor
from models.experimental.functional_stable_diffusion.tt.ttnn_functional_unet_2d_condition_model import (
    UNet2DConditionModel,
)


def ttnn_to_torch(input):
    input = ttnn.to_layout(input, ttnn.ROW_MAJOR_LAYOUT)
    input = ttnn.from_device(input)
    input = ttnn.to_torch(input)
    return input


def torch_to_ttnn(input, device, layout=ttnn.TILE_LAYOUT):
    input = ttnn.from_torch(input, ttnn.bfloat16)
    input = ttnn.to_layout(input, layout)
    input = ttnn.to_device(input, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return input


@pytest.mark.parametrize(
    "batch_size, in_channels, input_height, input_width",
    [
        (2, 4, 32, 32),
    ],
)
def test_unet_2d_condition_model_256x256(device, batch_size, in_channels, input_height, input_width):
    # setup pytorch model
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)

    model = pipe.unet
    model.eval()

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model, custom_preprocessor=custom_preprocessor, device=device
    )

    timestep_shape = [1, 1, 2, 320]
    encoder_hidden_states_shape = [1, 2, 77, 768]
    class_labels = None
    attention_mask = None
    cross_attention_kwargs = None
    return_dict = True
    config = model.config
    config["cross_attention_dim"] = 1280

    hidden_states_shape = [batch_size, in_channels, input_height, input_width]

    input = torch.randn(hidden_states_shape)
    timestep = torch.randn(timestep_shape)
    encoder_hidden_states = torch.randn(encoder_hidden_states_shape)

    torch_output = model(input, timestep=torch.randn([]), encoder_hidden_states=encoder_hidden_states.squeeze(0)).sample

    input = ttnn.from_torch(input, ttnn.bfloat16)
    input = ttnn.to_layout(input, ttnn.TILE_LAYOUT)
    input = ttnn.to_device(input, device, memory_config=ttnn.L1_MEMORY_CONFIG)

    timestep = ttnn.from_torch(timestep, ttnn.bfloat16)
    timestep = ttnn.to_layout(timestep, ttnn.TILE_LAYOUT)
    timestep = ttnn.to_device(timestep, device, memory_config=ttnn.L1_MEMORY_CONFIG)

    encoder_hidden_states = ttnn.from_torch(encoder_hidden_states, ttnn.bfloat16)
    encoder_hidden_states = ttnn.to_layout(encoder_hidden_states, ttnn.TILE_LAYOUT)
    encoder_hidden_states = ttnn.to_device(encoder_hidden_states, device, memory_config=ttnn.L1_MEMORY_CONFIG)

    ttnn_output = UNet2DConditionModel(
        input,
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
        class_labels=class_labels,
        attention_mask=attention_mask,
        cross_attention_kwargs=cross_attention_kwargs,
        return_dict=return_dict,
        parameters=parameters,
        device=device,
        config=config,
        # num_upsamplers=1,
    )
    ttnn_output = ttnn_to_torch(ttnn_output)
    assert_with_pcc(torch_output, ttnn_output, pcc=0.99)


@pytest.mark.parametrize(
    "batch_size, in_channels, input_height, input_width",
    [
        (2, 4, 64, 64),
    ],
)
def test_unet_2d_condition_model_512x512(device, batch_size, in_channels, input_height, input_width):
    # setup pytorch model
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)

    model = pipe.unet
    model.eval()

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model, custom_preprocessor=custom_preprocessor, device=device
    )

    timestep_shape = [1, 1, 2, 320]
    encoder_hidden_states_shape = [1, 2, 77, 768]
    class_labels = None
    attention_mask = None
    cross_attention_kwargs = None
    return_dict = True
    config = model.config
    config["cross_attention_dim"] = 1280

    hidden_states_shape = [batch_size, in_channels, input_height, input_width]

    input = torch.randn(hidden_states_shape)
    timestep = torch.randn(timestep_shape)
    encoder_hidden_states = torch.randn(encoder_hidden_states_shape)

    torch_output = model(input, timestep=torch.randn([]), encoder_hidden_states=encoder_hidden_states.squeeze(0)).sample

    input = ttnn.from_torch(input, ttnn.bfloat16)
    input = ttnn.to_layout(input, ttnn.TILE_LAYOUT)
    input = ttnn.to_device(input, device, memory_config=ttnn.L1_MEMORY_CONFIG)

    timestep = ttnn.from_torch(timestep, ttnn.bfloat16)
    timestep = ttnn.to_layout(timestep, ttnn.TILE_LAYOUT)
    timestep = ttnn.to_device(timestep, device, memory_config=ttnn.L1_MEMORY_CONFIG)

    encoder_hidden_states = ttnn.from_torch(encoder_hidden_states, ttnn.bfloat16)
    encoder_hidden_states = ttnn.to_layout(encoder_hidden_states, ttnn.TILE_LAYOUT)
    encoder_hidden_states = ttnn.to_device(encoder_hidden_states, device, memory_config=ttnn.L1_MEMORY_CONFIG)

    ttnn_output = UNet2DConditionModel(
        input,
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
        class_labels=class_labels,
        attention_mask=attention_mask,
        cross_attention_kwargs=cross_attention_kwargs,
        return_dict=return_dict,
        parameters=parameters,
        device=device,
        config=config,
    )
    ttnn_output = ttnn_to_torch(ttnn_output)
    assert_with_pcc(torch_output, ttnn_output, pcc=0.99)
