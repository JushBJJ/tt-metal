# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn

from tt_lib.fallback_ops import fallback_ops
from models.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor
from models.experimental.functional_stable_diffusion.tt.ttnn_functional_basic_transformer_block import (
    basic_transformer_block,
)


def torch_to_ttnn(input, device, layout=ttnn.TILE_LAYOUT):
    input = ttnn.from_torch(input, ttnn.bfloat16)
    input = ttnn.to_layout(input, layout)
    input = ttnn.to_device(input, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return input


def ttnn_to_torch(input):
    input = ttnn.to_layout(input, ttnn.ROW_MAJOR_LAYOUT)
    input = ttnn.from_device(input)
    input = ttnn.to_torch(input)
    return input


def transformer_2d_model(
    hidden_states,
    parameters,
    config,
    encoder_hidden_states=None,
    timestep=None,
    class_labels=None,
    cross_attention_kwargs=None,
    return_dict=True,
    num_attention_heads=16,
    attention_head_dim=88,
    in_channels=None,
    out_channels=None,
    num_layers=1,
    norm_num_groups=32,
    num_vector_embeds=None,
    patch_size=None,
    num_embeds_ada_norm=None,
    use_linear_projection=False,
    norm_type="layer_norm",
    eps=1e-5,
    device=None,
):
    inner_dim = num_attention_heads * attention_head_dim

    is_input_continuous = (in_channels is not None) and (patch_size is None)
    is_input_vectorized = num_vector_embeds is not None
    is_input_patches = in_channels is not None and patch_size is not None
    assert (
        is_input_continuous and (not is_input_patches) and (not is_input_vectorized)
    ), "we only support continuous input."
    if norm_type == "layer_norm" and num_embeds_ada_norm is not None:
        deprecation_message = (
            f"The configuration file of this model: transformer_2d_model is outdated. `norm_type` is either not set or"
            " incorrectly set to `'layer_norm'`.Make sure to set `norm_type` to `'ada_norm'` in the config."
            " Please make sure to update the config accordingly as leaving `norm_type` might led to incorrect"
            " results in future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it"
            " would be very nice if you could open a Pull request for the `transformer/config.json` file"
        )
        deprecate(
            "norm_type!=num_embeds_ada_norm",
            "1.0.0",
            deprecation_message,
            standard_warn=False,
        )
        norm_type = "ada_norm"

    is_input_continuous = (in_channels is not None) and (patch_size is None)

    # 1. Input
    if is_input_continuous:
        batch, _, height, width = hidden_states.shape
        residual = hidden_states

        hidden_states = ttnn.group_norm(
            input_tensor=hidden_states,
            num_groups=norm_num_groups,
            epsilon=eps,
            weight=parameters.norm.weight,
            bias=parameters.norm.bias,
        )

        if not use_linear_projection:
            hidden_states = ttnn_to_torch(hidden_states)
            hidden_states = torch_to_tt_tensor_rm(hidden_states, device)

            proj_in = fallback_ops.Conv2d(
                weights=parameters.proj_in.weight,
                biases=parameters.proj_in.bias,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            hidden_states = proj_in(hidden_states)

            hidden_states = tt_to_torch_tensor(hidden_states)
            hidden_states = torch_to_ttnn(hidden_states, device=device)

            inner_dim = hidden_states.shape[1]

            hidden_states = ttnn.permute(hidden_states, (0, 2, 3, 1))

            hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.ROW_MAJOR_LAYOUT)
            hidden_states = ttnn.reshape(hidden_states, (1, batch, height * width, inner_dim))

        else:
            inner_dim = hidden_states.shape[1]
            hidden_states = ttnn.permute(hidden_states, (0, 2, 3, 1))

            hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.ROW_MAJOR_LAYOUT)
            hidden_states = ttnn.reshape(hidden_states, (1, batch, height * width, inner_dim))

            hidden_states = ttnn.to_device(hidden_states, device)
            hidden_states = ttnn.matmul(hidden_states, parameters.proj_in.weight)
            hidden_states = ttnn.add(hidden_states, parameters.proj_in.bias)

    # 2. Blocks
    for d in range(num_layers):
        hidden_states = basic_transformer_block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            attention_mask=None,
            cross_attention_kwargs=cross_attention_kwargs,
            class_labels=class_labels,
            config=config,
            parameters=parameters.transformer_blocks[d],
            device=device,
        )

    # 3. Output
    if is_input_continuous:
        if not use_linear_projection:
            hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.ROW_MAJOR_LAYOUT)
            hidden_states = ttnn.reshape(hidden_states, (batch, height, width, inner_dim))

            hidden_states = ttnn.permute(hidden_states, (0, 3, 1, 2))

            hidden_states = ttnn_to_torch(hidden_states)
            hidden_states = torch_to_tt_tensor_rm(hidden_states, device)

            proj_out = fallback_ops.Conv2d(
                weights=parameters.proj_out.weight,
                biases=parameters.proj_out.bias,
                in_channels=out_channels,
                out_channels=in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            hidden_states = proj_out(hidden_states)

            hidden_states = tt_to_torch_tensor(hidden_states)
            hidden_states = torch_to_ttnn(hidden_states, device=device)
        else:
            hidden_states = ttnn.to_device(hidden_states, device)
            hidden_states = ttnn.matmul(hidden_states, parameters.proj_out.weight)
            hidden_states = ttnn.add(hidden_states, parameters.proj_out.bias)

            hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.ROW_MAJOR_LAYOUT)
            hidden_states = ttnn.reshape(hidden_states, (batch, height, width, inner_dim))

            hidden_states = ttnn.permute(hidden_states, (0, 3, 1, 2))

        output = ttnn.add(
            hidden_states,
            residual,
        )

    if not return_dict:
        return (output,)
    return output
