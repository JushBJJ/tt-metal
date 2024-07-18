# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn

from torchvision import models
from loguru import logger
from models.utility_functions import comp_allclose, comp_pcc
from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc_without_tensor_printout
import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters
from models.experimental.functional_vgg.tt import ttnn_vgg


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype, math_fidelity", ((1, ttnn.bfloat16, ttnn.bfloat16, ttnn.MathFidelity.LoFi),)
)
def test_vgg16(
    device,
    imagenet_sample_input,
    use_program_cache,
    batch_size,
    act_dtype,
    weight_dtype,
    math_fidelity,
):
    torch_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    torch_model.to(torch.bfloat16)
    torch_model.eval()
    torch_input_tensor = imagenet_sample_input.to(torch.bfloat16)
    torch_input_tensor_nchw = torch_input_tensor = imagenet_sample_input.to(torch.bfloat16)
    golden_output = torch_model(torch_input_tensor_nchw)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        device=device,
        custom_preprocessor=ttnn_vgg.custom_preprocessor,
    )

    model_config = {
        "MATH_FIDELITY": math_fidelity,
        "WEIGHTS_DTYPE": weight_dtype,
        "ACTIVATIONS_DTYPE": act_dtype,
    }

    conv_config = ttnn.Conv2dConfig(
        dtype=model_config["ACTIVATIONS_DTYPE"],
        weights_dtype=model_config["WEIGHTS_DTYPE"],
        math_fidelity=model_config["MATH_FIDELITY"],
        activation="relu",
        deallocate_activation=True,
        input_channels_alignment=16,
        act_block_h_override=0,
        transpose_shards=True,
    )

    torch_input_tensor = torch.permute(torch_input_tensor_nchw, (0, 2, 3, 1))
    tt_input_tensor = ttnn.from_torch(torch_input_tensor, ttnn.bfloat16)

    ttnn_output = ttnn_vgg.ttnn_vgg16(device, tt_input_tensor, parameters, batch_size, model_config, conv_config)
    torch_output_tensor = ttnn.to_torch(ttnn_output)

    passing, pcc_msg = check_with_pcc_without_tensor_printout(
        (torch_output_tensor.squeeze(0)).squeeze(0), golden_output, pcc=0.99
    )
    logger.info(f"PCC: {pcc_msg}")
