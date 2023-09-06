"""
SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

SPDX-License-Identifier: Apache-2.0
"""

from pathlib import Path
import sys
import torch
import pytest
from loguru import logger

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../../../..")

from models.utility_functions import (
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
    comp_allclose,
    comp_pcc,
)
import tt_lib
from tests.models.swin.tt.swin_stage import TtSwinStage
from tests.models.swin.tt.swin_patch_merging import TtSwinPatchMerging
from transformers import SwinModel


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_swin_stage_inference(pcc, reset_seeds):
    device = tt_lib.device.CreateDevice(0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    STAGE_LAYER_INDEX = 0
    base_address = f"encoder.layers.{STAGE_LAYER_INDEX}"

    model = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

    # Torch swinstage
    torch_model = model.encoder.layers[STAGE_LAYER_INDEX]

    # Tt swinstage
    dim = 96
    input_resolution = (56, 56)
    depth = 2
    num_heads = 3

    tt_model = TtSwinStage(
        config=model.config,
        dim=dim,
        input_resolution=input_resolution,
        depth=depth,
        num_heads=num_heads,
        downsample=TtSwinPatchMerging,
        state_dict=model.state_dict(),
        base_address=base_address,
        device=device,
    )

    # Run torch model
    hidden_states = torch.rand(1, 3136, 96)
    input_dimensions = (56, 56)

    torch_output = torch_model(hidden_states, input_dimensions)

    # Run tt model
    hidden_states = torch.unsqueeze(hidden_states, 0)
    tt_hidden_states = torch_to_tt_tensor_rm(hidden_states, device)

    tt_output = tt_model(tt_hidden_states, input_dimensions)

    # Compare outputs
    tt_output_torch = tt_to_torch_tensor(tt_output[0])
    tt_output_torch = tt_output_torch.squeeze(0)
    does_pass, pcc_message = comp_pcc(torch_output[0], tt_output_torch, pcc)

    logger.info(comp_allclose(torch_output[0], tt_output_torch))
    logger.info(pcc_message)

    tt_lib.device.CloseDevice(device)
    if does_pass:
        logger.info("SwinStage Passed!")
    else:
        logger.warning("SwinStage Failed!")

    assert does_pass
