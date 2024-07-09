# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import timm
import torch
import pytest
import tt_lib
import torch.nn as nn
from loguru import logger
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.functional_vovnet.tt import ttnn_functional_vovnet


def test_effective_se_module(
    reset_seeds,
    device,
):
    rf_model = timm.create_model("hf_hub:timm/ese_vovnet19b_dw.ra_in1k", pretrained=True)
    rf_model = rf_model.eval()
    model = rf_model.stages[0].blocks[0].attn

    torch_input = torch.randn((1, 56, 56, 256), dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16)  # , device=device

    torch_input = torch_input.permute(0, 3, 1, 2)
    torch_output = model(torch_input.float())

    tt_model = ttnn_functional_vovnet.effective_se_module(
        device=device,
        torch_model=model.state_dict(),
        path="fc",
        input_tensor=ttnn_input,
        input_params=(1, 56, 56, 256),
        conv_params=(1, 1, 0, 0),
    )
    tt_output = ttnn.to_torch(tt_model)

    assert_with_pcc(torch_output, tt_output.permute(0, 3, 1, 2), 0.99)
