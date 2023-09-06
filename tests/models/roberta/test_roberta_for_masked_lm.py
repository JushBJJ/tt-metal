"""
SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

SPDX-License-Identifier: Apache-2.0
"""

import math
from pathlib import Path
import sys
import torch
import torch.nn as nn
import numpy as np
from loguru import logger

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

from tests.models.roberta.roberta_common import (
    torch2tt_tensor,
    tt2torch_tensor,
)
from tests.models.roberta.roberta_for_masked_lm import TtRobertaForMaskedLM

import tt_lib
from tt_lib.fallback_ops import fallback_ops
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_allclose, comp_pcc

from transformers import RobertaForMaskedLM
from transformers import AutoTokenizer


def test_roberta_masked_lm_inference():
    torch.manual_seed(1234)
    device = tt_lib.device.CreateDevice(0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    base_address = f""

    torch_model = RobertaForMaskedLM.from_pretrained("roberta-base")
    torch_model.eval()
    with torch.no_grad():
        # Tt roberta
        tt_model = TtRobertaForMaskedLM(
            config=torch_model.config,
            base_address=base_address,
            device=device,
            state_dict=torch_model.state_dict(),
            reference_model=torch_model,
        )

        tt_model.eval()

        tokenizer = AutoTokenizer.from_pretrained("roberta-base")

        text_1 = "The Milky Way is a <mask> galaxy."
        text_2 = "The capital of the UK is <mask>."
        inputs = tokenizer(text_1, return_tensors="pt")

        # Run torch model
        torch_output = torch_model(**inputs)

        torch_output = torch_output.logits
        # retrieve index of <mask>
        mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(
            as_tuple=True
        )[0]

        predicted_token_id = torch_output[0, mask_token_index].argmax(axis=-1)
        decoded_token = tokenizer.decode(predicted_token_id)
        logger.info("Torch predicted token")
        logger.info(decoded_token)

        # Run tt model
        tt_attention_mask = torch.unsqueeze(inputs.attention_mask, 0)
        tt_attention_mask = torch.unsqueeze(tt_attention_mask, 0)
        tt_attention_mask = torch2tt_tensor(tt_attention_mask, device)

        tt_output = tt_model(inputs.input_ids, tt_attention_mask)

        tt_output = tt_output.logits

        # Convert to torch

        tt_output_torch = tt2torch_tensor(tt_output)
        tt_output_torch = tt_output_torch.squeeze(0)

        # retrieve index of <mask>
        mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(
            as_tuple=True
        )[0]
        # Get predicted token
        predicted_token_id = tt_output_torch[0, mask_token_index].argmax(axis=-1)
        decoded_token = tokenizer.decode(predicted_token_id)
        logger.info("TT predicted token")
        logger.info(decoded_token)

        # Compare outputs
        does_pass, pcc_message = comp_pcc(torch_output, tt_output_torch, 0.98)

        logger.info(comp_allclose(torch_output, tt_output_torch))
        logger.info(pcc_message)

        tt_lib.device.CloseDevice(device)

        if does_pass:
            logger.info("RobertaForMaskedLM Passed!")
        else:
            logger.warning("RobertaForMaskedLM Failed!")

        assert does_pass


if __name__ == "__main__":
    test_roberta_masked_lm_inference()
