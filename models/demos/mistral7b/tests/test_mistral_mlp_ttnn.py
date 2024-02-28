# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import json
from pathlib import Path
import ttnn
from models.demos.mistral7b.tt.model_config_ttnn import TtModelArgs, get_model_config
from models.demos.mistral7b.tt.mistral_mlp_ttnn import TtMistralMLP
from models.demos.mistral7b.reference.model import FeedForward
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)


@pytest.mark.parametrize(
    "model_config",
    ("BFLOAT16-DRAM", "BFLOAT8-DRAM"),
)
def test_mistral_mlp_inference(model_config, model_location_generator, device):
    ttnn.enable_program_cache()

    dtype = {"BFLOAT16": ttnn.bfloat16, "BFLOAT8": ttnn.bfloat8_b}[model_config.split("-")[0]]
    model_config = get_model_config(model_config)

    mistral_path = Path(model_location_generator(model_config["DEFAULT_CACHE_PATH"], model_subdir="mistral"))
    state_dict = torch.load(mistral_path / "consolidated.00.pth")

    with open(mistral_path / "params.json", "r") as f:
        model_args = TtModelArgs(**json.loads(f.read()))
    state_dict = {k[22:]: v for k, v in state_dict.items() if (k.startswith("layers.0.feed_forward"))}
    base_address = "layers.0"

    model_args.max_batch_size = 1
    model_args.WEIGHTS_DTYPE = dtype
    reference_model = FeedForward(args=model_args)
    reference_model.load_state_dict(state_dict)

    tt_model = TtMistralMLP(
        device=device,
        state_dict=state_dict,
        base_address=base_address,
        model_config=model_config,
    )
    torch_input = torch.randn(1, 1, 17, 4096)
    reference_output = reference_model(torch_input)
    tt_input = ttnn.from_torch(
        torch_input, device=device, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
    )

    logger.info("Compilation pass for Mistral_MLP")
    tt_output = tt_model(tt_input)

    logger.info("Performance pass for Mistral_MLP")
    tt_output = tt_model(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output)

    pcc_required = 0.99
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(pcc_message)

    if passing:
        logger.info("Mistral_MLP Passed!")
    else:
        logger.warning("Mistral_MLP Failed!")

    assert passing, f"Mistral_MLP output does not meet PCC requirement {pcc_required}: {pcc_message}."
