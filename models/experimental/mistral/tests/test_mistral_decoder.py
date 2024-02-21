# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import tt_lib
import pytest
from loguru import logger
import json
from pathlib import Path
from models.experimental.mistral.tt.mistral_common import precompute_freqs, generate_cos_sin_cache
from models.experimental.mistral.tt.mistral_decoder import TtTransformerBlock
from models.experimental.mistral.tt.model_config import TtModelArgs, get_model_config
from models.experimental.mistral.reference.model import TransformerBlock
from models.utility_functions import torch_to_tt_tensor_rm, tt2torch_tensor
from models.experimental.mistral.mistral_helper_funcs import unpad_from_zero, format_tensor, get_freqs_cis
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)


@pytest.mark.parametrize(
    "model_config",
    ("BFLOAT16-DRAM", "BFLOAT16-L1", "BFLOAT8-DRAM", "BFLOAT8-L1"),
)
@pytest.mark.parametrize(
    "iterations",
    ((3),),
)
@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_mistral_decoder_inference(pcc, model_config, model_location_generator, device, iterations):
    dtype = model_config.split("-")[0]

    mistral_path = Path(model_location_generator("mistral-7B-v0.1", model_subdir="mistral"))
    state_dict = torch.load(mistral_path / "consolidated.00.pth")
    with open(mistral_path / "params.json", "r") as f:
        model_args = TtModelArgs(**json.loads(f.read()))

    state_dict = {k[9:]: v for k, v in state_dict.items() if (k.startswith("layers.0."))}
    base_address = f""

    # base_address = f"layers.0."

    model_args.max_batch_size = 32

    reference_model = TransformerBlock(args=model_args)
    reference_model.load_state_dict(state_dict)

    # TODO Scale the model (mixtral) to multiple devices when T3000 is available
    devices = [
        device,
    ]

    # Setup mem config based on the test
    # TODO move this to model config
    dtype_str, mem_config_str = model_config.split("-")
    if mem_config_str == "DRAM":
        output_mem_config = tt_lib.tensor.MemoryConfig(
            tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM
        )
    elif mem_config_str == "L1":
        output_mem_config = tt_lib.tensor.MemoryConfig(
            tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.L1
        )
    else:
        raise ValueError(f"Invalid memory configuration {mem_config_str}")

    # Initialize TT model
    tt_model = TtTransformerBlock(
        args=model_args,
        devices=devices,
        state_dict=state_dict,
        base_address=base_address,
        layer_num=None,  # single layer
        model_config=get_model_config(model_config),
    )

    generation_start_pos = 0
    generation_length = iterations
    all_tests_pass = True

    seqlen = 1
    batch = 32

    cos, sin = precompute_freqs(model_args.head_dim, model_args.max_seq_len * 2)
    freqs_cis = torch.complex(cos, sin)

    tt_model.tt_cos_cached, tt_model.tt_sin_cached = generate_cos_sin_cache(
        devices, model_args.head_dim, "", model_args.max_seq_len * 2, 1000, tt_model.model_config
    )

    # TODO Update start_pos (check llama test for reference)
    for i in range(generation_length):
        print(f"[Decoder] Generating token {i}")

        # input = torch.randn(1, 32, 4096)
        pt_decode_input = (torch.rand(batch, seqlen, model_args.dim) * 2) - 1
        tt_decode_input = pt_decode_input.clone()
        start_pos = generation_start_pos + i

        decode_input, start_pos, attn_mask = tt_model.prepare_inputs(tt_decode_input, start_pos)
        # Run TT model
        tt_out = tt_model(decode_input, start_pos, attn_mask)
        # tt_output = tt_model(tt_input, bcast_freq_xq, bcast_freq_xk, tt_position, mask, seqlen)

        tt_output_torch = tt2torch_tensor(tt_out).permute(2, 1, 0, 3).squeeze(1)  # [seq, batch, hidden_dim]

        freqs_cis_i = freqs_cis[start_pos, :].unsqueeze(0)
        positions = torch.tensor([start_pos])

        # Reference model
        # mask = tt2torch_tensor(attn_mask[0])
        ref_output = reference_model(pt_decode_input, freqs_cis_i, positions, mask=None)  # mask)

        passing, pcc_message = comp_pcc(ref_output, tt_output_torch, pcc)

        logger.info(comp_allclose(ref_output, tt_output_torch))
        logger.info(pcc_message)

        if passing:
            logger.info("Mistral Decoder Block Passed!")
        else:
            logger.warning("Mistral Decoder Block Failed!")
            all_tests_pass = False

    if all_tests_pass:
        logger.info(f"All {generation_length} Mistral decode iterations Passed!")
    else:
        logger.warning("One or more iterations of Mistral decode Failed!")
        assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
