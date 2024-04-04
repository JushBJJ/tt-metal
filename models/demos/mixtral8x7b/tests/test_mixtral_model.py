# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from loguru import logger
import ttnn
from models.demos.mixtral8x7b.tt.mixtral_common import (
    prepare_inputs_ttnn,
)
from models.demos.mixtral8x7b.tt.mixtral_model import TtTransformer
from models.demos.mixtral8x7b.tt.model_config import TtModelArgs
from models.demos.mixtral8x7b.reference.model import Transformer
from models.demos.mixtral8x7b.reference.tokenizer import Tokenizer
from models.utility_functions import comp_pcc, comp_allclose, get_devices_for_t3000


class Emb(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(32000, 4096)

    def forward(self, x):
        return self.emb(x)


@pytest.mark.parametrize(
    "n_layers",
    (1, 2, 3, 4, 6, 8, 32),
)
@pytest.mark.parametrize(
    "iterations",
    (1,),
)
def test_mixtral_model_inference(all_devices, iterations, n_layers, reset_seeds):
    pcc = 0.99
    dtype = ttnn.bfloat8_b

    devices = all_devices
    num_devices = len(devices)
    assert num_devices == 8, "This test requires a T3000 (8 devices)"
    devices = get_devices_for_t3000(devices, num_devices)

    model_args = TtModelArgs(devices[0])
    model_args.n_layers = n_layers

    state_dict = torch.load(model_args.state_dict_path)
    keys_dict = list(state_dict.keys())[:]
    remv = [f"layers.{i}" for i in range(n_layers, 32)]
    for k in keys_dict:
        if any([r in k for r in remv]):
            state_dict.pop(k)

    tokenizer = Tokenizer(model_args.tokenizer_path)

    prompts = [""] * 32

    encoded_prompts = [tokenizer.encode(prompt) for prompt in prompts]

    reference_model = Transformer(args=model_args)
    reference_model.load_state_dict(state_dict)
    reference_model.eval()

    # Embedding on host
    embd = Emb()
    embd.load_state_dict({"emb.weight": state_dict["tok_embeddings.weight"]})

    # Load TTNN model
    tt_model = TtTransformer(
        devices=devices,
        state_dict=state_dict,
        args=model_args,
        layers=list(range(model_args.n_layers)),
        dtype=dtype,
    )

    generation_start_pos = 0
    generation_length = iterations
    all_tests_pass = True

    seqlen = 1  # Generating one token per user at a time
    batch = 32

    # Select the first token from the prompts for initial decoding
    encoded_prompts_tensor = torch.tensor(encoded_prompts)  # [:,0]
    pt_decode_input = embd(encoded_prompts_tensor[:, 0]).view(batch, seqlen, -1)

    tt_decode_input = pt_decode_input

    for i in range(generation_length):
        print(f"[Decode] Generating token {i}")

        start_pos = generation_start_pos + i
        current_pos = start_pos % model_args.sliding_window

        decode_input, rot_mat = prepare_inputs_ttnn(
            tt_decode_input,
            model_args.dim,
            model_args.head_dim,
            model_args.max_seq_len,
            tt_model.devices,
        )

        # Run TT model
        tt_out = tt_model(decode_input, start_pos, current_pos, rot_mat)
        # Convert ttnn tensor to torch tensor
        tt_output_torch = ttnn.to_torch(tt_out[0]).squeeze(1).view(batch, seqlen, -1)

        positions = torch.LongTensor([start_pos])
        ref_output = reference_model(pt_decode_input, positions)

        # Measure PCC
        passing, pcc_message = comp_pcc(
            ref_output.view(batch, seqlen, -1), tt_output_torch.view(batch, seqlen, -1), pcc
        )
        logger.info(comp_allclose(ref_output, tt_output_torch))
        logger.info(pcc_message)

        if passing:
            logger.info("Mistral Model Passed!")
        else:
            logger.warning("Mistral Model Failed!")
            all_tests_pass = False

    if all_tests_pass:
        logger.info(f"All {generation_length} Mistral decode iterations Passed!")
    else:
        logger.warning("One or more iterations of Mistral decode Failed!")
        assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
