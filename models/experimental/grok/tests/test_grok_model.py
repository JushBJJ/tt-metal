# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os
import torch
import pytest
import numpy as np
from loguru import logger
from sklearn.metrics import top_k_accuracy_score

# Set Grok flags for CI, if CI environment is setup
if os.getenv("CI") == "true":
    os.environ["GROK_CKPT_DIR"] = "/mnt/MLPerf/tt_dnn-models/Grok/Grok-1/"
    os.environ["GROK_TOKENIZER_PATH"] = "/mnt/MLPerf/tt_dnn-models/Grok/Grok-1/"
    os.environ["GROK_CACHE_PATH"] = "/mnt/MLPerf/tt_dnn-models/Grok/Grok-1/"
    os.environ["TT_METAL_ASYNC_DEVICE_QUEUE"] = "1"
    os.environ["WH_ARCH_YAML"] = "wormhole_b0_80_arch_eth_dispatch.yaml"

import ttnn
from ttnn import ConcatMeshToTensor

from models.experimental.grok.tt.grok_common import prepare_inputs_ttnn, prepare_rotation_mat_ttnn
from models.experimental.grok.tt.grok_model import TtTransformer
from models.experimental.grok.reference.model import Grok1ModelForCausalLM as Transformer
from models.experimental.grok.reference.tokenizer import Tokenizer
from models.experimental.grok.tt.model_config import TtModelArgs
from models.utility_functions import comp_pcc, comp_allclose
from transformers import AutoTokenizer
from models.experimental.grok.reference.configuration_grok1 import Grok1Config


@pytest.mark.parametrize(
    "validation_type",
    ("pcc", "output"),
)
@pytest.mark.parametrize(
    "n_layers",
    (1, 2, 16),
)
@pytest.mark.parametrize(
    "iterations",
    (1, 10),
)
def test_grok_model_inference(t3k_device_mesh, use_program_cache, reset_seeds, iterations, n_layers, validation_type):
    pcc = 0.97
    dtype = ttnn.bfloat8_b

    model_args = TtModelArgs(t3k_device_mesh.get_device(0))
    model_args.n_layers = n_layers

    state_dict = model_args.load_state_dict()

    # tokenizer = Tokenizer(model_args.tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained("hpcai-tech/grok-1", trust_remote_code=True)

    prompts = ["Once"] * 32
    encoded_prompts = [tokenizer.encode(prompt) for prompt in prompts]

    if validation_type == "pcc":
        grok1_config = Grok1Config.from_json_file("models/experimental/grok/reference/config.json")
        grok1_config.num_hidden_layers = model_args.n_layers
        reference_model = Transformer(config=grok1_config)
        reference_model.load_state_dict(state_dict)
        reference_model.eval()

    # Embedding on host
    embd = torch.nn.Embedding(model_args.vocab_size, model_args.hidden_size)
    embd.load_state_dict({"weight": state_dict["model.embed_tokens.weight"]})

    # Load TTNN model
    tt_model = TtTransformer(
        device_mesh=t3k_device_mesh,
        state_dict=state_dict,
        args=model_args,
        layers=list(range(model_args.n_layers)),
        dtype=dtype,
    )

    rot_mat = prepare_rotation_mat_ttnn(
        model_args.head_dim,
        model_args.max_seq_len,
        tt_model.device_mesh,
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
    ref_past_key_values = None
    tt_tokens = []

    for i in range(generation_length):
        logger.info(f"[Decode] Generating token {i}")

        current_pos = generation_start_pos + i

        decode_input, attn_mask = prepare_inputs_ttnn(
            tt_decode_input,
            model_args.dim,
            current_pos,
            tt_model.device_mesh,
        )

        # Run TT model
        tt_multidevice_out = tt_model(decode_input, current_pos, attn_mask, rot_mat)
        # Work around program cache issue https://github.com/tenstorrent/tt-metal/issues/7159
        del decode_input, attn_mask
        # Convert ttnn tensor to torch tensor
        tt_output_torch = (
            ttnn.to_torch(tt_multidevice_out, mesh_composer=ConcatMeshToTensor(t3k_device_mesh, dim=-1))
            .squeeze(1)
            .view(batch, seqlen, -1)
            .detach()
            .float()
        )

        # Measure PCC
        if validation_type == "pcc":
            positions = torch.LongTensor([current_pos])
            ref_output, ref_past_key_values = reference_model(
                inputs_embeds=pt_decode_input,
                past_key_values=ref_past_key_values,
                position_ids=positions,
                use_cache=True,
                return_dict=False,
            )
            ref_output = ref_output.detach().float()

            passing, pcc_message = comp_pcc(
                ref_output.view(batch, seqlen, -1), tt_output_torch.view(batch, seqlen, -1), pcc
            )
            logger.info(comp_allclose(ref_output, tt_output_torch))
            logger.info(pcc_message)

            reference_top1 = np.argmax(ref_output, axis=-1).squeeze()
            top1_acc = top_k_accuracy_score(
                reference_top1, tt_output_torch.squeeze(), k=1, labels=np.arange(tt_output_torch.shape[-1])
            )
            top5_acc = top_k_accuracy_score(
                reference_top1, tt_output_torch.squeeze(), k=5, labels=np.arange(tt_output_torch.shape[-1])
            )
            logger.info(f"Mean Top-1: {top1_acc}")
            logger.info(f"Mean Top-5: {top5_acc}")

            ref_token_batch = ref_output.squeeze().argmax(axis=-1)
            tt_token_batch = tt_output_torch.squeeze().argmax(axis=-1)
            logger.info(f"ref_output: {tokenizer.decode(ref_token_batch[0].item())}")
            logger.info(f"tt_output: {tokenizer.decode(tt_token_batch[0].item())}")
            pt_decode_input = embd(ref_token_batch).view(batch, seqlen, -1)
            tt_decode_input = pt_decode_input  # teacher forcing for PCC test
        else:
            tt_token_batch = tt_output_torch.squeeze().argmax(axis=-1)
            tt_tokens.append(tt_token_batch[0].item())
            logger.info(f'tt_output_torch: {"".join(tokenizer.decode(tt_tokens))}')
            tt_decode_input = embd(tt_token_batch).view(batch, seqlen, -1)

        if validation_type == "pcc":
            if passing:
                logger.info("Mistral Model Passed!")
            else:
                logger.warning("Mistral Model Failed!")
                all_tests_pass = False

    if validation_type == "output":
        if iterations == 1:  # First generated token will be a empty character, so just ignore output validation
            all_tests_pass = True
        elif iterations == 10:
            expected_output = "# The 10 Best Places to Live"
            logger.info(f"Expected output: {expected_output}")
            if "".join(tokenizer.decode(tt_tokens)) == expected_output:
                all_tests_pass = True
            else:
                all_tests_pass = False
        elif iterations == 127:  # TODO Check the baseline output for 127 iterations
            logger.info("Output validation not yet implemented for 127 iterations.")
            all_tests_pass = True
        else:
            logger.info("Output validation not  implemented for this iteration count!")
            all_tests_pass = True

    if all_tests_pass:
        logger.info(f"All {generation_length} Mistral decode iterations Passed!")
    else:
        logger.warning("One or more iterations of Mistral decode Failed!")
        if validation_type == "pcc":
            assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
        else:
            logger.info(f'Generated output: {"".join(tokenizer.decode(tt_tokens))}')
            assert all_tests_pass, f"Expected output did not match the generated output!"
