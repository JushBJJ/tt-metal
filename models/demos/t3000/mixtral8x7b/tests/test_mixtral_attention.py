# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from loguru import logger

import ttnn
from models.demos.t3000.mixtral8x7b.tt.mixtral_attention import TtMixtralAttention
from models.demos.t3000.mixtral8x7b.tt.mixtral_common import (
    prepare_inputs_ttnn,
)
from models.demos.t3000.mixtral8x7b.tt.model_config import TtModelArgs
from models.demos.t3000.mixtral8x7b.reference.model import Attention, precompute_freqs_cis
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from ttnn import ReplicateTensorToMesh, ConcatMeshToTensor


def test_mixtral_attention_inference(device_mesh, reset_seeds):
    pcc = 0.99
    dtype = ttnn.bfloat8_b

    model_args = TtModelArgs(device_mesh.get_device(0))
    state_dict = torch.load(model_args.consolidated_weights_path(0), map_location="cpu")

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    partial_state_dict = {k[19:]: v for k, v in state_dict.items() if (k.startswith("layers.0.attention."))}

    reference_model = Attention(args=model_args)
    reference_model.load_state_dict(partial_state_dict)

    batch = 32
    seq_len = 1  # length to generate

    tt_model = TtMixtralAttention(device_mesh, state_dict, args=model_args, layer_num=0, dtype=dtype)
    generation_start_pos = 0
    generation_length = 1
    all_tests_pass = True

    for i in range(generation_length):
        pt_attention_input = (torch.rand(batch, seq_len, model_args.dim) * 2) - 1
        tt_attention_input = pt_attention_input.clone()
        start_pos = generation_start_pos + i
        attention_input, rot_mat = prepare_inputs_ttnn(
            tt_attention_input,
            tt_model.hidden_size,
            tt_model.head_dim,
            tt_model.max_seq_len,
            device_mesh,
        )
        current_pos = start_pos % model_args.sliding_window
        tt_out = tt_model(
            attention_input,
            start_pos,
            current_pos,
            rot_mat,
        )
        tt_output_torch = ttnn.to_torch(tt_out, mesh_composer=ConcatMeshToTensor(device_mesh, dim=0))[0].view(
            batch, 1, -1
        )

        positions = torch.LongTensor([start_pos])
        freqs_cis_i = precompute_freqs_cis(model_args.head_dim, 128_000)[positions]
        reference_output = reference_model(pt_attention_input, freqs_cis_i, positions, mask=None)
        passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc)

        logger.info(comp_allclose(reference_output, tt_output_torch))
        logger.info(pcc_message)

        if passing:
            logger.info(f"[start_pos={start_pos}] Mistral_Attention Passed!")
        else:
            logger.warning(f"[start_pos={start_pos}] Mistral_Attention Failed!")
            all_tests_pass = False

    if all_tests_pass:
        logger.info("Mistral Attention output Passed!")
    else:
        logger.warning("Mistral Attention output Failed!")
        assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
