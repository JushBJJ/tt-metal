# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os
import torch
import pytest
import numpy as np
from loguru import logger
from sklearn.metrics import top_k_accuracy_score
import time

# Set Mixtral flags for CI, if CI environment is setup
if os.getenv("CI") == "true":
    os.environ["MIXTRAL_CKPT_DIR"] = "/mnt/MLPerf/tt_dnn-models/Mistral/Mixtral-8x7B-v0.1/"
    os.environ["MIXTRAL_TOKENIZER_PATH"] = "/mnt/MLPerf/tt_dnn-models/Mistral/Mixtral-8x7B-v0.1/"
    os.environ["MIXTRAL_CACHE_PATH"] = "/mnt/MLPerf/tt_dnn-models/Mistral/Mixtral-8x7B-v0.1/"
    os.environ["TT_METAL_ASYNC_DEVICE_QUEUE"] = "1"
    os.environ["WH_ARCH_YAML"] = "wormhole_b0_80_arch_eth_dispatch.yaml"

import ttnn
from ttnn import ReplicateTensorToMesh, ConcatMeshToTensor

from models.demos.t3000.mixtral8x7b.tt.mixtral_common import (
    prepare_inputs_ttnn_prefill,
    prepare_rotation_mat_ttnn,
    get_rot_transformation_mat,
)
from models.demos.t3000.mixtral8x7b.tt.mixtral_model import TtTransformer
from models.demos.t3000.mixtral8x7b.reference.model import Transformer
from models.demos.t3000.mixtral8x7b.reference.tokenizer import Tokenizer
from models.demos.t3000.mixtral8x7b.tt.model_config import TtModelArgs
from models.utility_functions import comp_pcc, comp_allclose


class Emb(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(32000, 4096)

    def forward(self, x):
        return self.emb(x)


@pytest.mark.parametrize(
    "seq_len",
    (8192,),
)
@pytest.mark.parametrize(
    "n_layers",
    (32,),
)
def test_mixtral_model_inference(t3k_device_mesh, use_program_cache, reset_seeds, n_layers, seq_len):
    pcc = 0.96
    dtype = ttnn.bfloat8_b

    model_args = TtModelArgs(t3k_device_mesh.get_device(0))
    model_args.n_layers = n_layers
    batch = 1
    if seq_len > 2048:
        model_args.max_seq_len = seq_len
        model_args.max_batch_size = 1
    state_dict = model_args.load_state_dict()

    tokenizer = Tokenizer(model_args.tokenizer_path)
    prompt_file = "models/demos/t3000/mixtral8x7b/tests/tale_of_two_cities.txt"
    with open(prompt_file, "r") as f:
        prompt = f.read()
    print("Prompt: ", len(prompt))
    # prompt = prompt * 5
    print("Prompt length: ", len(tokenizer.encode(prompt)))
    encoded_prompts = tokenizer.encode(prompt)[:seq_len]
    reference_model = Transformer(args=model_args)
    reference_model.load_state_dict(state_dict)
    reference_model.eval()

    # Embedding on host
    embd = Emb()
    embd.load_state_dict({"emb.weight": state_dict["tok_embeddings.weight"]})
    rot_mats = prepare_rotation_mat_ttnn(
        model_args.head_dim, model_args.max_seq_len, t3k_device_mesh, mode="prefill", seq_len=seq_len
    )
    head_dim = model_args.dim // model_args.n_heads
    transformation_mat_torch = get_rot_transformation_mat(head_dim)
    transformation_mats = ttnn.as_tensor(
        transformation_mat_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=t3k_device_mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ReplicateTensorToMesh(t3k_device_mesh),
    )
    # Load TTNN model
    tt_model = TtTransformer(
        device_mesh=t3k_device_mesh,
        state_dict=state_dict,
        args=model_args,
        layers=list(range(model_args.n_layers)),
        dtype=dtype,
    )

    # Select the first token from the prompts for initial decoding
    encoded_prompts_tensor = torch.tensor(encoded_prompts)  # [:,0]
    # pt_decode_input = (torch.rand(batch, seq_len, model_args.dim) * 2) - 1  #
    pt_decode_input = embd(encoded_prompts_tensor).view(batch, seq_len, -1)
    # pt_decode_input = torch.load("ref_output_prefil_24L_8192.pt")

    tt_decode_input = pt_decode_input

    start_pos = 0
    current_pos = start_pos % model_args.sliding_window

    for iter in range(1):
        start_time = time.time()
        decode_input, attn_mask, attn_mask_torch = prepare_inputs_ttnn_prefill(
            tt_decode_input,
            tt_model.device_mesh,
        )

        # Run TT model
        tt_out = tt_model(
            decode_input, start_pos, current_pos, attn_mask, rot_mats, transformation_mats, 0, mode="prefill"
        )
        # Work around program cache issue https://github.com/tenstorrent/tt-metal/issues/7159
        del decode_input, attn_mask
        # Convert ttnn tensor to torch tensor
        tt_output_torch = (
            ttnn.to_torch(tt_out, mesh_composer=ConcatMeshToTensor(t3k_device_mesh, dim=0))[0]
            .squeeze(1)
            .view(batch, seq_len, -1)
            .detach()
            .float()
        )

        # logger.info(f"seqlen: {seq_len}, iter: {iter}, TTNN Inference time: {time.time() - start_time:.2f} sec")

    # Measure PCC
    # ref_output = torch.load("ref_output_prefil_32L_8192.pt")
    pt_decode_input = torch.load("ref_output_prefil_24L_8192.pt")
    positions = torch.LongTensor(range(seq_len))
    attn_mask = torch.full((seq_len, seq_len), torch.finfo(torch.float32).min)
    attn_mask_torch = torch.triu(attn_mask, diagonal=1)
    ref_output = reference_model(pt_decode_input, positions, attn_mask_torch, mode="prefill").detach().float()
    # torch.save(ref_output.view(batch, seq_len, -1), "ref_output_prefil_26L_8192.pt")
    # print("done")

    passing, pcc_message = comp_pcc(ref_output.view(batch, seq_len, -1), tt_output_torch.view(batch, seq_len, -1), pcc)
    logger.info(comp_allclose(ref_output, tt_output_torch))
    logger.info(pcc_message)
    for layer in range(24, n_layers):
        ref = reference_model.layers[layer]
        tt = tt_model.layers[layer]
        for mod in range(len(ref.comps)):
            passing, pcc_message = comp_pcc(
                ref.comps[mod].view(batch, seq_len, -1), tt.comps[mod].view(batch, seq_len, -1), pcc
            )
            print("layer: ", layer, "mod: ", mod, pcc_message)
    if passing:
        logger.info(f"Mistral decode Passed!")
    else:
        logger.warning("Mistral decode Failed!")
        assert passing, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
