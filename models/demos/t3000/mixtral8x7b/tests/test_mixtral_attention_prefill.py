# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os
import torch
import pytest
from loguru import logger

# Set Mixtral flags for CI, if CI environment is setup
if os.getenv("CI") == "true":
    os.environ["MIXTRAL_CKPT_DIR"] = "/mnt/MLPerf/tt_dnn-models/Mistral/Mixtral-8x7B-v0.1/"
    os.environ["MIXTRAL_TOKENIZER_PATH"] = "/mnt/MLPerf/tt_dnn-models/Mistral/Mixtral-8x7B-v0.1/"
    os.environ["MIXTRAL_CACHE_PATH"] = "/mnt/MLPerf/tt_dnn-models/Mistral/Mixtral-8x7B-v0.1/"
    os.environ["TT_METAL_ASYNC_DEVICE_QUEUE"] = "1"

import ttnn
from ttnn import ReplicateTensorToMesh, ConcatMeshToTensor
from models.demos.t3000.mixtral8x7b.tt.mixtral_attention import TtMixtralAttention
from models.demos.t3000.mixtral8x7b.tt.mixtral_common import (
    prepare_inputs_ttnn_prefill,
    prepare_rotation_mat_ttnn,
    get_rot_transformation_mat,
)
from models.demos.t3000.mixtral8x7b.reference.model import Attention, precompute_freqs_cis
from models.demos.t3000.mixtral8x7b.tt.model_config import TtModelArgs
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)


@torch.no_grad()
def test_mixtral_attention_inference(t3k_device_mesh, use_program_cache, reset_seeds):
    pcc = 0.99
    dtype = ttnn.bfloat8_b
    model_args = TtModelArgs(t3k_device_mesh.get_device(0))
    state_dict = model_args.load_state_dict()
    batch = 1
    seq_len = 128  # length to generate
    model_args.max_batch_size = batch
    model_args.max_seq_len = seq_len
    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    partial_state_dict = {k[19:]: v for k, v in state_dict.items() if (k.startswith("layers.0.attention."))}

    reference_model = Attention(args=model_args)
    reference_model.load_state_dict(partial_state_dict)

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

    tt_model = TtMixtralAttention(
        t3k_device_mesh, state_dict, args=model_args, layer_num=0, dtype=dtype, transformation_mats=transformation_mats
    )

    generation_start_pos = 0
    generation_length = 1
    all_tests_pass = True

    for i in range(generation_length):
        pt_attention_input = (torch.rand(batch, seq_len, model_args.dim) * 2) - 1
        tt_attention_input = pt_attention_input
        start_pos = generation_start_pos + i
        attention_input, attn_mask, attn_mask_torch = prepare_inputs_ttnn_prefill(
            tt_attention_input,
            # tt_model.hidden_size,
            model_args.dim,
            start_pos,
            model_args.sliding_window,
            t3k_device_mesh,
        )

        current_pos = start_pos % model_args.sliding_window
        tt_out = tt_model.forward_prefill(
            attention_input,
            start_pos,
            current_pos,
            attn_mask,
            rot_mats,
        )
        # Work around program cache issue https://github.com/tenstorrent/tt-metal/issues/7159
        del attention_input, attn_mask
        tt_output_torch = (
            ttnn.to_torch(tt_out, mesh_composer=ConcatMeshToTensor(t3k_device_mesh, dim=0))[0]
            .squeeze(2)
            .view(batch, seq_len, -1)
        )  # [ batch, seq, hidden_dim]
        positions = torch.LongTensor(range(seq_len))
        freqs_cis_i = precompute_freqs_cis(model_args.head_dim, 128_000)[positions]
        reference_output = reference_model(pt_attention_input, freqs_cis_i, positions, mask=attn_mask_torch)

        passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc)

        logger.info(comp_allclose(reference_output, tt_output_torch))
        logger.info(pcc_message)

        if passing:
            logger.info(f"[start_pos={start_pos}] Mistral_Attention Passed!")
        else:
            logger.warning(f"[start_pos={start_pos}] Mistral_Attention Failed!")
            all_tests_pass = False

        tt_layer_present_all = [ttnn.from_device(lp) for lp in tt_model.layer_past]
        tt_layer_present_all = [
            ttnn.to_torch(lp, mesh_composer=ConcatMeshToTensor(t3k_device_mesh, dim=1)) for lp in tt_layer_present_all
        ]
        pytorch_layer_present = [
            reference_model.cache_k.clone().permute(0, 2, 1, 3),  # [batch, n_kv_heads, seq, head_dim]
            reference_model.cache_v.clone().permute(0, 2, 1, 3),  # [batch, n_kv_heads, seq, head_dim]
        ]
        for cache_pt, cache_tt in zip(pytorch_layer_present, tt_layer_present_all):
            cache_pt = cache_pt[:, :, :seq_len, :]
            cache_tt = cache_tt[:, :, :seq_len, :]
            # print(cache_pt, cache_tt)
            does_pass, output_pcc = comp_pcc(cache_pt, cache_tt, pcc)
            logger.info(f"Output: {output_pcc}")

            if does_pass:
                logger.info(f"KV Cache Passed!")
            else:
                logger.warning(f"KV Cache Failed! PCC value is lower than {pcc}")
                all_tests_pass = False

    if all_tests_pass:
        logger.info("Mistral Attention output Passed!")
    else:
        logger.warning("Mistral Attention output Failed!")
        assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
