# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from loguru import logger
import json
from pathlib import Path

import ttnn
from models.demos.mistral7b.tt.mistral_attention_ttnn import TtMistralAttention
from models.demos.mistral7b.tt.mistral_common_ttnn import (
    precompute_freqs,
    generate_cos_sin_cache_ttnn,
    prepare_inputs_ttnn,
)
from models.demos.mistral7b.tt.model_config_ttnn import TtModelArgs, get_model_config
from models.demos.mistral7b.reference.model import Attention
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)


@pytest.mark.parametrize(
    "model_config",
    ("BFLOAT16-DRAM", "BFLOAT8-DRAM"),
)
@pytest.mark.parametrize(
    "iterations",
    ((1),),
)
def test_mistral_attention_inference(
    model_config,
    iterations,
    model_location_generator,
    device,
):
    ttnn.enable_program_cache()
    dtype_str, mem_config_str = model_config.split("-")
    if dtype_str == "BFLOAT16":
        dtype = ttnn.bfloat16
        pcc = 0.99
    elif dtype_str == "BFLOAT8":
        dtype = ttnn.bfloat8_b
        pcc = 0.98
    else:
        raise ValueError(f"Unknown dtype {dtype_str}")

    model_config = get_model_config(model_config)

    mistral_path = Path(model_location_generator(model_config["DEFAULT_CACHE_PATH"], model_subdir="mistral"))
    state_dict = torch.load(mistral_path / "consolidated.00.pth")

    base_address = f""
    with open(mistral_path / "params.json", "r") as f:
        model_args = TtModelArgs(**json.loads(f.read()))

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    partial_state_dict = {k[19:]: v for k, v in state_dict.items() if (k.startswith("layers.0.attention."))}

    model_args.max_batch_size = 32
    reference_model = Attention(args=model_args)
    reference_model.load_state_dict(partial_state_dict)

    # TODO Scale the model (mixtral) to multiple devices when T3000 is available
    devices = [
        device,
    ]

    batch = 32
    seq_len = 1

    tt_cos_cached, tt_sin_cached = generate_cos_sin_cache_ttnn(
        devices, model_args.head_dim, "", model_args.max_seq_len * 2, 10000, dtype
    )
    tt_model = TtMistralAttention(
        devices,
        state_dict,
        model_config=model_config,
        layer_num=0,
        dtype=dtype,
        configuration=model_args,
        tt_cos_cached=tt_cos_cached,
        tt_sin_cached=tt_sin_cached,
    )
    generation_start_pos = 0
    generation_length = iterations
    all_tests_pass = True

    cos, sin = precompute_freqs(model_args.head_dim, model_args.max_seq_len * 2)
    freqs_cis = torch.complex(cos, sin)

    for i in range(generation_length):
        pt_attention_input = (torch.rand(batch, seq_len, model_args.dim) * 2) - 1
        tt_attention_input = pt_attention_input.clone()
        start_pos = generation_start_pos + i
        attention_input, start_pos, attn_mask, current_pos = prepare_inputs_ttnn(
            tt_attention_input,
            start_pos,
            tt_model.hidden_size,
            tt_model.n_local_heads,
            tt_model.sliding_window,
            tt_model.devices,
            tt_model.num_devices,
        )

        tt_out = tt_model(
            attention_input,
            start_pos,
            current_pos,
            attn_mask,
        )
        assert isinstance(tt_out, list)  # tt_out should be replicated on N devices
        tt_out = tt_out[0]
        tt_output_torch = ttnn.to_torch(tt_out).permute(1, 0, 2)  # [ batch, seq, hidden_dim]

        # empty_tensor = torch.zeros((start_pos+1, 64))
        # cos, sin = precompute_freqs(model_args.head_dim, 1)
        # freqs_cis = torch.complex(cos, sin)
        freqs_cis_i = freqs_cis[start_pos, :].unsqueeze(0)
        positions = torch.tensor([start_pos])
        # mask = torch.randn(1, 1)

        reference_output = reference_model(pt_attention_input, freqs_cis_i, positions, mask=None)

        passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc)

        logger.info(comp_allclose(reference_output, tt_output_torch))
        logger.info(pcc_message)

        if passing:
            logger.info(f"[start_pos={start_pos}] Mistral_Attention Passed!")
        else:
            logger.warning(f"[start_pos={start_pos}] Mistral_Attention Failed!")
            all_tests_pass = False

        # Check kv cache
        # PyTorch output --------------------------------------------------------------------
        pytorch_layer_present = [
            reference_model.cache_k.clone().permute(0, 2, 1, 3),  # [batch, n_kv_heads, seq, head_dim]
            reference_model.cache_v.clone().permute(0, 2, 1, 3),  # [batch, n_kv_heads, seq, head_dim]
        ]
        # TT hardware execution -------------------------------------------------------------
        tt_layer_present = []
        for layer_past in tt_model.layer_past_list:
            tt_layer_present.append([ttnn.to_torch(cache) for cache in layer_past])
        # concat the pasts by heads
        if len(devices) > 1:
            tt_layer_present = [
                torch.cat([tt_cache for tt_cache in tt_cache_head], dim=1) for tt_cache_head in zip(*tt_layer_present)
            ]
        else:
            tt_layer_present = tt_layer_present[0]

        for i, (cache_pt, cache_tt) in enumerate(zip(pytorch_layer_present, tt_layer_present)):
            # print("CACHE PT", cache_pt, cache_pt.shape) #[32, 8, 4096, 128]
            # print("CACHE TT", cache_tt, cache_tt.shape) #[32, 8, 4096, 128]

            if i == 0:
                logger.info(
                    f"Skipping K cache comparison, since tt_lib rot_embed op does a different permutation from reference PyTorch code"
                )
                continue

            cache_length_to_check = min(model_args.sliding_window, generation_start_pos + generation_length + 1)
            cache_pt = cache_pt[:, :, generation_start_pos:cache_length_to_check, :]
            cache_tt = cache_tt[:, :, generation_start_pos:cache_length_to_check, :]
            does_pass, output_pcc = comp_pcc(cache_pt, cache_tt, pcc)
            logger.info(f"V cache output: {output_pcc}")

            if does_pass:
                logger.info(f"V Cache Passed!")
            else:
                logger.warning(f"V Cache Failed! PCC value is lower than {pcc}")
                all_tests_pass = False

    if all_tests_pass:
        logger.info("Mistral Attention output Passed!")
    else:
        logger.warning("Mistral Attention output Failed!")
        assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
