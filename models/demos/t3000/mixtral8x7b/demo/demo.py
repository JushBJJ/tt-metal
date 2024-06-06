# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os
import torch
import json
import tt_lib as ttl
import pytest
from loguru import logger
from time import time

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
    prepare_inputs_ttnn,
    prepare_inputs_ttnn_prefill,
    prepare_rotation_mat_ttnn,
    sample,
    cache_attention,
    get_rot_transformation_mat,
)
from models.demos.t3000.mixtral8x7b.tt.mixtral_model import TtTransformer
from models.demos.t3000.mixtral8x7b.tt.mixtral_embedding import TtMixtralEmbedding
from models.demos.t3000.mixtral8x7b.reference.tokenizer import Tokenizer


from models.demos.t3000.mixtral8x7b.tt.model_config import TtModelArgs


class Emb(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(32000, 4096)

    def forward(self, x):
        return self.emb(x)


# load from json, return as a list
def load_inputs(user_input, batch):
    if isinstance(user_input, str):
        with open(user_input, "r") as f:
            user_input = json.load(f)
    assert len(user_input) >= batch, f"Number of users (batch) must be {batch}!"
    in_prompt = []
    for i in range(batch):
        in_prompt.append(user_input[i]["prompt"])
    return in_prompt


def preprocess_inputs(input_prompts, tokenizer, model_args, dtype, instruct, device_mesh):
    """
    Run tokenizer on inputs, and create embeddings for the first token of each input
    """
    if instruct:
        # Pre append [INST] and post append [/INST] to the encoded prompts if instruct mode
        encoded_prompts = [tokenizer.encode("[INST] " + prompt + " [/INST]") for prompt in input_prompts]
    else:
        encoded_prompts = [tokenizer.encode(prompt) for prompt in input_prompts]

    prompt_lens = [len(x) for x in encoded_prompts]

    min_prompt_len = min(prompt_lens)
    max_prompt_len = max(prompt_lens)
    assert (
        max_prompt_len <= model_args.max_seq_len
    ), f"Max prompt length {max_prompt_len} exceeds model max seq len {model_args.max_seq_len}"
    assert min_prompt_len > 0, "Minimum prompt length must be greater than 0"
    assert (
        min_prompt_len <= model_args.max_seq_len
    ), f"Minimum prompt length {min_prompt_len} exceeds model max seq len {model_args.max_seq_len}"

    prefill_seq_len = 128 if min_prompt_len < 128 else (1024 if min_prompt_len < 1024 else 2048)
    input_tokens_prefill = torch.full((len(input_prompts), prefill_seq_len), tokenizer.pad_id, dtype=torch.int32)
    input_tokens_decode = torch.full(
        (len(input_prompts), max_prompt_len - prefill_seq_len), tokenizer.pad_id, dtype=torch.int32
    )

    logger.info(f"# of users: {len(encoded_prompts)}")
    for i, encoded in enumerate(encoded_prompts):
        input_tokens_prefill[i, : len(encoded[:prefill_seq_len])] = torch.tensor(encoded[:prefill_seq_len]).to(
            input_tokens_prefill
        )
        # Right padding
        input_tokens_decode[i, : len(encoded[prefill_seq_len:])] = torch.tensor(encoded[prefill_seq_len:]).to(
            input_tokens_decode
        )

    input_mask_bool = input_tokens_decode != tokenizer.pad_id
    input_mask = input_mask_bool.int()  # from_torch doesn't support bool type

    # convert to ttnn tensor
    # Encoded input tokens need to be uint32 for embedding. Otherwise the dtype conversion to bfloat16 will change the tokenizer ID
    input_tokens_prefill_tt = [
        ttnn.from_torch(
            input_tokens_prefill[i, :].unsqueeze(0),
            device=device_mesh,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ReplicateTensorToMesh(device_mesh),
        )
        for i in range(len(encoded_prompts))
    ]
    input_tokens_decode_tt = [
        ttnn.from_torch(
            input_tokens_decode[:, i].unsqueeze(0),
            device=device_mesh,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ReplicateTensorToMesh(device_mesh),
        )
        for i in range(max_prompt_len - prefill_seq_len)
    ]
    input_mask_tt = [
        ttnn.from_torch(
            input_mask[:, i].unsqueeze(0),
            device=device_mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ReplicateTensorToMesh(device_mesh),
        )
        for i in range(max_prompt_len - prefill_seq_len)
    ]
    return (
        input_tokens_prefill_tt,
        input_tokens_decode_tt,
        max_prompt_len,
        input_mask_tt,
        input_tokens_prefill,
        input_tokens_decode,
        input_mask_bool,
        prefill_seq_len,
    )


@torch.no_grad()
def run_mixtral_demo(user_input, batch_size, device_mesh, instruct_mode):
    assert batch_size == 32, "Batch size must be 32"

    dtype = ttnn.bfloat8_b

    embed_on_host = True  # Do embedding and argmax on host. TODO Seeing bad output when on device
    seqlen = 1  # Generating one token per user at a time

    logger.info(f"Reading inputs...")
    if len(user_input) == 1:
        input_prompts = user_input * 32  # Always process 32 users
    else:
        input_prompts = load_inputs(user_input, 32)

    # Load model args, weights, and tokenizer
    model_args = TtModelArgs(device_mesh.get_device(0), instruct=instruct_mode)
    tokenizer = Tokenizer(model_args.tokenizer_path)

    model_args.n_layers = 32  # Full model

    logger.info("Loading weights...")
    state_dict = torch.load(model_args.state_dict_path)
    # If not using the full model, remove the layers that are not used
    keys_dict = list(state_dict.keys())[:]
    remv = [f"layers.{i}" for i in range(model_args.n_layers, 32)]
    for k in keys_dict:
        if any([r in k for r in remv]):
            state_dict.pop(k)

    # Embedding on host
    if embed_on_host:
        embd = Emb()
        embd.load_state_dict({"emb.weight": state_dict["tok_embeddings.weight"]})

    logger.info("Loading weights finished!")

    # Preprocess initial prompt inputs
    (
        input_tokens_prefill_tt,
        input_tokens_decode_tt,
        max_prompt_len,
        input_mask,
        input_tokens_prefill_pt,
        input_tokens_decode_pt,
        input_mask_pt,
        prefill_seq_len,
    ) = preprocess_inputs(input_prompts, tokenizer, model_args, dtype, instruct_mode, device_mesh)

    # TODO should we just change the pad after initial pad of the inputs?
    if instruct_mode:
        tokenizer._model.pad_id = tokenizer._model.eos_id

    # Load TTNN mixtral model
    logger.info("Loading weights to device...")
    tt_model = TtTransformer(
        device_mesh=device_mesh,
        state_dict=state_dict,
        args=model_args,
        layers=list(range(model_args.n_layers)),
        dtype=dtype,
    )

    if not embed_on_host:
        tt_embds = TtMixtralEmbedding(
            device_mesh=device_mesh,
            args=model_args,
            weight_cache_path=model_args.weight_cache_path(dtype),
            state_dict=state_dict,
            dtype=ttnn.bfloat16,  # Row major layout requires bfloat16
        )

    logger.info("Finished loading weights to device.")
    # Prepare the first token embedding for each user
    if embed_on_host:
        pt_decode_input = embd(input_tokens_decode_pt[:, 0]).view(batch_size, 1, -1)
        pt_prefill_input = [embd(input_tokens_prefill_pt[b, :]).view(1, prefill_seq_len, -1) for b in batch_size]
    else:  # Embedding on device
        # Each device does its own embedding
        decode_input_11BH = tt_embds(input_tokens_decode_tt[0])
        # Reshape and change row major to tile layout
        decode_input_11BH = ttnn.reshape(decode_input_11BH, ttnn.Shape([1, 1, batch_size, model_args.dim]))

        decode_input_11BH = ttnn.to_layout(decode_input_11BH, layout=ttnn.TILE_LAYOUT)
        # decode_input_11BH = [ttnn.experimental.tensor.tilize(decode_input_11BH[i]) for i in range(len(devices))]
        # decode_input_11BH = [ttnn.experimental.tensor.tilize_with_val_padding(decode_input_11BH[i], ) for i in range(len(devices))]")

    # PREFILL
    rot_mats_prefill = prepare_rotation_mat_ttnn(
        model_args.head_dim, model_args.max_seq_len, tt_model.device_mesh, mode="prefill", seq_len=prefill_seq_len
    )
    head_dim = model_args.dim // model_args.n_heads
    transformation_mat_torch = get_rot_transformation_mat(head_dim)
    transformation_mats = ttnn.as_tensor(
        transformation_mat_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=tt_model.device_mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ReplicateTensorToMesh(tt_model.device_mesh),
    )

    for batch_id in range(batch_size):
        prefill_input, attn_mask, _ = prepare_inputs_ttnn_prefill(
            pt_prefill_input[batch_id],
            model_args.dim,
            start_pos,
            model_args.sliding_window,
            tt_model.device_mesh,
        )
        tt_out = tt_model(
            prefill_input,
            start_pos,
            current_pos,
            attn_mask,
            rot_mats_prefill,
            transformation_mats,
            mode="prefill",
            user_id=batch_id,
        )

    del prefill_input, attn_mask, rot_mats_prefill, transformation_mats
    # DONE PREFILL
    # Prepare inputs for decode mode (rotary embeddings, attention mask, padding)
    rot_mats = prepare_rotation_mat_ttnn(
        model_args.head_dim,
        model_args.max_seq_len,
        tt_model.device_mesh,
    )

    generation_start_pos = prefill_seq_len
    max_generated_tokens = 50

    cache_attention(device_mesh, state_dict, model_args, rot_mats, generation_start_pos, max_generated_tokens, dtype)

    logger.info("Starting inference...")

    # Keep track of generated outputs to print out every iteration
    all_outputs = [[] for _ in range(batch_size)]

    # TODO Debug (only device 0 is doing argmax, otherwise it throws an error)
    # Alternatively, send the output back to device: tt_lib.tensor.Tensor.to()
    ttl.device.SetDefaultDevice(device_mesh.get_device(0))

    # Keep running inference as long as there is a user in the batch still decoding or max tokens per user are decoded
    for iteration in range(max_generated_tokens):
        iteration_time_start = time()
        start_pos = generation_start_pos + iteration
        current_pos = start_pos % model_args.sliding_window

        if embed_on_host:
            decode_input_11BH, attn_mask = prepare_inputs_ttnn(
                pt_decode_input,
                model_args.dim,
                start_pos,
                model_args.sliding_window,
                tt_model.device_mesh,
            )

        # Run ttnn mixtral model
        tt_out_11BH = tt_model(decode_input_11BH, start_pos, current_pos, attn_mask, rot_mats)
        # Work around program cache issue https://github.com/tenstorrent/tt-metal/issues/7159
        del decode_input_11BH, attn_mask
        if embed_on_host:
            # Convert ttnn tensor to torch tensor
            tt_output_torch = (
                ttnn.to_torch(tt_out_11BH, mesh_composer=ConcatMeshToTensor(device_mesh, dim=0))[0]
                .squeeze(1)
                .view(batch_size, 1, -1)
                .detach()
                .float()
            )
            # tt_token_batch = tt_output_torch.squeeze().argmax(axis=-1)
            # Argmax on host to get the new generated tokens
            tt_token_batch = sample(tt_output_torch, temperature=0, top_p=0.8)
            # Update the users that are still in prefill and the ones generating new tokens
            if iteration < max_prompt_len:
                tt_token_batch = torch.where(
                    input_mask_pt[:, iteration], input_tokens_decode_pt[:, iteration], tt_token_batch[:, 0]
                ).unsqueeze(1)
            # Next PT input embedding
            pt_decode_input = embd(tt_token_batch).view(batch_size, 1, -1)
        else:  # Embedding/argmax on device
            # TODO Update argmax to ttnn when OP becomes available
            tt_out_B11B = ttnn.experimental.tensor.argmax(tt_out_11BH, dim=-1)
            tt_out_1B = ttnn.reshape(tt_out_B11B[:1, :, :, :], ttnn.Shape([1, batch_size]))  # [1, 32] Bfloat16
            # Update the users that are still in prefill and the ones generating new tokens
            if iteration < max_prompt_len:
                decode_input_1B = ttnn.where(input_mask[iteration], input_tokens_decode_tt[iteration], tt_out_1B)
            else:
                decode_input_1B = tt_out_1B

            # Next TT input embeddings
            decode_input_1BH = tt_embds(decode_input_1B)
            decode_input_11BH = ttnn.reshape(decode_input_1BH, ttnn.Shape([1, 1, batch_size, model_args.dim]))
            decode_input_11BH = ttnn.to_layout(decode_input_11BH, layout=ttnn.TILE_LAYOUT)

            # Convert ttnn tensor to torch tensor and print decoded output (from a single device)
            # tt_output_torch = ttnn.to_torch(decode_input_1B).transpose(0, 1)
            tt_token_batch = ttnn.to_torch(decode_input_1B).transpose(0, 1)

        # Get the generated tokens for each user for printing in the log
        for user in range(batch_size):
            user_tok = int(tt_token_batch[user].item())
            if user_tok != tokenizer.eos_id:  # Stop saving the ouput after hitting the EOS token
                all_outputs[user].append(user_tok)

        iteration_time = time() - iteration_time_start
        tokens_per_second_per_user = 1 / iteration_time
        # Print out generated outputs for each user at the end of every iteration
        if os.getenv("CI") != "true":  # Avoid printing every iteration in CI
            if len(user_input) == 1:
                logger.info("[User 0] {}".format("".join(tokenizer.decode(all_outputs[0]))))
            else:
                for user in range(batch_size):
                    logger.info("[User {}] {}".format(user, "".join(tokenizer.decode(all_outputs[user]))))

        # Always print iteration perf
        logger.info(
            f"Iteration {iteration}: {1000*iteration_time:.2f}ms @ {tokens_per_second_per_user:.1f} tok/s/user ({batch_size*tokens_per_second_per_user:.1f} tok/s throughput)"
        )

    # In CI only print the final generated output to avoid spamming the logs
    if os.getenv("CI") == "true":
        if len(user_input) == 1:
            logger.info("[User 0] {}".format("".join(tokenizer.decode(all_outputs[0]))))
        else:
            for user in range(batch_size):
                logger.info("[User {}] {}".format(user, "".join(tokenizer.decode(all_outputs[user]))))


@pytest.mark.timeout(10000)
@pytest.mark.parametrize(
    "input_prompts, instruct_weights",
    [
        ("models/demos/t3000/mixtral8x7b/demo/input_data.json", False),
        ("models/demos/t3000/mixtral8x7b/demo/input_data_questions.json", True),
    ],
    ids=["general_weights", "instruct_weights"],
)
def test_mixtral8x7b_demo(t3k_device_mesh, use_program_cache, input_prompts, instruct_weights):
    return run_mixtral_demo(
        user_input=input_prompts, batch_size=32, device_mesh=t3k_device_mesh, instruct_mode=instruct_weights
    )
