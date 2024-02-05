# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


def prepare_attention_mask(attention_mask, target_length, heads=8):
    head_size = heads
    if attention_mask is None:
        return attention_mask

    if attention_mask.shape[-1] != target_length:
        assert False, "Attention Mask has always been None, This is not implemented!"

    return attention_mask


def batch_to_head_dim(tensor, heads=8):
    head_size = heads
    _, batch_size, seq_len, dim = tensor.shape
    tensor = ttnn.to_layout(
        tensor, layout=ttnn.ROW_MAJOR_LAYOUT
    )  # TILE_LAYOUT is not compatible with tensor shape, hence we used ROW_MAJOR_LAYOUT.
    tensor = ttnn.reshape(tensor, (batch_size // head_size, head_size, seq_len, dim))
    tensor = ttnn.permute(tensor, (0, 2, 1, 3))
    tensor = ttnn.reshape(tensor, (1, batch_size // head_size, seq_len, dim * head_size))
    tensor = ttnn.to_layout(tensor, ttnn.TILE_LAYOUT)
    return tensor


def head_to_batch_dim(tensor, heads=8):
    head_size = heads
    _, batch_size, seq_len, dim = tensor.shape
    tensor = ttnn.to_layout(
        tensor, layout=ttnn.ROW_MAJOR_LAYOUT
    )  # TILE_LAYOUT is not compatible with tensor shape, hence we used ROW_MAJOR_LAYOUT.
    tensor = ttnn.reshape(tensor, (batch_size, seq_len, head_size, dim // head_size))
    tensor = ttnn.permute(tensor, (0, 2, 1, 3))
    tensor = ttnn.reshape(tensor, (1, batch_size * head_size, seq_len, dim // head_size))
    tensor = ttnn.to_layout(tensor, ttnn.TILE_LAYOUT)
    return tensor


def get_attention_scores(query, key, attention_mask=None, scale=64, device=None):
    t_key = ttnn.permute(key, (0, 1, 3, 2))
    temp = ttnn.matmul(query, t_key)

    scale_tensor = query @ t_key  # instead of full
    scale_tensor = ttnn.mul(scale_tensor, scale)
    attention_scores = ttnn.mul(scale_tensor, temp)

    if attention_mask is not None:
        attention_scores = ttnn.add(attention_scores, attention_mask)

    attention_probs = ttnn.softmax(attention_scores, dim=-1)

    return attention_probs


def cross_attention(
    hidden_states,
    encoder_hidden_states,
    query_dim: int = None,
    cross_attention_dim=None,
    heads: int = 8,
    dim_head: int = 64,
    attention_mask=None,
    cross_attention_kwargs={},
    *,
    parameters,
    device,
):
    _, batch_size, sequence_length, _ = hidden_states.shape
    attention_mask = prepare_attention_mask(attention_mask, sequence_length)
    query_weight = parameters.to_q.weight

    hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)
    query = ttnn.matmul(hidden_states, query_weight)

    query = head_to_batch_dim(query)

    encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

    key_weight = parameters.to_k.weight
    encoder_hidden_states = ttnn.to_layout(encoder_hidden_states, ttnn.TILE_LAYOUT)
    key = ttnn.matmul(encoder_hidden_states, key_weight)

    if len(key.shape) <= 3:
        key = ttnn.from_device(key)
        key = ttnn.to_torch(key).unsqueeze(0)
        key = ttnn.from_torch(key)
        key = ttnn.to_device(key, device)

    value_weight = parameters.to_v.weight
    value = ttnn.matmul(encoder_hidden_states, value_weight)

    if len(value.shape) <= 3:
        value = ttnn.from_device(value)
        value = ttnn.to_torch(value).unsqueeze(0)
        value = ttnn.from_torch(value)
        value = ttnn.to_device(value, device)

    key = head_to_batch_dim(key)

    value = head_to_batch_dim(value)

    attention_probs = get_attention_scores(query, key, attention_mask, device=device)

    padding_needed = attention_probs.shape[-1] - value.shape[-2]
    value = ttnn.pad(value, ((0, padding_needed), (0, 0)), 0)

    hidden_states = ttnn.matmul(attention_probs, value)

    hidden_states = batch_to_head_dim(hidden_states)

    out_weight = parameters.to_out[0].weight

    hidden_states = ttnn.matmul(hidden_states, out_weight)
    if parameters.to_out[0].bias is not None:
        out_bias = parameters.to_out[0].bias
        hidden_states = ttnn.add(hidden_states, out_bias)

    return hidden_states
