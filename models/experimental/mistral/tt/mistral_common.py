# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from typing import Tuple
import tt_lib
from models.utility_functions import (
    tt2torch_tensor,
    torch2tt_tensor,
    nearest_32,
)


def tt_all_reduce(tensors, output_mem_config=None):
    """
    reduction on a list of tensors
    """
    if len(tensors) == 1:
        return tensors[0]
    base_tensor = tensors[0]
    for tensor in tensors[1:]:
        # base_tensor = tt_lib.tensor.add(base_tensor, tensor, output_mem_config=output_mem_config)  Cbinding doesnt support this optional argument passed in as None
        if output_mem_config is not None:
            base_tensor = tt_lib.tensor.add(base_tensor, tensor, output_mem_config)
        else:
            base_tensor = tt_lib.tensor.add(base_tensor, tensor)
    dev = base_tensor.device()
    # Emulate replication on all chips
    res_pt = tt2torch_tensor(base_tensor)
    res = [torch2tt_tensor(res_pt.clone(), dev) for _ in range(len(tensors))]
    return res


def generate_cos_sin_cache(
    tt_devices,
    head_dim,
    base_url,
    max_position_embeddings=2048,
    base=10000,
    model_config=None,
):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))

    t = torch.arange(
        max_position_embeddings,
        device=inv_freq.device,
        dtype=inv_freq.dtype,
    )
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    # Different from paper, but it uses a different permutation in order to obtain the same calculation
    emb = torch.cat((freqs, freqs), dim=-1)

    tt_cos_cached_host = torch2tt_tensor(
        emb.cos()[None, None, :, :],
        None,
        tt_memory_config=model_config["COS_CACHED_WEIGHTS_MEMCFG"],
        tt_dtype=model_config["COS_CACHED_WEIGHTS_DTYPE"],
    )
    tt_cos_cached = [
        tt_cos_cached_host.to(tt_device, model_config["COS_CACHED_WEIGHTS_MEMCFG"]) for tt_device in tt_devices
    ]
    tt_sin_cached_host = torch2tt_tensor(
        emb.sin()[None, None, :, :],
        None,
        tt_memory_config=model_config["SIN_CACHED_WEIGHTS_MEMCFG"],
        tt_dtype=model_config["SIN_CACHED_WEIGHTS_DTYPE"],
    )
    tt_sin_cached = [
        tt_sin_cached_host.to(tt_device, model_config["SIN_CACHED_WEIGHTS_MEMCFG"]) for tt_device in tt_devices
    ]

    return tt_cos_cached, tt_sin_cached


def precompute_freqs(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for sine and cosine values with given dimensions.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tensors containing cosine and sine values.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs).float()
    return torch.cos(freqs), torch.sin(freqs)


def freqs_to_rotation_matrix(cos_freqs, sin_freqs):
    """
    Transform cos/sin frequencies to a rotation matrix.
    """
    emb_size, emb_dim = cos_freqs.shape
    dhead = emb_dim * 2
    rot_emb_matrix = torch.zeros(emb_size, dhead, dhead)
    rot_emb_matrix[..., torch.arange(0, dhead, 2), torch.arange(0, dhead, 2)] = cos_freqs.clone()
    rot_emb_matrix[..., torch.arange(1, dhead, 2), torch.arange(1, dhead, 2)] = cos_freqs.clone()
    rot_emb_matrix[..., torch.arange(0, dhead, 2), torch.arange(1, dhead, 2)] = -sin_freqs.clone()
    rot_emb_matrix[..., torch.arange(1, dhead, 2), torch.arange(0, dhead, 2)] = sin_freqs.clone()

    rot_emb_matrix = rot_emb_matrix.transpose(-1, -2)  # Necessary for correct rotation when applied as (x @ R)
    return rot_emb_matrix


def prepare_inputs(x, start_pos, hidden_size, n_local_heads, sliding_window, devices, num_devices):
    """
    Prepare inputs for decode mode. Assume that current token is at
    start_pos, and KV cache has valid data up to start_pos.
    x: (batch, seq, hidden_dim)
    start_pos: int
    """
    assert x.size(2) == hidden_size
    assert len(x.size()) == 3

    batch = x.size(0)
    seq_len = x.size(1)
    assert seq_len == 1, "Only supporting decode mode"
    x = x.transpose(0, 1).unsqueeze(1)  # [seq_len, 1, batch, hidden_dim]

    padded_layer_past_len = min(nearest_32(start_pos + 1), sliding_window)
    current = start_pos % sliding_window
    attn_mask = torch.zeros(seq_len, 1, batch, padded_layer_past_len)

    if start_pos < sliding_window:
        attn_mask[:, :, :, current + 1 :] = torch.finfo(attn_mask.dtype).min
    else:
        attn_mask[:, :, :, :current] = torch.finfo(attn_mask.dtype).min
        attn_mask[:, :, :, sliding_window - current :] = torch.finfo(attn_mask.dtype).min
    attn_mask = attn_mask.expand(-1, n_local_heads, -1, -1)

    # expected shapes:
    # x: (seq_len, 1, batch, hidden_dim)
    # start_pos: int
    # attn_mask: [seq_len, n_heads, batch, padded_layer_past_len]
    assert x.size() == (seq_len, 1, batch, hidden_size)
    assert attn_mask.size() == (seq_len, n_local_heads, batch, padded_layer_past_len)

    xs, attn_masks = [], []
    for i in range(num_devices):
        device = devices[i]
        xs.append(torch2tt_tensor(x.clone(), device))
        attn_masks.append(torch2tt_tensor(attn_mask.clone(), device))

    return (
        xs,
        start_pos,
        attn_masks,
        current,
    )
