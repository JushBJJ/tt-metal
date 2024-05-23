# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import torch
from torch import nn
import ttnn.experimental as tt_lib
import ttnn

from models.experimental.llama2_70b.reference.llama.llama import Llama
from models.experimental.llama2_70b.reference.llama.llama.model import precompute_freqs_cis, apply_rotary_emb
from models.experimental.llama2_70b.tt.model_config import (
    get_model_config,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_pcc,
)
from models.utility_functions import torch2tt_tensor, tt2torch_tensor, skip_for_grayskull, get_devices_for_t3000
from models.experimental.llama2_70b.tt.llama_common import (
    get_llama_path,
)

from models.experimental.llama2_70b.tt.llama_common import precompute_freqs, freqs_to_rotation_matrix, gather_rotary_emb


def get_rotation_mat(dhead, end, start_pos, seqlen, batch):
    cos, sin = precompute_freqs(dhead, end)
    rot_mat = freqs_to_rotation_matrix(cos, sin)
    position_ids = torch.ones(seqlen, batch, dtype=torch.long) * start_pos
    rot_emb = gather_rotary_emb(rot_mat, position_ids)
    return rot_emb


class TtLlamaRotary(torch.nn.Module):
    def __init__(
        self,
        device,
        head_dim: int,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.device = device

        tile_width = 32

        self.transformation_mat = torch2tt_tensor(get_rot_transformation_mat(dhead=tile_width), device)

    def apply_rotary(self, x, cos, sin):
        # n_head = 8 for Q
        # n_head = 1 for K

        compute_kernel_config = tt_lib.tensor.WormholeComputeKernelConfig(
            # math_fidelity=ttl.tensor.MathFidelity.LoFi,
            math_fidelity=tt_lib.tensor.MathFidelity.HiFi4,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        rotary_output = tt_lib.tensor.rotary_embedding_llama(
            x, cos, sin, self.transformation_mat, compute_kernel_config=compute_kernel_config
        )

        return rotary_output

    def forward(self, xq, xk, cos, sin):
        xq = self.apply_rotary(xq, cos, sin)
        xk = self.apply_rotary(xk, cos, sin)
        return xq, xk


class PytorchLlamaRotaryModel(torch.nn.Module):
    def __init__(self, hf_reference_model, layer_num):
        super().__init__()
        self.n_heads = hf_reference_model.params.n_heads
        self.n_kv_heads = hf_reference_model.params.n_kv_heads
        self.head_dim = hf_reference_model.params.dim // self.n_heads

    def forward(self, xq, xk, freqs_cis):
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)

        return xq, xk


def get_rot_transformation_mat(dhead):
    rot_emb_matrix = torch.zeros(1, 1, dhead, dhead)
    rot_emb_matrix[..., torch.arange(0, dhead, 2), torch.arange(1, dhead, 2)] = 1
    rot_emb_matrix[..., torch.arange(1, dhead, 2), torch.arange(0, dhead, 2)] = -1
    return rot_emb_matrix


def compute_gather_cos_sin(dhead, end, position_ids):
    cos, sin = precompute_freqs(dhead, end)
    position_id_expanded = position_ids.unsqueeze(1).expand(-1, cos.shape[-1])
    cos = cos.gather(0, position_id_expanded)
    sin = sin.gather(0, position_id_expanded)
    cos = torch.stack([cos, cos], dim=-1).flatten(-2).unsqueeze(0).unsqueeze(0)
    sin = torch.stack([sin, sin], dim=-1).flatten(-2).unsqueeze(0).unsqueeze(0)
    return cos, sin


def run_test_rotary_embedding_llamma(
    devices,
    batch,
    seq_len,
    pcc,
    n_heads,
    n_kv_heads,
    head_dim,
    max_seq_len,
    datatype=ttnn.bfloat16,
):
    device = devices[0]

    # Prepare input
    torch.manual_seed(0)
    inp = [
        (torch.rand(batch, n_heads, seq_len, head_dim) * 2) - 1,
        (torch.rand(batch, n_kv_heads, seq_len, head_dim) * 2) - 1,
    ]
    freqs_cis = precompute_freqs_cis(
        # Note that self.params.max_seq_len is multiplied by 2 because the token limit for the Llama 2 generation of models is 4096.
        # Adding this multiplier instead of using 4096 directly allows for dynamism of token lengths while training or fine-tuning.
        head_dim,
        max_seq_len * 2,
    )  # torch.Size([8192, 64])

    start_pos = 0  # Must pick non-zero start pos to get non-zero freqs_cis
    freqs_cis = freqs_cis[start_pos : start_pos + seq_len]

    # PyTorch Ground Truth output --------------------------------------------------------------------
    torch_xq = inp[0].transpose(1, 2)
    torch_xk = inp[1].transpose(1, 2)

    torch_xq, torch_xk = apply_rotary_emb(torch_xq, torch_xk, freqs_cis=freqs_cis)

    torch_xq = torch_xq.transpose(1, 2)
    torch_xk = torch_xk.transpose(1, 2)

    pytorch_out = (torch_xq, torch_xk)

    # TT hardware / Modified PyTorch execution -------------------------------------------------------------
    tt_model = TtLlamaRotary(
        device,
        head_dim,
    )

    cos, sin = compute_gather_cos_sin(
        dhead=head_dim, end=max_seq_len * 2, position_ids=torch.arange(start_pos, start_pos + seq_len)
    )
    tt_inp = [inp[0], inp[1], cos, sin]
    tt_inp = [torch2tt_tensor(i, device, tt_dtype=datatype) for i in tt_inp]

    tt_out = tt_model(*tt_inp)
    tt_out = [tt2torch_tensor(tt_out_tensor) for tt_out_tensor in tt_out]

    # check outputs ----------------------------------------------------------------------
    does_pass = True
    for i in range(2):
        out_pass, output_pcc = comp_pcc(pytorch_out[i], tt_out[i], pcc)
        # Check each shape matches
        assert pytorch_out[i].shape == tt_out[i].shape
        logger.info(f"PCC value: {output_pcc}")
        does_pass = does_pass and out_pass

        mae = torch.mean(torch.abs(pytorch_out[i] - tt_out[i]))
        logger.info(f"MAE: {mae}")

        max_incorrect = torch.max(torch.abs(pytorch_out[i] - tt_out[i]))
        logger.info(f"Max incorrect: {max_incorrect}")

        max_gt = torch.max(torch.abs(pytorch_out[i]))
        logger.info(f"Max ground truth: {max_gt}")

    if does_pass:
        logger.info("Llama QKV output Passed!")
    else:
        logger.warning("Llama QKV output Failed!")
        assert does_pass, f"PCC value is lower than {pcc}"


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "batch, seq_len",
    (
        (1, 128),
        (1, 256),
        (1, 512),
        (1, 2048),
        (1, 4096),
        # (1, 8192),
    ),
    ids=(
        "prefill_128",
        "prefill_256",
        "prefill_512",
        "prefill_2k",
        "prefill_4k",
        # "prefill_8k",
    ),
)
@pytest.mark.parametrize(
    "n_heads, n_kv_heads, head_dim",
    (
        (8, 1, 64),
        (8, 1, 128),
        (11, 3, 128),
        (71, 32, 64),
        (8, 1, 96),
    ),
)
@pytest.mark.parametrize("datatype", (ttnn.bfloat16,))
@pytest.mark.parametrize("pcc", (0.9997,))
def test_LlamaAttention_inference(
    batch,
    seq_len,
    n_heads,
    n_kv_heads,
    head_dim,
    datatype,
    pcc,
    all_devices,
    use_program_cache,
):
    devices = all_devices
    compute_grid_size = devices[0].compute_with_storage_grid_size()
    if compute_grid_size.x < 8 or compute_grid_size.y < 8:
        pytest.skip(f"Requires grid size of at least {(8, 8)} to run")

    # Constants
    max_seq_len = max(4096, seq_len)

    for i in range(3 if use_program_cache else 1):
        run_test_rotary_embedding_llamma(
            devices, batch, seq_len, pcc, n_heads, n_kv_heads, head_dim, max_seq_len, datatype
        )

        # shift input/output tensor by creating very small tensor between loop
        inp = torch.rand(1, 1, 32, 32)
        test_tensor = (
            tt_lib.tensor.Tensor(
                inp.reshape(-1).tolist(),
                inp.shape,
                ttnn.bfloat16,
                tt_lib.tensor.Layout.ROW_MAJOR,
            )
            .to(tt_lib.tensor.Layout.TILE)
            .to(devices[0])
        )

    assert devices[0].num_program_cache_entries() == 2
