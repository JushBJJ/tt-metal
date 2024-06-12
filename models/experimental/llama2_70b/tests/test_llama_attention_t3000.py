# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

from models.utility_functions import skip_for_grayskull
from models.experimental.llama2_70b.tt.llama_common import setup_llama_env, check_device_mesh
from models.experimental.llama2_70b.tests.test_llama_attention import run_test_LlamaAttention_inference


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "llama_version",
    (("llama3"),),
)
@pytest.mark.parametrize(
    "batch, seq_len, pcc",
    ((32, 1, 0.9997), (1, 128, 0.9997), (1, 2048, 0.9997), (1, 8192, 0.9997)),
    ids=("decode", "prefill_128", "prefill_2k", "prefill_8k"),
)
def test_LlamaAttention_inference_t3000(
    batch,
    seq_len,
    pcc,
    t3k_device_mesh,
    llama_version,
    use_program_cache,
):
    model_config, ckpt_dir, tokenizer_path, cache_path = setup_llama_env(
        llama_version=llama_version, batch=batch, seq_len=seq_len
    )

    check_device_mesh(t3k_device_mesh, model_config)
    run_test_LlamaAttention_inference(
        t3k_device_mesh,
        batch,
        seq_len,
        pcc,
        model_config,
        llama_version,
        ckpt_dir,
        tokenizer_path,
        cache_path,
    )
