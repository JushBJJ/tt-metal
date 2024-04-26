# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import ttnn
import traceback

from tests.ttnn.python_api_testing.sweep_tests import ttnn_ops
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

from ttnn.model_preprocessing import preprocess_model, preprocess_model_parameters
import transformers
from models.demos.bert.tt import ttnn_bert


def run_preprocess_model_parameters_tests(
    input_shape,
    dtype,
    dlayout,
    in_mem_config,
    output_mem_config,
    data_seed,
    device,
):
    torch.manual_seed(data_seed)
    x = torch.Tensor(size=input_shape[0]).uniform_(-0.1, 0.1).to(torch.bfloat16)

    model_name = "phiyodr/bert-large-finetuned-squad2"

    config = transformers.BertConfig.from_pretrained(model_name)
    model = transformers.models.bert.modeling_bert.BertAttention(config).eval()
    model = model.to(torch.bfloat16)

    torch_hidden_states = x
    sequence_size = x.shape[1]
    torch_attention_mask = torch.ones(1, sequence_size, dtype=torch.bfloat16)

    try:
        # run Torch model
        ref_value, *_ = model(torch_hidden_states, attention_mask=torch_attention_mask)

        # get parameters
        parameters = preprocess_model_parameters(
            initialize_model=lambda: model,
            device=device,
        )

        hidden_states = ttnn.from_torch(torch_hidden_states, dtype[0], layout=dlayout[0], device=device)
        attention_mask = ttnn.from_torch(torch_attention_mask, dtype[0], layout=dlayout[0], device=device)

        # get TT output
        tt_att_result = ttnn_bert.bert_attention(
            config,
            hidden_states,
            attention_mask=attention_mask,
            parameters=parameters,
        )

        # add deallocation
        ttnn.deallocate(hidden_states)
        ttnn.deallocate(attention_mask)

        tt_result = ttnn.to_torch(tt_att_result)

        # add deallocation
        ttnn.deallocate(tt_att_result)

    except Exception as e:
        logger.warning(f"Test execution crashed: {e}")
        print(traceback.format_exc())
        raise e

    assert len(tt_result.shape) == len(ref_value.shape)
    assert tt_result.shape == ref_value.shape

    # compare tt and golden outputs
    success, pcc_value = comp_pcc(ref_value, tt_result)
    logger.debug(pcc_value)
    logger.debug(success)

    assert success


test_sweep_args = [
    (
        [(4, 1216, 1024)],
        [ttnn.bfloat8_b],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        ttnn.DRAM_MEMORY_CONFIG,
        8687804,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (test_sweep_args),
)
def test_preprocess_model_parameters(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    run_preprocess_model_parameters_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device)
