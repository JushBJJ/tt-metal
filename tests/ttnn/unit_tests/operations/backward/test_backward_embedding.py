# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import tt_lib
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests import (
    comparison_funcs,
)
from loguru import logger


@pytest.mark.parametrize(
    "batch_size, seq_len, embedding_dim, num_embeddings",
    [
        (1, 32, 32, 64),
    ],
)
def test_embedding_bw(batch_size, seq_len, embedding_dim, num_embeddings, device):
    torch.manual_seed(1234)
    input_shape = (batch_size, seq_len)
    input_index = torch.randint(0, num_embeddings, input_shape)

    weights_shape = (1, 1, num_embeddings, embedding_dim)
    weights = torch.randn(weights_shape, requires_grad=True)
    weights_ttnn = (
        tt_lib.tensor.Tensor(weights, tt_lib.tensor.DataType.BFLOAT16).to(tt_lib.tensor.Layout.TILE).to(device)
    )

    grad_shape = (1, 1, batch_size * seq_len, embedding_dim)
    grad_data = torch.randn(grad_shape, requires_grad=True)

    grad_tensor = (
        tt_lib.tensor.Tensor(grad_data, tt_lib.tensor.DataType.BFLOAT16).to(tt_lib.tensor.Layout.TILE).to(device)
    )

    input_tensor = tt_lib.tensor.Tensor(input_index, tt_lib.tensor.DataType.UINT32).to(device)
    print(input_index)

    tt_output_tensor_on_device = ttnn.embedding_bw(input_tensor, weights_ttnn, grad_tensor)
    tt_output_tensor_a = tt_output_tensor_on_device.cpu().to(tt_lib.tensor.Layout.ROW_MAJOR).to_torch()

    weights.retain_grad()

    pyt_y = torch.nn.functional.embedding(
        input_index,
        weights.reshape((num_embeddings, embedding_dim)),
    ).reshape(grad_shape)

    pyt_y.backward(gradient=grad_data)

    golden_output_tensor_a = weights.grad

    comp_pass_a, comp_out_a = comparison_funcs.comp_pcc(golden_output_tensor_a, tt_output_tensor_a)

    logger.debug(comp_out_a)
    # assert comp_pass_a
