# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import tt_lib
import ttnn
from tests.tt_eager.python_api_testing.unit_testing.backward_ops.utility_funcs import (
    data_gen_with_range,
    compare_pcc,
)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        # (torch.Size([1, 1, 320, 384])),
        # (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_atan2(input_shapes, device):
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 100, device)
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, required_grad=True)
    other_data, other_tensor = data_gen_with_range(input_shapes, -100, 100, device, required_grad=True)

    pyt_y = torch.atan2(in_data, other_data)

    tt_output_tensor_on_device = ttnn.atan2_bw(grad_tensor, input_tensor, other_tensor, ttnn.DRAM_MEMORY_CONFIG)

    in_data.retain_grad()
    other_data.retain_grad()

    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad, other_data.grad]
    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass
