# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import tt_lib as ttl
from models.utility_functions import comp_allclose_and_pcc, skip_for_wormhole_b0
from tests.tt_eager.python_api_testing.unit_testing.misc.test_moreh_matmul import get_tensors
from loguru import logger


# TODO: add this feature in get_tensors method
def get_bias_tensors(bias_shape, require_bias_grad, device):
    npu_dtype = ttl.tensor.DataType.BFLOAT16
    cpu_dtype = torch.bfloat16
    npu_layout = ttl.tensor.Layout.TILE
    cpu_layout = ttl.tensor.Layout.ROW_MAJOR

    bias = torch.randint(-10, 10, bias_shape, dtype=cpu_dtype)
    tt_bias = ttl.tensor.Tensor(bias, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)

    tt_bias_grad = None
    if require_bias_grad:
        bias_grad = torch.full(bias_shape, float("nan"), dtype=cpu_dtype)
        tt_bias_grad = ttl.tensor.Tensor(bias_grad, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)

    return tt_bias, bias, tt_bias_grad


@pytest.mark.parametrize(
    "shapes",
    (
        # input, weight, bias(1d or scalar), output
        ([1, 1, 1, 31], [1, 1, 30, 31], [1, 1, 1, 30], [1, 1, 1, 30]),
        ([1, 1, 1, 31], [1, 1, 30, 31], [1, 1, 1, 1], [1, 1, 1, 30]),
        ([1, 1, 31, 31], [1, 1, 30, 31], [1, 1, 1, 30], [1, 1, 31, 30]),
        ([1, 1, 31, 31], [1, 1, 30, 31], [1, 1, 1, 1], [1, 1, 31, 30]),
        ([4, 4, 2, 31], [1, 1, 30, 31], [1, 1, 1, 30], [4, 4, 2, 30]),
        ([4, 4, 2, 31], [1, 1, 30, 31], [1, 1, 1, 1], [4, 4, 2, 30]),
        ([1, 1, 2, 2047], [1, 1, 1023, 2047], [1, 1, 1, 1023], [1, 1, 2, 1023]),
        ([1, 1, 2, 2047], [1, 1, 1023, 2047], [1, 1, 1, 1], [1, 1, 2, 1023]),
        ([1, 1, 32, 64], [1, 1, 1024, 64], [1, 1, 1, 1024], [1, 1, 32, 1024]),
        ([1, 1, 32, 64], [1, 1, 1024, 64], [1, 1, 1, 1], [1, 1, 32, 1024]),
        ([1, 1, 32, 1023], [1, 1, 2047, 1023], [1, 1, 1, 2047], [1, 1, 32, 2047]),
        ([1, 1, 32, 1023], [1, 1, 2047, 1023], [1, 1, 1, 1], [1, 1, 32, 2047]),
        ([2, 4, 4, 1024], [1, 1, 2047, 1024], [1, 1, 1, 2047], [2, 4, 4, 2047]),
        ([2, 4, 4, 1024], [1, 1, 2047, 1024], [1, 1, 1, 1], [2, 4, 4, 2047]),
    ),
)
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("has_output", [False, True])
def test_moreh_linear(shapes, has_bias, has_output, device):
    torch.manual_seed(3072)
    input_shape, weight_shape, bias_shape, output_shape = shapes
    tt_input, tt_weight, _, _, _, _, torch_input, torch_weight, _ = get_tensors(
        input_shape, weight_shape, output_shape, False, False, False, device
    )

    npu_dtype = ttl.tensor.DataType.BFLOAT16
    npu_layout = ttl.tensor.Layout.TILE
    cpu_dtype = torch.bfloat16
    torch_output = torch.randint(-2, 3, output_shape, dtype=cpu_dtype)
    tt_output = (
        ttl.tensor.Tensor(torch_output, npu_dtype).pad_to_tile(1).to(npu_layout).to(device) if has_output else None
    )

    if has_bias:
        tt_bias, torch_bias, _ = get_bias_tensors(bias_shape, False, device)
    else:
        tt_bias, torch_bias = None, None

    ## TT Op
    tt_output = ttl.operations.primary.moreh_linear(tt_input, tt_weight, bias=tt_bias, output=tt_output)

    ## reference
    torch_output = torch.nn.functional.linear(torch_input, torch_weight[0][0], torch_bias)

    ## test for equivalance
    rtol = atol = 0.1
    cpu_layout = ttl.tensor.Layout.ROW_MAJOR
    ttcpu_output = tt_output.cpu().to(cpu_layout).unpad_from_tile(output_shape).to_torch()
    passing, output_pcc = comp_allclose_and_pcc(torch_output, ttcpu_output, pcc=0.999, rtol=rtol, atol=atol)
    logger.debug(f"Passing = {passing}")
    logger.debug(f"Output PCC = {output_pcc}")

    assert passing


def moreh_linear_backward(shapes, requires_input_grad, requires_weight_grad, requires_bias_grad, device):
    input_shape, weight_shape, bias_shape, output_shape = shapes
    if not requires_input_grad and not requires_weight_grad and not requires_bias_grad:
        pytest.skip("At least one grad is requires")

    (
        tt_input,
        tt_weight,
        _,
        tt_output_grad,
        tt_input_grad,
        tt_weight_grad,
        torch_input,
        torch_weight,
        torch_output_grad,
    ) = get_tensors(input_shape, weight_shape, output_shape, requires_input_grad, requires_weight_grad, False, device)

    tt_bias, torch_bias, tt_bias_grad = get_bias_tensors(bias_shape, requires_bias_grad, device)

    ## tt linear backward
    tt_input_grad, tt_weight_grad, tt_bias_grad = ttl.operations.primary.moreh_linear_backward(
        tt_output_grad,
        tt_input,
        tt_weight,
        are_required_outputs=(requires_input_grad, requires_weight_grad, requires_bias_grad),
        bias=tt_bias,
        input_grad=tt_input_grad,
        weight_grad=tt_weight_grad,
        bias_grad=tt_bias_grad,
    )
    ## reference
    torch_weight = torch_weight.reshape(-1, torch_weight.shape[3])
    torch_output = torch.nn.functional.linear(
        torch_input.requires_grad_(requires_input_grad),
        torch_weight.requires_grad_(requires_weight_grad),
        torch_bias.requires_grad_(requires_bias_grad),
    )
    torch_output.backward(torch_output_grad)

    ## test for equivalance
    rtol = atol = 0.1
    cpu_layout = ttl.tensor.Layout.ROW_MAJOR
    if requires_input_grad:
        ttcpu_input_grad = tt_input_grad.cpu().to(cpu_layout).unpad_from_tile(input_shape).to_torch()
        passing, output_pcc = comp_allclose_and_pcc(torch_input.grad, ttcpu_input_grad, pcc=0.999, rtol=rtol, atol=atol)
        logger.debug(f"input_grad passing={passing} pcc={output_pcc}")
        assert passing
    else:
        assert tt_input_grad is None

    if requires_weight_grad:
        ttcpu_weight_grad = tt_weight_grad.cpu().to(cpu_layout).unpad_from_tile(weight_shape).to_torch()[0][0]
        passing, output_pcc = comp_allclose_and_pcc(
            torch_weight.grad, ttcpu_weight_grad, pcc=0.999, rtol=rtol, atol=atol
        )
        logger.debug(f"weight_grad passing={passing} pcc={output_pcc}")
        assert passing
    else:
        assert tt_weight_grad is None

    if requires_bias_grad:
        ttcpu_bias_grad = tt_bias_grad.cpu().to(cpu_layout).unpad_from_tile(bias_shape).to_torch()

        passing, output_pcc = comp_allclose_and_pcc(torch_bias.grad, ttcpu_bias_grad, pcc=0.999, rtol=rtol, atol=atol)
        logger.debug(f"bias_grad passing={passing} pcc={output_pcc}")
        assert passing
    else:
        assert tt_bias_grad is None
    return passing


@pytest.mark.parametrize(
    "shapes",
    (
        # input, weight, bias(1d or scalar), output
        ([1, 1, 1, 31], [1, 1, 30, 31], [1, 1, 1, 30], [1, 1, 1, 30]),
        ([1, 1, 1, 31], [1, 1, 30, 31], [1, 1, 1, 1], [1, 1, 1, 30]),
        ([1, 1, 31, 31], [1, 1, 30, 31], [1, 1, 1, 30], [1, 1, 31, 30]),
        ([1, 1, 31, 31], [1, 1, 30, 31], [1, 1, 1, 1], [1, 1, 31, 30]),
        ([4, 4, 2, 31], [1, 1, 30, 31], [1, 1, 1, 30], [4, 4, 2, 30]),
        ([4, 4, 2, 31], [1, 1, 30, 31], [1, 1, 1, 1], [4, 4, 2, 30]),
        ([1, 1, 2, 2047], [1, 1, 1023, 2047], [1, 1, 1, 1023], [1, 1, 2, 1023]),
        ([1, 1, 2, 2047], [1, 1, 1023, 2047], [1, 1, 1, 1], [1, 1, 2, 1023]),
        ([1, 1, 32, 64], [1, 1, 1024, 64], [1, 1, 1, 1024], [1, 1, 32, 1024]),
        ([1, 1, 32, 64], [1, 1, 1024, 64], [1, 1, 1, 1], [1, 1, 32, 1024]),
        ([1, 1, 32, 1023], [1, 1, 1536, 1023], [1, 1, 1, 1536], [1, 1, 32, 1536]),
        ([1, 1, 32, 1023], [1, 1, 1536, 1023], [1, 1, 1, 1], [1, 1, 32, 1536]),
        ([2, 4, 4, 1024], [1, 1, 1536, 1024], [1, 1, 1, 1536], [2, 4, 4, 1536]),
        # TODO: #5868
        # ([2, 4, 4, 1024], [1, 1, 1200, 1024], [1, 1, 1, 1], [2, 4, 4, 1200]),
    ),
)
@skip_for_wormhole_b0("disabled due to watcher error, see issue #5868")
@pytest.mark.parametrize(
    "requires_grads",
    (
        (True, False),
        (False, True),
        (True, True),
    ),
)
@pytest.mark.parametrize("requires_bias_grad", [True, False])
def test_moreh_linear_backward(shapes, requires_grads, requires_bias_grad, device):
    torch.manual_seed(3072)
    requires_input_grad, requires_weight_grad = requires_grads
    passing = moreh_linear_backward(shapes, requires_input_grad, requires_weight_grad, requires_bias_grad, device)
    assert passing


@pytest.mark.parametrize(
    "shapes",
    (
        # input, weight, bias(1d or scalar), output
        ([1, 1, 31, 31], [1, 1, 30, 31], [1, 1, 1, 30], [1, 1, 31, 30]),
        ([1, 1, 31, 31], [1, 1, 30, 31], [1, 1, 1, 1], [1, 1, 31, 30]),
        ([2, 4, 4, 1024], [1, 1, 1536, 1024], [1, 1, 1, 1536], [2, 4, 4, 1536]),
        ([1, 1, 32, 1023], [1, 1, 1536, 1023], [1, 1, 1, 1], [1, 1, 32, 1536]),
    ),
)
@skip_for_wormhole_b0("disabled due to watcher error, see issue #5868")
def test_moreh_linear_backward_enable_cache(shapes, device, use_program_cache):
    torch.manual_seed(3072)
    requires_input_grad, requires_weight_grad, requires_bias_grad = (True, True, True)
    for i in range(2):
        passing = moreh_linear_backward(shapes, requires_input_grad, requires_weight_grad, requires_bias_grad, device)
        assert passing
