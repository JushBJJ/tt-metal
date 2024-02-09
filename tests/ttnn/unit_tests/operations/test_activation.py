# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import torch.nn.functional as F

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


def run_activation_unary_test(device, h, w, ttnn_function, torch_function, pcc=0.99):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch_function(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn_function(input_tensor)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_hardtanh(device, h, w):
    run_activation_unary_test(device, h, w, ttnn.hardtanh, F.hardtanh)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_hardswish(device, h, w):
    run_activation_unary_test(device, h, w, ttnn.hardswish, F.hardswish)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_log_sigmoid(device, h, w):
    run_activation_unary_test(device, h, w, ttnn.log_sigmoid, F.logsigmoid)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_mish(device, h, w):
    run_activation_unary_test(device, h, w, ttnn.mish, lambda _x: F.mish(_x.to(torch.float)))


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_relu6(device, h, w):
    run_activation_unary_test(device, h, w, ttnn.relu6, F.relu6)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_sigmoid(device, h, w):
    run_activation_unary_test(device, h, w, ttnn.sigmoid, torch.sigmoid)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_sign(device, h, w):
    run_activation_unary_test(device, h, w, ttnn.sign, torch.sign)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_softsign(device, h, w):
    run_activation_unary_test(device, h, w, ttnn.softsign, F.softsign)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_swish(device, h, w):
    run_activation_unary_test(device, h, w, ttnn.swish, F.hardswish)


def torch_heaviside(x, *args, **kwargs):
    value = kwargs.pop("scalar")
    result = torch.heaviside(x, torch.tensor(value, dtype=x.dtype))
    return result


def torch_prelu(x, *args, **kwargs):
    weight = kwargs.pop("scalar")
    result = F.prelu(x, torch.tensor(weight, dtype=x.dtype))
    return result


def torch_relu_max(x, *args, **kwargs):
    upper_limit = kwargs.pop("scalar")
    capped_tensor = torch.min(x, torch.tensor(upper_limit, dtype=x.dtype))
    return torch.relu(capped_tensor)


def torch_relu_min(x, *args, **kwargs):
    lower_limit = kwargs.pop("scalar")
    capped_tensor = torch.max(x, torch.tensor(lower_limit, dtype=x.dtype))
    return torch.relu(capped_tensor)


def run_activation_test_scalarB(device, h, w, scalar, ttnn_function, torch_function, pcc=0.99):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand((h, w), dtype=torch.bfloat16)

    torch_output_tensor = torch_function(torch_input_tensor_a, scalar)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn_function(input_tensor_a, scalar)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


def run_activation_test_scalarB_key(device, h, w, scalar, ttnn_function, torch_function, pcc=0.99):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand((h, w), dtype=torch.bfloat16)

    torch_output_tensor = torch_function(torch_input_tensor_a, scalar=scalar)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn_function(input_tensor_a, scalar)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("scalar", [-0.5, 0, 0.5])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_scalarB_elu(device, h, w, scalar):
    run_activation_test_scalarB(device, h, w, scalar, ttnn.elu, F.elu)


@pytest.mark.parametrize("scalar", [0.5, 1.0])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_scalarB_hardshrink(device, h, w, scalar):
    run_activation_test_scalarB(device, h, w, scalar, ttnn.hardshrink, F.hardshrink)


@pytest.mark.parametrize("scalar", [0.88])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_scalarB_heaviside(device, h, w, scalar):
    run_activation_test_scalarB_key(device, h, w, scalar, ttnn.heaviside, torch_heaviside)


@pytest.mark.parametrize("scalar", [-0.5, 0, 0.5])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_scalarB_leaky_relu(device, h, w, scalar):
    run_activation_test_scalarB(device, h, w, scalar, ttnn.leaky_relu, F.leaky_relu)


@pytest.mark.parametrize("scalar", [-0.5, 1.0, 0.5])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_scalarB_prelu(device, h, w, scalar):
    run_activation_test_scalarB_key(device, h, w, scalar, ttnn.prelu, torch_prelu)


@pytest.mark.parametrize("scalar", [0.5])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_scalarB_softshrink(device, h, w, scalar):
    run_activation_test_scalarB(device, h, w, scalar, ttnn.softshrink, F.softshrink)


@pytest.mark.parametrize("scalar", [7.5])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_scalarB_relu_max(device, h, w, scalar):
    run_activation_test_scalarB_key(device, h, w, scalar, ttnn.relu_max, torch_relu_max)


@pytest.mark.parametrize("scalar", [-2.5])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_scalarB_relu_min(device, h, w, scalar):
    run_activation_test_scalarB_key(device, h, w, scalar, ttnn.relu_min, torch_relu_min)


def run_activation_test_scalarBC_key(device, h, w, scalar1, scalar2, ttnn_function, torch_function, pcc=0.99):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand((h, w), dtype=torch.bfloat16)

    torch_output_tensor = torch_function(torch_input_tensor_a, scalar1=scalar1, scalar2=scalar2)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn_function(input_tensor_a, scalar1, scalar2)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


def torch_clip(x, *args, **kwargs):
    min = kwargs.pop("scalar1")
    max = kwargs.pop("scalar2")
    return torch.clamp(x, min=min, max=max)


@pytest.mark.parametrize("scalar1", [-0.5, -0.1, -5.5])
@pytest.mark.parametrize("scalar2", [0.5, 1.5, 27.5])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_scalarBC_clip(device, h, w, scalar1, scalar2):
    run_activation_test_scalarBC_key(device, h, w, scalar1, scalar2, ttnn.clip, torch_clip)
