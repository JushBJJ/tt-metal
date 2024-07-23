# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
import ttnn._ttnn
import ttnn.experimental_operations
import ttnn.operations
import ttnn.operations.pool
import ttnn.operations.reduction
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import torch_random, is_wormhole_b0

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_allclose
import ttnn.experimental

import ttnn.experimental.tensor

import ttnn.experimental.operations

import ttnn.experimental.operations.primary
import tt_lib

import inspect


# Example function
def example_function(a, b, c=3, *args, **kwargs):
    pass


# Function to print the arguments of the given function
def print_function_arguments(func):
    # Get the signature of the function
    signature = inspect.signature(func)

    # Print the parameters in a readable format
    print(f"Function '{func.__name__}' arguments:")
    for name, param in signature.parameters.items():
        print(f"  - {name}: {param}")


# @pytest.mark.parametrize("batch_size", [1, 16])
# @pytest.mark.parametrize("h", [32, 64])
# @pytest.mark.parametrize("w", [32, 64])
# @pytest.mark.parametrize("dim", [-1, -2, (2, 1)])
# def test_sum(device, batch_size, h, w, dim):
#     torch.manual_seed(0)
#     if is_wormhole_b0():
#         pytest.skip("Issue #6991: PCC mismatch")

#     torch_input_tensor = torch_random((batch_size, h, w), -100, 100, dtype=torch.bfloat16)
#     torch_output_tensor = torch.sum(torch_input_tensor, dim=dim, keepdim=True)

#     input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

#     output_tensor = ttnn.sum(input_tensor, dim=dim)
#     output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
#     output_tensor = ttnn.from_device(output_tensor)

#     output_tensor = ttnn.to_torch(output_tensor)
#     assert_with_pcc(torch_output_tensor, output_tensor)


# @pytest.mark.parametrize("batch_size", [1, 16])
# @pytest.mark.parametrize("h", [32, 64])
# @pytest.mark.parametrize("w", [32, 64])
# def test_sum_global(device, batch_size, h, w):
#     torch.manual_seed(0)
#     if is_wormhole_b0():
#         pytest.skip("Issue #6991: PCC mismatch")

#     torch_input_tensor = torch_random((batch_size, h, w), -100, 100, dtype=torch.bfloat16)
#     torch_output_tensor = torch.sum(torch_input_tensor)

#     input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

#     output_tensor = ttnn.sum(input_tensor)
#     output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
#     output_tensor = ttnn.from_device(output_tensor)

#     output_tensor = ttnn.to_torch(output_tensor)
#     output_tensor = output_tensor[0, 0, 0]

#     assert_with_pcc(torch_output_tensor, output_tensor)


@pytest.mark.parametrize(
    "dtype",
    (
        # ttl.tensor.DataType.BFLOAT4_B,
        # ttl.tensor.DataType.BFLOAT8_B,
        ttnn.DataType.BFLOAT16,
        # ttl.tensor.DataType.FLOAT32,
    ),
    # ids=("BFLOAT4_B", "BFLOAT8_B", "BFLOAT16", "FLOAT32"),
)
@pytest.mark.parametrize(
    "shape_dim",
    (
        ((1, 1, 32, 32), 3),
        # ((1, 1, 32, 1024), 3),
        # ((1, 1, 1024, 32), 2),
        # ((1, 1, 2048, 1024), 3),
        # ((1, 1, 32, 32), 2),
        # ((1, 1, 32, 1024), 2),
        # ((1, 1, 2048, 1024), 2),
    ),
)
def test_sum(device, shape_dim, dtype):
    shape, dim = shape_dim
    torch.manual_seed(42)

    N = shape[0]
    C = shape[1]
    H = shape[2]
    W = shape[3]

    input_shape = (N, C, H, W)
    if dtype == ttnn.DataType.FLOAT32:
        torch_input_tensor = torch.ones(input_shape)
    else:
        # torch_input_tensor = (torch.ones(input_shape) * 0.234375).bfloat16()
        torch_input_tensor = torch.randn(input_shape).bfloat16()
        for i in range(32):
            torch_input_tensor[0, 0, i, :] = i
        print(torch_input_tensor)

    torch_output_tensor = torch.sum(torch_input_tensor, dim=dim)
    print(torch_output_tensor.shape)
    torch_output_tensor = torch_output_tensor[0, 0, :]
    # print("torch scalar output")
    # print(torch_output_tensor[0].item())
    # print(torch_output_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    # reduce in w or h
    kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    output_tensor = ttnn.sum(input_tensor, dim=dim, compute_kernel_config=kernel_config)

    # print(output_tensor)
    # print(output_tensor.shape)
    output_tensor = ttnn.to_torch(output_tensor)
    # print(output_tensor.shape)
    # print(output_tensor)
    if dim == 3:
        output_tensor = output_tensor[0, 0, :, 0]
    else:
        output_tensor = output_tensor[0, 0, 0, :]

    # for i in range(32):
    #     print(output_tensor[i].item())
    #     print(" ")

    passing, output_str = comp_allclose(output_tensor, torch_output_tensor, rtol=1e-05, atol=1e-08)
    print(f"ref vs tt = {output_str}")

    # Compute the absolute difference between the tensors
    abs_diff = torch.abs(torch_output_tensor - output_tensor)

    # Find the maximum absolute difference
    max_abs_diff = torch.max(abs_diff)
    print(f"The maximum absolute difference is: {max_abs_diff:.4f}")

    # # Get the indices of the elements with the maximum absolute difference
    # max_diff_indices = torch.where(abs_diff == max_abs_diff)
    # print("The values with the largest absolute difference are:")

    # value1 = output_tensor[max_diff_indices[0]].to(torch.float32).item()
    # value2 = torch_output_tensor[max_diff_indices[0]].to(torch.float32).item()
    # print(f"Tensor 1 value: {value1:.4f}, Tensor 2 value: {value2:.4f}")

    assert passing
