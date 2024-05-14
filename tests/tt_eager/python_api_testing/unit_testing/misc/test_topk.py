# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger

import tt_lib as ttl
from models.utility_functions import tt2torch_tensor, comp_pcc
from models.utility_functions import is_grayskull
import torch


def run_topk_test(N, C, H, W, k, dtype, device):
    torch.manual_seed(1234)
    shape = [N, C, H, W]
    torch_dtype = torch.bfloat16

    input = torch.randn(shape, dtype=torch_dtype)
    pyt_topk_values, pyt_topk_indices = torch.topk(input, k, dim=-1, largest=True, sorted=True)

    ttl_input = ttl.tensor.Tensor(input, dtype).to(ttl.tensor.Layout.TILE).to(device)
    ttl_topk_values, ttl_topk_indices = ttl.operations.primary.topk(ttl_input, k)

    assert list(ttl_topk_values.get_legacy_shape()) == [N, C, H, k]
    assert list(ttl_topk_indices.get_legacy_shape()) == [N, C, H, k]

    ttl_torch_topk_values = tt2torch_tensor(ttl_topk_values)
    ttl_torch_topk_indices = tt2torch_tensor(ttl_topk_indices)

    if dtype == ttl.tensor.DataType.BFLOAT8_B:
        pcc_values = 0.99
        pcc_index = 0.99
    else:
        pcc_index = 1.0
        pcc_values = 1.0

    # pcc is not a good measure for the raw indices
    # if index 49 and index 8 are tied, the order of the indices can be different
    # but the values associated with the indices should be the same
    # if index 7 and 8 are tied, but swapped, the pcc will be better than if index 49 and 8 are tied but swapped
    # so we will use pcc for the values and not the indices
    # to make sure the indices are correct, we gather the relevant values from the original torch tensor and test to see if they are similar
    # rounding may also cause more ties than expected
    ttl_torch_gather_from_device_indices = torch.gather(input, -1, ttl_torch_topk_indices.to(torch.int64))

    val_is_passing, val_pcc = comp_pcc(pyt_topk_values, ttl_torch_topk_values, pcc_values)
    ind_is_passing, ind_pcc = comp_pcc(pyt_topk_values, ttl_torch_gather_from_device_indices, pcc_index)

    # print("Pytorch and TTL TopK Values")
    # print(pyt_topk_values)
    # print(ttl_torch_topk_values)

    # print("Pytorch and TTL TopK Indices")
    # print(pyt_topk_indices)
    # print(ttl_torch_topk_indices)

    logger.debug(f"Values pcc = {val_pcc}")
    logger.debug(f"Indices pcc = {ind_pcc}")

    assert val_is_passing
    assert ind_is_passing


@pytest.mark.parametrize(
    "dtype",
    (
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.DataType.BFLOAT8_B,
        # ttl.tensor.DataType.FLOAT32, top bits in float32 get cut off somewhere, LLK does not work for this
    ),
    ids=[
        "BFLOAT16_B",
        "BFLOAT8_B",
        # "FLOAT32",
    ],
)
@pytest.mark.parametrize("N, C, H, W, k,", ((1, 1, 32, 64, 32),))
def test_topk(N, C, H, W, k, dtype, device):
    run_topk_test(N, C, H, W, k, dtype, device)
