# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import tt_lib as ttl
import pytest
from models.utility_functions import comp_allclose_and_pcc
from loguru import logger
from models.utility_functions import is_wormhole_b0

from tests.tt_eager.python_api_testing.unit_testing.misc.test_utils import (
    get_compute_kernel_options,
    compute_kernel_options,
    compute_kernel_ids,
)


@pytest.mark.parametrize(
    "shape_dim",
    (((32, 32), 1),),  # single tile
)
def test_softmax_for_dim_hw(shape_dim, device):
    device.enable_program_cache()

    compute_kernel_options = True
    shape, dim = shape_dim
    torch.manual_seed(0)

    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)

    # The x tensor is not actually used here.
    # For testing purposes, the tensor's values are being generated within the reader data movement kernel
    x = torch.ones(shape, dtype=torch.bfloat16) * 0.0

    dev_x = ttl.tensor.Tensor(x, ttl.tensor.DataType.BFLOAT16).to(ttl.tensor.Layout.TILE).to(device)

    tt_npu = ttl.operations.primary.moreh_softmax(dev_x, dim, compute_kernel_config=compute_kernel_config)

    tt_dev = tt_npu.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch().to(torch.bfloat16)

    print("tt_dev", tt_dev)
