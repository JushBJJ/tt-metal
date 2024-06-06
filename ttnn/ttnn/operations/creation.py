# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Union


import ttnn


def _golden_function(input_tensor: ttnn.Tensor, **_):
    import torch

    return torch.zeros_like(input_tensor)


zeros_like = ttnn.register_operation(golden_function=_golden_function)(ttnn._ttnn.operations.creation.zeros_like)


def _golden_function(input_tensor: ttnn.Tensor, **_):
    import torch

    return torch.ones_like(input_tensor)


ones_like = ttnn.register_operation(golden_function=_golden_function)(ttnn._ttnn.operations.creation.ones_like)


def _golden_function(input_tensor: ttnn.Tensor, *, fill_value: float, **_):
    import torch

    return torch.full_like(input_tensor, fill_value)


full_like = ttnn.register_operation(golden_function=_golden_function)(ttnn._ttnn.operations.creation.full_like)


def _golden_function(input_tensor: ttnn.Tensor, *, fill_value: float, **_):
    import torch

    return torch.empty_like(input_tensor, fill_value)


empty_like = ttnn.register_operation(golden_function=_golden_function)(ttnn._ttnn.operations.creation.empty_like)


def _golden_function(input_shape: ttnn.Shape, **_):
    import torch

    return torch.zeros(input_shape)


zeros = ttnn.register_operation(golden_function=_golden_function)(ttnn._ttnn.operations.creation.zeros)


def _golden_function(input_shape: ttnn.Shape, **_):
    import torch

    return torch.ones(input_shape)


ones = ttnn.register_operation(golden_function=_golden_function)(ttnn._ttnn.operations.creation.ones)


def _golden_function_full(input_shape: ttnn.Shape, fill_value: float, **_):
    import torch

    return torch.full(input_shape, fill_value=fill_value)


full = ttnn.register_operation(golden_function=_golden_function)(ttnn._ttnn.operations.creation.full)


def _golden_function(input_shape: ttnn.Shape, **_):
    import torch

    return torch.empty(input_shape)


empty = ttnn.register_operation(golden_function=_golden_function)(ttnn._ttnn.operations.creation.empty)


def _golden_function(start: int, end: int, step: int, **_):
    import torch

    return torch.arange(start, end, step)


def _arange_validate_input_tensors(operation_name, input_shape, *args, **kwargs):
    ...


@ttnn.register_operation(
    name="ttnn.arange",
    validate_input_tensors=_arange_validate_input_tensors,
    golden_function=_golden_function,
)
def arange(
    start: int,
    end: int,
    step: int,
    device,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    r"""

    arange(start: int, end: int, step: int, device, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor

    Returns a new 1D tensor with the incremented values in size specified by inputs start, end and step.

    Args:
        * :attr:`start`
        * :attr:`end`
        * :attr:`step`
    """

    output_tensor = ttnn.experimental.tensor.arange(start, end, step, device, output_mem_config=memory_config)

    return output_tensor


__all__ = []
