# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Union, Dict

import tt_lib as ttl

import ttnn

from tt_eager.tt_dnn.op_library.sliding_window_op_infra.tt_py_max_pool import (
    TTPyMaxPool,
    SlidingWindowOpParams,
)


class MaxPool2D:
    # kernel_size (Union[int, Tuple[int, int]]) – the size of the window to take a max over
    # stride (Union[int, Tuple[int, int]]) – the stride of the window. Default value is kernel_size
    # padding (Union[int, Tuple[int, int]]) – Implicit negative infinity padding to be added on both sides
    # dilation (Union[int, Tuple[int, int]]) – a parameter that controls the stride of elements in the window
    # return_indices (bool) – if True, will return the max indices along with the outputs. Useful for torch.nn.MaxUnpool2d later
    # ceil_mode (bool) – when True, will use ceil instead of floor to compute the output shape

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        dtype: ttnn.DataType = None,
        *,
        device: ttnn.Device,
        batch_size: int,
        input_height: int,
        input_width: int,
        reader_patterns_cache: Dict,
    ):
        if isinstance(kernel_size, int):
            window_h = kernel_size
            window_w = kernel_size
        else:
            window_h, window_w = kernel_size

        if isinstance(stride, int):
            stride_h = stride
            stride_w = stride
        else:
            stride_h, stride_w = stride

        if isinstance(padding, int):
            pad_h = padding
            pad_w = padding
        else:
            pad_h, pad_w = padding

        if isinstance(dilation, int):
            dilation_h = dilation
            dilation_w = dilation
        else:
            dilation_h, dilation_w = dilation
        assert dilation_h == 1, f"Only dilation_h = 1 supported. Found dilation_h={dilation_h}"
        assert dilation_w == 1, f"Only dilation_w = 1 supported. Found dilation_w={dilation_w}"

        sliding_window_op_params = SlidingWindowOpParams(
            stride_h=stride_h,
            stride_w=stride_w,
            pad_h=pad_h,
            pad_w=pad_w,
            window_h=window_h,
            window_w=window_w,
            batch_size=batch_size,
            input_h=input_height,
            input_w=input_width,
        )
        self.max_pool = TTPyMaxPool(sliding_window_op_params, device, reader_patterns_cache, pad_val=0xF7FF)

    def __call__(self, activation: ttnn.Tensor):
        return ttnn.Tensor(self.max_pool(activation.value))

    def copy_input_to_device(self, input: ttnn.Tensor):
        return ttnn.Tensor(self.max_pool.copy_input_to_device(input.value))

    def copy_output_from_device(self, output: ttnn.Tensor):
        return ttnn.Tensor(self.max_pool.copy_output_from_device(output.value))


def _torch_average_pool2d(input_tensor: ttnn.Tensor):
    import torch

    output_size = (1, 1)
    input_tensor = ttnn.from_device(input_tensor)
    input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
    input_tensor = ttnn.to_torch(input_tensor)

    return torch.nn.AdaptiveAvgPool2d(output_size)(input_tensor)


def _average_pool2d_validate_input_tensors(operation_name, input_tensor, *args, **kwargs):
    ttnn.validate_input_tensor(
        operation_name,
        input_tensor,
        ranks=(4,),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b, ttnn.uint16, ttnn.uint32),
        layouts=(ttnn.TILE_LAYOUT,),
        can_be_on_device=True,
        can_be_on_cpu=False,
    )


@ttnn.register_operation(
    name="ttnn.average_pool2d",
    validate_input_tensors=_average_pool2d_validate_input_tensors,
    torch_function=_torch_average_pool2d,
)
def average_pool2d(input_tensor: ttnn.Tensor) -> ttnn.Tensor:
    """
    average_pool2d(input_tensor: ttnn.Tensor) -> ttnn.Tensor
    """
    output = ttl.tensor.average_pool_2d(input_tensor.value)
    return ttnn.Tensor(output)
