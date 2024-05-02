# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List, Union, Optional

import sys

import ttnn

import tt_lib as ttl

THIS_MODULE = sys.modules[__name__]

__all__ = []


def register_ttl_binary_function(name, ttl_binary_function, doc):
    def _golden_function(input_tensor: ttnn.Tensor, parameter, **_):
        import torch

        name_to_torch_function = {"pow": torch.pow}
        torch_function = name_to_torch_function[name]
        return torch_function(input_tensor, parameter)

    def _binary_validate_input_tensors(operation_name, input_tensor, *args, **kwargs):
        ttnn.validate_input_tensor(
            operation_name,
            input_tensor,
            ranks=(2, 3, 4),
            dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
            layouts=(ttnn.TILE_LAYOUT,),
            can_be_on_device=True,
            can_be_on_cpu=False,
        )

    @ttnn.register_operation(
        name=f"ttnn.{name}",
        validate_input_tensors=_binary_validate_input_tensors,
        golden_function=_golden_function,
    )
    def binary_function(
        input_tensor: ttnn.Tensor, parameter: float, *, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG
    ) -> ttnn.Tensor:
        original_shape = input_tensor.shape
        input_tensor = ttnn.unsqueeze_to_4D(input_tensor)
        output_tensor = ttl_binary_function(input_tensor, parameter, output_mem_config=memory_config)
        output_tensor = ttnn.reshape(output_tensor, original_shape)
        return output_tensor

    if isinstance(binary_function, ttnn.decorators.Operation):
        binary_function.__name__ = f"ttnn.{name}"
        binary_function.decorated_function.__doc__ = doc + (
            binary_function.__doc__ if binary_function.__doc__ is not None else ""
        )

    setattr(THIS_MODULE, name, binary_function)


TTL_BINARY_FUNCTIONS = [
    (
        "pow",
        ttnn.experimental.tensor.pow,
        r"""pow(input_tensor: ttnn.Tensor, exponent: Union[ttnn.Tensor, float, int]) -> ttnn.Tensor

        Takes the power of each element in input with exponent and returns a tensor with the result.

        .. math::
            pow(\mathrm{{input\_tensor}}_i, \mathrm{{exponent}})

        Args:
            * :attr:`input_tensor`
            * :attr:`exponent`

        Example::

            >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> output = ttnn.pow(tensor, 2)

        """,
    ),
]


for binary_function_name, ttl_binary_function, doc in TTL_BINARY_FUNCTIONS:
    register_ttl_binary_function(binary_function_name, ttl_binary_function, doc)


def apply_activations(tensor, activations):
    import torch

    string_to_function = {
        "relu": torch.relu,
        "gelu": torch.nn.functional.gelu,
        "silu": torch.nn.functional.silu,
    }

    if activations is not None:
        for activation in activations:
            activation_function = string_to_function[activation]
            tensor = activation_function(tensor)
    return tensor


def _golden_function(input_tensor_a, input_tensor_b, *args, activations=None, **kwargs):
    output_tensor = input_tensor_a + input_tensor_b
    return apply_activations(output_tensor, activations)


add = ttnn.register_operation(golden_function=_golden_function)(ttnn._ttnn.operations.binary.add)
add_ = ttnn.register_operation(golden_function=_golden_function)(ttnn._ttnn.operations.binary.add_)


def _golden_function(input_tensor_a, input_tensor_b, *args, activations=None, **kwargs):
    output_tensor = input_tensor_a - input_tensor_b
    return apply_activations(output_tensor, activations)


subtract = ttnn.register_operation(golden_function=_golden_function)(ttnn._ttnn.operations.binary.subtract)
subtract_ = ttnn.register_operation(golden_function=_golden_function)(ttnn._ttnn.operations.binary.subtract_)


def _golden_function(input_tensor_a, input_tensor_b, *args, activations=None, **kwargs):
    output_tensor = input_tensor_a * input_tensor_b
    return apply_activations(output_tensor, activations)


multiply = ttnn.register_operation(golden_function=_golden_function)(ttnn._ttnn.operations.binary.multiply)
multiply_ = ttnn.register_operation(golden_function=_golden_function)(ttnn._ttnn.operations.binary.multiply_)

sub = subtract
mul = multiply
sub_ = subtract_
mul_ = multiply_

ttnn.Tensor.__add__ = lambda self, *args, **kwargs: add(self, *args, **kwargs)
ttnn.Tensor.__radd__ = lambda self, *args, **kwargs: add(self, *args, **kwargs)
ttnn.Tensor.__sub__ = lambda self, *args, **kwargs: sub(self, *args, **kwargs)
ttnn.Tensor.__mul__ = lambda self, *args, **kwargs: mul(self, *args, **kwargs)
ttnn.Tensor.__rmul__ = lambda self, *args, **kwargs: mul(self, *args, **kwargs)


def register_ttl_elt_binary_function(name, ttl_elt_binary_function, op_name):
    def _golden_function(input_tensor_a: ttnn.Tensor, input_tensor_b: ttnn.Tensor, **_):
        import torch

        name_to_torch_function = {
            "ldexp": torch.ldexp,
            "logaddexp": torch.logaddexp,
            "logaddexp2": torch.logaddexp2,
            "logical_and": torch.logical_and,
            "logical_or": torch.logical_or,
            "logical_xor": torch.logical_xor,
            "xlogy": torch.xlogy,
            "maximum": torch.maximum,
            "minimum": torch.minimum,
        }
        torch_function = name_to_torch_function[name]
        return torch_function(input_tensor_a, input_tensor_b)

    def _elt_binary_validate_input_tensors(operation_name, input_tensor_a, input_tensor_b, *args, **kwargs):
        if operation_name == "ttnn.ldexp":
            supported_dtypes = (ttnn.bfloat16,)
        else:
            supported_dtypes = (ttnn.bfloat16, ttnn.bfloat8_b)
        ttnn.validate_input_tensor(
            operation_name,
            input_tensor_a,
            ranks=(2, 3, 4),
            dtypes=supported_dtypes,
            layouts=(ttnn.TILE_LAYOUT,),
            can_be_on_device=True,
            can_be_on_cpu=False,
        )
        ttnn.validate_input_tensor(
            operation_name,
            input_tensor_b,
            ranks=(2, 3, 4),
            dtypes=supported_dtypes,
            layouts=(ttnn.TILE_LAYOUT,),
            can_be_on_device=True,
            can_be_on_cpu=False,
        )

    @ttnn.register_operation(
        name=f"ttnn.{name}",
        validate_input_tensors=_elt_binary_validate_input_tensors,
        golden_function=_golden_function,
    )
    def elt_binary_function(
        input_tensor_a: ttnn.Tensor,
        input_tensor_b: Union[ttnn.Tensor, int, float],
        *,
        memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    ) -> ttnn.Tensor:
        if not (input_tensor_a.shape == input_tensor_b.shape):
            raise RuntimeError("input_tensors must be of same size!")

        if not isinstance(input_tensor_a, ttnn.Tensor) or not isinstance(input_tensor_b, ttnn.Tensor):
            raise TypeError("Expected both arguments to be a ttnn.Tensor")

        if not ttnn.is_tensor_storage_on_device(input_tensor_a) or not ttnn.is_tensor_storage_on_device(input_tensor_b):
            raise RuntimeError("input_tensors must be on device!")

        original_shape = input_tensor_a.shape

        input_tensor_a = ttnn.unsqueeze_to_4D(input_tensor_a)
        input_tensor_b = ttnn.unsqueeze_to_4D(input_tensor_b)

        output_tensor = ttl_elt_binary_function(input_tensor_a, input_tensor_b, output_mem_config=memory_config)

        output_tensor = ttnn.reshape(output_tensor, original_shape)
        return output_tensor

    if isinstance(elt_binary_function, ttnn.decorators.Operation):
        elt_binary_function.__name__ = f"ttnn.{name}"
        elt_binary_function.decorated_function.__doc__ = f"""{name}(input_tensor_a: ttnn.Tensor, input_tensor_b: ttnn.Tensor, *, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor

            Performs eltwise-binary {op_name} operation on two tensors :attr:`input_a` and :attr:`input_b`.

            .. math::
                {name.replace('_',' ')}(\\mathrm{{input\\_tensor\\_a}}_i \\; , \\; \\mathrm{{input\\_tensor\\_b}}_i )

            Args:
                * :attr:`input_tensor_a`
                * :attr:`input_tensor_b`

            Example::
                >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.tensor(([[1, 2], [3, 4]]), dtype=torch.bfloat16)), device)
                >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.tensor(([[1, 1], [4, 4]]), dtype=torch.bfloat16)), device)
                >>> output = ttnn.{name}(tensor1, tensor2)

            {elt_binary_function.__doc__}

            """

    setattr(THIS_MODULE, name, elt_binary_function)


TTL_BINARY_ELTWISE_FUNCTIONS = [
    ("ldexp", ttl.tensor.ldexp, "ldexp (input_a * 2**input_b)"),
    ("logaddexp", ttl.tensor.logaddexp, "logaddexp (log(exp(input_a) + exp(input_b)))"),
    ("logaddexp2", ttl.tensor.logaddexp2, "logaddexp2 (log2(2^(input_a) + 2^(input_b)))"),
    ("logical_and", ttl.tensor.logical_and, "logical AND (input_a && input_b) "),
    ("logical_or", ttl.tensor.logical_or, "logical OR (input_a || input_b)"),
    ("logical_xor", ttl.tensor.logical_xor, "logical XOR (input_a ^ input_b) "),
    ("xlogy", ttl.tensor.xlogy, "xlogy (input_a * log( input_b ))"),
    ("maximum", ttl.tensor.max, "maximum "),
    ("minimum", ttl.tensor.min, "minimum "),
]


for elt_binary_function_name, ttl_elt_binary_function, op_name in TTL_BINARY_ELTWISE_FUNCTIONS:
    register_ttl_elt_binary_function(elt_binary_function_name, ttl_elt_binary_function, op_name)


def _nextafter_validate_input_tensors(operation_name, input_tensor_a, input_tensor_b, *args, **kwargs):
    ttnn.validate_input_tensor(
        operation_name,
        input_tensor_a,
        ranks=(4,),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
        layouts=(ttnn.TILE_LAYOUT,),
        can_be_on_device=True,
        can_be_on_cpu=False,
    )
    ttnn.validate_input_tensor(
        operation_name,
        input_tensor_b,
        ranks=(4,),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
        layouts=(ttnn.TILE_LAYOUT,),
        can_be_on_device=True,
        can_be_on_cpu=False,
        can_be_a_scalar=False,
    )


def _golden_function(input_tensor_a: ttnn.Tensor, input_tensor_b: ttnn.Tensor, **_):
    import torch

    return torch.nextafter(input_tensor_a, input_tensor_b)


@ttnn.register_operation(
    name="ttnn.nextafter",
    validate_input_tensors=_nextafter_validate_input_tensors,
    golden_function=_golden_function,
)
def nextafter(
    input_tensor_a: ttnn.Tensor,
    input_tensor_b: ttnn.Tensor,
    *,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    dtype: Optional[ttnn.DataType] = None,
) -> ttnn.Tensor:
    r"""
    nextafter(input_tensor_a: ttnn.Tensor, input_tensor_b: ttnn.Tensor, *, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG, dtype: Optional[ttnn.DataType] = None) -> ttnn.Tensor

    Returns the next floating-point value after input_a towards input_b of the input tensors input_a and input_b.

    .. math::
        \mathrm{{input\_tensor\_a}}_i , \mathrm{{input\_tensor\_b}}_i

    Args:
        * :attr:`input_tensor_a`
        * :attr:`input_tensor_b`

    Keyword args:
        :attr:`memory_config` (ttnn.MemoryConfig): memory config for the output tensor
        :attr:`dtype` (Optional[ttnn.DataType]): data type for the output tensor


    """

    output = ttnn.experimental.tensor.nextafter(
        input_tensor_a,
        input_tensor_b,
        output_mem_config=memory_config,
    )
    return output


def _polyval_validate_input_tensors(operation_name, input_tensor, *args, **kwargs):
    ttnn.validate_input_tensor(
        operation_name,
        input_tensor,
        ranks=(4,),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
        layouts=(ttnn.TILE_LAYOUT,),
        can_be_on_device=True,
        can_be_on_cpu=False,
    )


def torch_polyval(input_tensor, coeff):
    curVal = 0
    for curValIndex in range(len(coeff) - 1):
        curVal = (curVal + coeff[curValIndex]) * input_tensor[0]
    return curVal + coeff[len(coeff) - 1]


def _golden_function(input_tensor: ttnn.Tensor, coeff: List[float], **_):
    return torch_polyval(input_tensor, coeff)


@ttnn.register_operation(
    name="ttnn.polyval",
    validate_input_tensors=_polyval_validate_input_tensors,
    golden_function=_golden_function,
)
def polyval(
    input_tensor: ttnn.Tensor,
    coeff: List[float],
    *,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    dtype: Optional[ttnn.DataType] = None,
) -> ttnn.Tensor:
    r"""
    polyval(input_tensor_a: ttnn.Tensor, coeff: List[float], *, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG, dtype: Optional[ttnn.DataType] = None) -> ttnn.Tensor

    Returns tensor with the polyval of all of elements of the input tensor input with coefficients coeffs.

    .. math::
        \mathrm{{input\_tensor\_a}}_i , \mathrm{{coeff}}_i

    Args:
        * :attr:`input_tensor_a`
        * :attr:`coeff`

    Keyword args:
        :attr:`memory_config`
        :attr:`dtype`


    """

    output = ttnn.experimental.tensor.polyval(
        input_tensor,
        coeff,
        output_mem_config=memory_config,
    )
    return output


__all__ = []
