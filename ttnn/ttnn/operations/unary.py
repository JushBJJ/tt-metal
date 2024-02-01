# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import sys

import tt_lib as ttl

import ttnn


THIS_MODULE = sys.modules[__name__]

__all__ = []


def register_ttl_unary_function(name, ttl_unary_function):
    def _torch_unary(input_tensor: ttnn.Tensor, **_):
        import torch

        name_to_torch_function = {
            "exp": torch.exp,
            "tanh": torch.tanh,
            "gelu": torch.nn.functional.gelu,
            "rsqrt": torch.rsqrt,
            "relu": torch.relu,
            "silu": torch.nn.functional.silu,
            "log": torch.log,
            "sin": torch.sin,
            "cos": torch.cos,
            "tan": torch.tan,
            "asin": torch.asin,
            "acos": torch.acos,
            "atan": torch.atan,
        }
        torch_function = name_to_torch_function[name]
        input_tensor = ttnn.to_torch(input_tensor)
        return torch_function(input_tensor)

    def _unary_validate_input_tensors(operation_name, input_tensor, *args, **kwargs):
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
        validate_input_tensors=_unary_validate_input_tensors,
        torch_function=_torch_unary,
    )
    def unary_function(
        input_tensor: ttnn.Tensor, *, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG
    ) -> ttnn.Tensor:
        original_shape = input_tensor.shape
        input_tensor = ttnn.unsqueeze_to_4D(input_tensor)
        ttl_input_tensor = input_tensor.value

        if not isinstance(input_tensor, ttnn.Tensor):
            raise TypeError("Expected first argument to be a ttnn.Tensor")

        if not ttnn.has_storage_type_of(input_tensor, ttnn.DEVICE_STORAGE_TYPE):
            raise RuntimeError("input_tensor must be on device!")
        ttl_input_tensor = input_tensor.value

        ttl_output_tensor = ttl_unary_function(ttl_input_tensor, output_mem_config=memory_config)

        output_tensor = ttnn.Tensor(ttl_output_tensor)
        output_tensor = ttnn.reshape(output_tensor, original_shape)
        return output_tensor

    unary_function.__name__ = f"ttnn.{name}"
    unary_function.__doc__ = f"""{name}(input_tensor: ttnn.Tensor, *, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor

        Applies {name} to :attr:`input_tensor` element-wise.

        .. math::
            {name}(\\mathrm{{input\\_tensor}}_i)

        Args:
            * :attr:`input_tensor`

        Example::

            >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> output = ttnn.{name}(tensor)

        {unary_function.__doc__}

        """
    setattr(THIS_MODULE, name, unary_function)


TTL_UNARY_FUNCTIONS = [
    ("exp", ttl.tensor.exp),
    ("tanh", ttl.tensor.tanh),
    ("gelu", ttl.tensor.gelu),
    ("relu", ttl.tensor.relu),
    ("rsqrt", ttl.tensor.rsqrt),
    ("silu", ttl.tensor.silu),
    ("log", ttl.tensor.log),
    ("sin", ttl.tensor.sin),
    ("cos", ttl.tensor.cos),
    ("tan", ttl.tensor.tan),
    ("asin", ttl.tensor.asin),
    ("acos", ttl.tensor.acos),
    ("atan", ttl.tensor.atan),
]


for unary_function_name, ttl_unary_function in TTL_UNARY_FUNCTIONS:
    register_ttl_unary_function(unary_function_name, ttl_unary_function)

__all__ = []
