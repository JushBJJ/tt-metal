# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


from typing import Optional, Tuple

import ttnn

import tt_lib as ttl


def _torch_layer_norm(
    input_tensor: ttnn.Tensor, *, epsilon=1e-12, residual_input_tensor=None, weight=None, bias=None, **_
):
    import torch

    input_tensor = ttnn.from_device(input_tensor)
    input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
    input_tensor = ttnn.to_torch(input_tensor)

    if residual_input_tensor is not None:
        residual_input_tensor = ttnn.from_device(residual_input_tensor)
        residual_input_tensor = ttnn.to_layout(residual_input_tensor, ttnn.ROW_MAJOR_LAYOUT)
        residual_input_tensor = ttnn.to_torch(residual_input_tensor)
        input_tensor += residual_input_tensor

    if weight is not None:
        weight = ttnn.from_device(weight)
        weight = ttnn.to_layout(weight, ttnn.ROW_MAJOR_LAYOUT)
        weight = ttnn.to_torch(weight)
        if len(weight.shape) == 2:
            weight = weight[0]

    if bias is not None:
        bias = ttnn.from_device(bias)
        bias = ttnn.to_layout(bias, ttnn.ROW_MAJOR_LAYOUT)
        bias = ttnn.to_torch(bias)
        if len(bias.shape) == 2:
            bias = bias[0]

    return torch.nn.functional.layer_norm(input_tensor, (input_tensor.shape[-1],), weight, bias, eps=epsilon)


def _layer_norm_validate_input_tensors(
    operation_name, input_tensor, *args, weight=None, bias=None, residual_input_tensor=None, **kwargs
):
    ttnn.validate_input_tensor(
        operation_name,
        input_tensor,
        ranks=(2, 3, 4),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
        layouts=(ttnn.TILE_LAYOUT,),
        can_be_on_device=True,
        can_be_on_cpu=False,
    )
    ttnn.validate_input_tensor(
        operation_name,
        weight,
        ranks=(1, 2, 3, 4),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
        layouts=(ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
        can_be_on_device=True,
        can_be_on_cpu=False,
        is_optional=True,
    )
    ttnn.validate_input_tensor(
        operation_name,
        bias,
        ranks=(1, 2, 3, 4),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
        layouts=(ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
        can_be_on_device=True,
        can_be_on_cpu=False,
        is_optional=True,
    )
    ttnn.validate_input_tensor(
        operation_name,
        residual_input_tensor,
        ranks=(2, 3, 4),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
        layouts=(ttnn.TILE_LAYOUT,),
        can_be_on_device=True,
        can_be_on_cpu=False,
        is_optional=True,
    )


@ttnn.register_operation(
    name="ttnn.layer_norm",
    validate_input_tensors=_layer_norm_validate_input_tensors,
    torch_function=_torch_layer_norm,
)
def layer_norm(
    input_tensor: ttnn.Tensor,
    *,
    epsilon: float = 1e-12,
    weight: Optional[ttnn.Tensor] = None,
    bias: Optional[ttnn.Tensor] = None,
    residual_input_tensor: Optional[ttnn.Tensor] = None,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    r"""
    layer_norm(input_tensor: ttnn.Tensor, *, epsilon: float = 1e-12, weight: Optional[ttnn.Tensor] = None, bias: Optional[ttnn.Tensor] = None, residual_input_tensor: Optional[ttnn.Tensor] = None, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor

    Compute layer_norm over :attr:`input_tensor`.

    """

    original_shape = input_tensor.shape
    input_tensor = ttnn.unsqueeze_to_4D(input_tensor)
    if residual_input_tensor is not None:
        residual_input_tensor = ttnn.unsqueeze_to_4D(residual_input_tensor)
    if weight is not None:
        weight = ttnn.unsqueeze_to_4D(weight)
    if bias is not None:
        bias = ttnn.unsqueeze_to_4D(bias)

    ttl_input_tensor = input_tensor.value
    residual_input_tensor = residual_input_tensor.value if residual_input_tensor is not None else None
    ttl_weight = weight.value if weight is not None else None
    ttl_bias = bias.value if bias is not None else None

    if residual_input_tensor is not None:
        output_tensor = ttl.tensor.add_layernorm(
            ttl_input_tensor, residual_input_tensor, epsilon, ttl_weight, ttl_bias, output_mem_config=memory_config
        )
    else:
        output_tensor = ttl.tensor.layernorm(
            ttl_input_tensor, epsilon, ttl_weight, ttl_bias, output_mem_config=memory_config
        )

    output_tensor = ttnn.Tensor(output_tensor)
    output_tensor = ttnn.reshape(output_tensor, original_shape)
    return output_tensor


def _rms_norm_validate_input_tensors(operation_name, input_tensor, weight, *args, **kwargs):
    ttnn.validate_input_tensor(
        operation_name,
        input_tensor,
        ranks=(2, 3, 4),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
        layouts=(ttnn.TILE_LAYOUT,),
        can_be_on_device=True,
        can_be_on_cpu=False,
    )
    ttnn.validate_input_tensor(
        operation_name,
        weight,
        ranks=(1, 2, 3, 4),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
        layouts=(ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
        can_be_on_device=True,
        can_be_on_cpu=False,
    )


@ttnn.register_operation(
    name="ttnn.rms_norm",
    validate_input_tensors=_rms_norm_validate_input_tensors,
)
def rms_norm(input_tensor: ttnn.Tensor, weight: ttnn.Tensor, *, epsilon: float = 1e-6) -> ttnn.Tensor:
    r"""
    rms_norm(input_tensor: ttnn.Tensor, weight: ttnn.Tensor, *, epsilon: float = 1e-6) -> ttnn.Tensor

    Compute rms_norm over :attr:`input_tensor`.

    """

    if not ttnn.has_storage_type_of(input_tensor, ttnn.DEVICE_STORAGE_TYPE):
        raise RuntimeError("rms_norm only supports device storage type")

    original_shape = input_tensor.shape
    input_tensor = ttnn.unsqueeze_to_4D(input_tensor)
    weight = ttnn.unsqueeze_to_4D(weight)

    ttl_input_tensor = input_tensor.value
    ttl_weight = weight.value
    ttl_output_tensor = ttl.tensor.rmsnorm(ttl_input_tensor, epsilon, ttl_weight)

    output_tensor = ttnn.Tensor(ttl_output_tensor)
    output_tensor = ttnn.reshape(output_tensor, original_shape)

    return output_tensor


def _torch_group_norm(input_tensor: ttnn.Tensor, *, num_groups, epsilon=1e-05, weight=None, bias=None, **_):
    import torch

    input_tensor = ttnn.from_device(input_tensor)
    input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
    input_tensor = ttnn.to_torch(input_tensor)

    if weight is not None:
        weight = ttnn.from_device(weight)
        weight = ttnn.to_layout(weight, ttnn.ROW_MAJOR_LAYOUT)
        weight = ttnn.to_torch(weight)
        if len(weight.shape) == 2:
            weight = weight[0]

    if bias is not None:
        bias = ttnn.from_device(bias)
        bias = ttnn.to_layout(bias, ttnn.ROW_MAJOR_LAYOUT)
        bias = ttnn.to_torch(bias)
        if len(bias.shape) == 2:
            bias = bias[0]

    return torch.nn.functional.group_norm(input_tensor, num_groups, weight, bias, eps=epsilon)


def _group_norm_validate_input_tensors(operation_name, input_tensor, *args, weight=None, bias=None, **kwargs):
    ttnn.validate_input_tensor(
        operation_name,
        input_tensor,
        ranks=(2, 3, 4),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
        layouts=(ttnn.TILE_LAYOUT,),
        can_be_on_device=True,
        can_be_on_cpu=False,
    )
    ttnn.validate_input_tensor(
        operation_name,
        weight,
        ranks=(1, 2, 3, 4),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
        layouts=(ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
        can_be_on_device=True,
        can_be_on_cpu=False,
        is_optional=True,
    )
    ttnn.validate_input_tensor(
        operation_name,
        bias,
        ranks=(1, 2, 3, 4),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
        layouts=(ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
        can_be_on_device=True,
        can_be_on_cpu=False,
        is_optional=True,
    )


@ttnn.register_operation(
    name="ttnn.group_norm",
    validate_input_tensors=_group_norm_validate_input_tensors,
    torch_function=_torch_group_norm,
)
def group_norm(
    input_tensor: ttnn.Tensor,
    *,
    num_groups: int,
    epsilon: float = 1e-12,
    weight: Optional[ttnn.Tensor] = None,
    bias: Optional[ttnn.Tensor] = None,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    dtype: Optional[ttnn.DataType] = None,
    core_grid: Optional[Tuple[int, int]] = None,
) -> ttnn.Tensor:
    r"""
    group_norm(input_tensor: ttnn.Tensor, *, num_groups: int, epsilon: float = 1e-12, weight: Optional[ttnn.Tensor] = None, bias: Optional[ttnn.Tensor] = None) -> ttnn.Tensor

    Compute group_norm over :attr:`input_tensor`.

    """

    if ttnn.is_sharded(input_tensor):
        if input_tensor.shape.rank != 4:
            raise TypeError("The input tensor rank must equal to 4")

        if input_tensor.shape[-1] % int(num_groups * ttnn.TILE_SIZE) != 0:
            raise TypeError("number of channels must be divisible by number of groups * tile size")

        if ttnn.get_memory_config(input_tensor).memory_layout == ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED:
            raise TypeError("Cannot be width sharded")

        if (input_tensor.shape[0] * input_tensor.shape[1] * input_tensor.shape[2]) % ttnn.TILE_SIZE != 0:
            raise TypeError("input tensor dim NHW must be divisible by tile size")

        output_dtype = input_tensor.dtype if dtype is None else dtype

        if weight is not None:
            weight = ttnn.unsqueeze_to_4D(weight)

        if bias is not None:
            bias = ttnn.unsqueeze_to_4D(bias)

        output_tensor = ttl.operations.primary.groupnorm(
            input_tensor.value,
            num_groups,
            epsilon,
            weight.value,
            bias.value,
            output_mem_config=memory_config,
            program_config=ttl.operations.primary.GroupNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(core_grid[1], core_grid[0]),
                out_data_format=output_dtype,
                inplace=False
                if (input_tensor.layout == ttnn.TILE_LAYOUT and input_tensor.dtype == output_dtype)
                else True,
            ),
        )

        return ttnn.Tensor(output_tensor)

    else:
        output = _torch_group_norm(input_tensor, num_groups=num_groups, epsilon=epsilon, weight=weight, bias=bias)
        return ttnn.from_torch(output, dtype=input_tensor.dtype, layout=input_tensor.layout, device=input_tensor.device)


__all__ = []
