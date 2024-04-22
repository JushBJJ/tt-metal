# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Union, Dict, Optional
import warnings
import math
import ttnn
from tt_eager.tt_dnn.op_library.sliding_window_op_infra.sliding_window_op_utils import calculate_shard_grid, roundup
from tt_eager.tt_dnn.op_library.sliding_window_op_infra.tt_py_composite_conv import (
    TTPyCompositeConv,
    SlidingWindowOpParams,
    _nearest_32,
    find_closest_common_largest_divisor,
    find_closest_largest_divisor,
    find_closest_largest_divisor_with_num_padding,
)
from models.utility_functions import (
    is_grayskull,
    is_wormhole_b0,
)
import ttnn.experimental


class Conv2d:
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        padding_mode: str = "zeros",
        dtype: ttnn.DataType = None,
        *,
        device: ttnn.Device,
        use_1d_systolic_array: bool,
        batch_size: int,
        input_height: int,
        input_width: int,
        reader_patterns_cache: Optional[Dict],
        weight: ttnn.Tensor,
        bias: ttnn.Tensor = None,
        math_fidelity: ttnn.MathFidelity = None,
        weights_dtype: ttnn.DataType = None,
        activation: str = None,
        conv_blocking_and_parallelization_config_override: Dict = None,
        reallocate_halo_output: bool = False,
        using_parameters_cache: bool = False,
        move_weights_to_device: bool = True,
        use_shallow_conv_variant: bool = False,
        enable_auto_formatting: bool = False,
        deallocate_activation: bool = False,
        padded_input_channels: Optional[int] = None,
        compute_kernel_config: Union[ttnn.GrayskullComputeKernelConfig, ttnn.WormholeComputeKernelConfig] = None,
        use_dram_for_matmul: bool = False,
        output_layout: ttnn.Layout = ttnn.TILE_LAYOUT,
    ):
        assert (
            padding_mode == "zeros"
        ), f"Only convs with padding_mode=zeroes supported. Found padding_mode set to {padding_mode}."
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

        assert dilation_h == 1, f"Only convs with dilation == 1 supported. Found dilation_h={dilation_h}"
        assert dilation_w == 1, f"Only convs with dilation == 1 supported. Found dilation_w={dilation_w}"
        assert groups == 1, "Only convs with groups == 1 supported"
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
        fuse_relu = False
        if activation is not None:
            activation = activation.lower()
            assert activation == "relu", f"Only support relu fusion with conv. Got activation={activation}."
            fuse_relu = True
        self.conv = TTPyCompositeConv(
            sliding_window_op_params,
            weight,
            out_channels,
            in_channels,
            device,
            use_1d_systolic_array,
            reader_patterns_cache,
            bias=bias,
            conv_blocking_and_parallelization_config_override=conv_blocking_and_parallelization_config_override,
            fuse_relu=fuse_relu,
            output_dtype=dtype,
            weights_dtype=weights_dtype,
            math_fidelity=math_fidelity,
            move_utwh_output=reallocate_halo_output,
            using_parameters_cache=using_parameters_cache,
            move_weights_to_device=move_weights_to_device,
            use_shallow_conv_variant=use_shallow_conv_variant,
            enable_auto_formatting=enable_auto_formatting,
            deallocate_activation=deallocate_activation,
            padded_input_channels=padded_input_channels,
            compute_kernel_config=compute_kernel_config,
            use_dram_for_matmul=use_dram_for_matmul,
            output_layout=output_layout,
        )
        self.batch_size = batch_size
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = (input_height + (2 * pad_h) - dilation_h * (window_h - 1) - 1) // stride_h + 1
        self.output_width = (input_width + (2 * pad_w) - dilation_w * (window_w - 1) - 1) // stride_w + 1
        self.in_channels = in_channels
        self.out_channels = out_channels

    @ttnn.register_operation(
        name="ttnn.Conv2d.__call__", validate_input_tensors=lambda *args, **kwargs: None, is_method=True
    )
    def __call__(self, activation: ttnn.Tensor):
        return self.conv(activation)

    @ttnn.register_operation(
        name="ttnn.Conv2d.copy_input_to_device", validate_input_tensors=lambda *args, **kwargs: None, is_method=True
    )
    def copy_input_to_device(self, input: ttnn.Tensor):
        return self.conv.copy_input_to_device(input)

    @ttnn.register_operation(
        name="ttnn.Conv2d.copy_output_from_device", validate_input_tensors=lambda *args, **kwargs: None, is_method=True
    )
    def copy_output_from_device(self, output: ttnn.Tensor):
        return self.conv.copy_output_from_device(output)

    def get_parallel_config(self):
        return self.conv.get_parallel_config()


# user facing
class ConvConfig:
    def __init__(
        self,
        *,
        # default config values if user does not set them
        math_fidelity=ttnn.MathFidelity.HiFi4,
        dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
        activation=None,
        # following config values are set by conv op later if user does not set them
        act_block_h=None,
        height_sharding=None,
        core_grid=None,
    ):
        self.math_fidelity = math_fidelity
        self.dtype = dtype
        self.weights_dtype = weights_dtype
        self.math_approx_mode = math_approx_mode
        self.fp32_dest_acc_en = fp32_dest_acc_en
        self.packer_l1_acc = packer_l1_acc
        self.activation = activation
        self.act_block_h = act_block_h
        self.height_sharding = height_sharding
        self.core_grid = core_grid


# internal. not user facing
class ParallelConfig:
    def __init__(
        self,
        num_cores_y: int,
        num_cores_x: int,
        num_cores_nhw: int,
        shard_scheme: ttnn.TensorMemoryLayout,
        shard_orientation: ttnn.ShardOrientation,
    ):
        # TODO: using core range set would be better
        self.grid_size = ttnn.experimental.tensor.CoreCoord(num_cores_x, num_cores_y)
        self.num_cores_nhw = num_cores_nhw
        self.shard_scheme = shard_scheme
        self.shard_orientation = shard_orientation


# internal helper function. not exposed to user.
def get_shard_grid_from_core_grid(core_grid):
    shard_grid = None
    if isinstance(core_grid, ttnn.CoreGrid):
        grid_coord = ttnn.experimental.tensor.CoreCoord(core_grid.x - 1, core_grid.y - 1)
        shard_grid = ttnn.experimental.tensor.CoreRangeSet(
            {ttnn.experimental.tensor.CoreRange(ttnn.experimental.tensor.CoreCoord(0, 0), grid_coord)}
        )
    elif isinstance(core_grid, (list, tuple)):
        if len(core_grid) != 2:
            raise RuntimeError("Invalid core_grid")
        if not isinstance(core_grid[0], ttnn.CoreGrid):
            raise RuntimeError("Invalid core_grid type")
        if not isinstance(core_grid[1], ttnn.CoreGrid):
            raise RuntimeError("Invalid core_grid type")

        grid_coord_1 = ttnn.experimental.tensor.CoreCoord(core_grid[0].x - 1, core_grid[0].y - 1)
        grid_coord_2 = ttnn.experimental.tensor.CoreCoord(core_grid[1].x - 1, core_grid[0].y)
        shard_grid = ttnn.experimental.tensor.CoreRangeSet(
            {
                ttnn.experimental.tensor.CoreRange(ttnn.experimental.tensor.CoreCoord(0, 0), grid_coord_1),
                ttnn.experimental.tensor.CoreRange(ttnn.experimental.tensor.CoreCoord(0, core_grid[0].y), grid_coord_2),
            }
        )
    elif isinstance(core_grid, ttnn.experimental.tensor.CoreRangeSet):
        shard_grid = core_grid
    else:
        raise RuntimeError("Invalid core_grid type")
    return shard_grid


# internal helper function. not exposed to user.
def determine_parallel_config(
    is_1d_systolic,
    batch_size,
    input_channels,
    output_height,
    output_width,
    output_channels,
    device,
    config_override=None,
    is_out_tiled=True,
):
    if config_override is None:
        config_override = {}
    for k in config_override.keys():
        assert k == "grid_size" or k == "num_cores_nhw"

    conv_out_2d_matrix_height = batch_size * output_height * output_width
    # pad height to 32
    conv_out_2d_matrix_height = _nearest_32(conv_out_2d_matrix_height)
    if is_out_tiled:
        conv_out_2d_matrix_height_ntiles = (int)(conv_out_2d_matrix_height / 32)
        conv_out_2d_matrix_width_ntiles = (int)(_nearest_32(output_channels) / 32)
    else:
        conv_out_2d_matrix_height_ntiles = conv_out_2d_matrix_height
        conv_out_2d_matrix_width_ntiles = output_channels

    compute_with_storage_grid_size = device.compute_with_storage_grid_size()
    device_grid_size = (compute_with_storage_grid_size.x, compute_with_storage_grid_size.y)
    max_num_cores = device_grid_size[0] * device_grid_size[1]

    def calculate_num_cores_nhw(override):
        num_cores_nhw = (
            find_closest_largest_divisor(conv_out_2d_matrix_height_ntiles, max_num_cores)
            if is_1d_systolic
            else find_closest_largest_divisor_with_num_padding(conv_out_2d_matrix_height_ntiles, device_grid_size[0])
        )
        if override is not None and num_cores_nhw != override:
            warnings.warn(f"Overriding config: num_cores_nhw from {num_cores_nhw} to user provided config={override}")
            num_cores_nhw = override
        return num_cores_nhw

    def calculate_grid_size(num_cores_nhw, override):
        if is_1d_systolic:
            grid_size = [
                device_grid_size[0] if num_cores_nhw >= device_grid_size[0] else num_cores_nhw,
                math.ceil(num_cores_nhw / device_grid_size[0]),
            ]  # for 1d systolic array, grid size is the tightest bound of num_cores_nhw as a rectangle (x,y)
            assert (
                num_cores_nhw <= grid_size[0] * grid_size[1]
            ), "Error: For 1d systolic conv, num_cores_nhw must be <= grid size"
        else:
            grid_size = [
                num_cores_nhw,
                find_closest_common_largest_divisor(
                    conv_out_2d_matrix_width_ntiles, _nearest_32(input_channels) // 32, device_grid_size[1]
                ),
            ]
            assert (
                num_cores_nhw == grid_size[0]
            ), "Error: For 2d systolic conv, num_cores_nhw must be == # of cols in grid size"

        if override is not None and grid_size != override:
            warnings.warn(f"Overriding config: grid_size from {grid_size} to user provided config={override}")
            grid_size = override
        return grid_size

    num_cores_nhw = calculate_num_cores_nhw(config_override.get("num_cores_nhw", None))
    grid_size = calculate_grid_size(num_cores_nhw, config_override.get("grid_size", None))
    shard_scheme = ttnn.TensorMemoryLayout.HEIGHT_SHARDED if is_1d_systolic else ttnn.TensorMemoryLayout.BLOCK_SHARDED
    shard_orientation = ttnn.ShardOrientation.ROW_MAJOR if is_1d_systolic else ttnn.ShardOrientation.COL_MAJOR
    return ParallelConfig(grid_size[1], grid_size[0], num_cores_nhw, shard_scheme, shard_orientation)


# internal helper function. not exposed to user.
def get_grid_size_and_num_cores_nhw_from_core_grid(core_grid, height_sharded):
    if isinstance(core_grid, ttnn.CoreGrid):
        if height_sharded:
            num_cores_nhw = core_grid.x * core_grid.y
        else:
            num_cores_nhw = core_grid.x
        grid_size = core_grid
    elif isinstance(core_grid, (list, tuple)):
        if len(core_grid) != 2:
            raise RuntimeError("Invalid core_grid")
        if not isinstance(core_grid[0], ttnn.CoreGrid):
            raise RuntimeError("Invalid core_grid type")
        if not isinstance(core_grid[1], ttnn.CoreGrid):
            raise RuntimeError("Invalid core_grid type")
        assert height_sharded
        num_cores_nhw = (core_grid[0].x * core_grid[0].y) + core_grid[1].x
    elif isinstance(core_grid, ttnn.experimental.tensor.CoreRangeSet):
        grid_size = core_grid.bounding_box().grid_size()
        num_cores = core_grid.num_cores()
        if height_sharded:
            num_cores_nhw = num_cores
        else:
            num_cores_nhw = grid_size.x
    else:
        raise RuntimeError("Invalid core_grid type")
    return grid_size, num_cores_nhw


# internal helper function. not exposed to user.
def create_sharded_memory_config_from_parallel_config(tensor_shape, parallel_config, tile_size):
    # tensor_shape is [N, H, W, C]
    assert len(tensor_shape) == 4
    assert tensor_shape[0] == 1 and tensor_shape[1] == 1  # todo: add support for generic non-2d shapes
    channels = tensor_shape[3]
    num_cores_nhw = parallel_config.num_cores_nhw
    num_cores_x = parallel_config.grid_size.x
    num_cores_y = parallel_config.grid_size.y
    shard_scheme = parallel_config.shard_scheme
    shard_orientation = parallel_config.shard_orientation
    is_1d_systolic = shard_scheme == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    if is_1d_systolic:
        logical_grid_size = (num_cores_nhw, 1)
    else:
        logical_grid_size = (num_cores_x, num_cores_y)

    shard_grid, shard_layout = calculate_shard_grid((num_cores_x, num_cores_y), num_cores_nhw)
    assert shard_layout == shard_scheme
    nhw_shape = tensor_shape[0] * tensor_shape[1] * tensor_shape[2]
    nhw_padded = roundup(nhw_shape, num_cores_nhw * tile_size)
    nhw_shard = nhw_padded // num_cores_nhw
    assert channels % logical_grid_size[1] == 0
    shard_shape = [nhw_shard, channels // logical_grid_size[1]]
    shard_halo = False
    shard_spec = ttnn.experimental.tensor.ShardSpec(shard_grid, shard_shape, shard_orientation, shard_halo)
    return ttnn.MemoryConfig(shard_scheme, ttnn.BufferType.L1, shard_spec)


# 4d -> nhwc
@ttnn.register_operation(name="ttnn.conv2d", is_cpp_function=True)
def conv2d(
    *,
    input_tensor: ttnn.Tensor,  # may or may not be sharded
    weight_tensor: ttnn.Tensor,
    device: ttnn.Device,
    in_channels: int,
    out_channels: int,
    batch_size: int,
    input_height: int,
    input_width: int,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1,
    bias_tensor: ttnn.Tensor = None,
    conv_config: ConvConfig = None,  # manual override by user
    reshard_if_not_optimal=False,  # default
) -> ttnn.Tensor:
    output_height = ((int)((input_height - kernel_size[0] + 2 * padding[0]) / stride[0])) + 1
    output_width = ((int)((input_width - kernel_size[1] + 2 * padding[1]) / stride[1])) + 1
    if conv_config is None:
        conv_config = ConvConfig()
    config_shard_grid = None
    if conv_config.core_grid is not None:
        config_shard_grid = get_shard_grid_from_core_grid(conv_config.core_grid)

    needs_reshard = False
    input_memory_config = ttnn.get_memory_config(input_tensor)
    if ttnn.is_sharded(input_tensor):
        input_shard_scheme = input_memory_config.memory_layout
        input_shard_orientation = input_memory_config.shard_spec.orientation
        input_shard_grid = input_memory_config.shard_spec.grid
        if not (
            input_shard_scheme == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
            or input_shard_scheme == ttnn.TensorMemoryLayout.BLOCK_SHARDED
        ):
            needs_reshard = True
        if (
            input_shard_scheme == ttnn.TensorMemoryLayout.BLOCK_SHARDED
            and input_shard_orientation != ttnn.ShardOrientation.COL_MAJOR
        ):
            needs_reshard = True
        if (
            input_shard_scheme == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
            and input_shard_orientation != ttnn.ShardOrientation.ROW_MAJOR
        ):
            needs_reshard = True
        if config_shard_grid is not None:
            if config_shard_grid != input_shard_grid:
                needs_reshard = True
        input_height_sharded = input_shard_scheme == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        if conv_config.height_sharding is not None:
            if input_height_sharded != conv_config.height_sharding:
                needs_reshard = True
    else:
        needs_reshard = True
    parallel_config = None
    optimal_parallel_config = determine_parallel_config(
        True if conv_config.height_sharding is None else conv_config.height_sharding,
        batch_size,
        in_channels,
        output_height,
        output_width,
        out_channels,
        device,
    )
    if needs_reshard:
        if conv_config.height_sharding is None:
            # default shard scheme is height sharding
            conv_config.height_sharding = True
        if conv_config.core_grid is None:
            parallel_config = optimal_parallel_config
        else:
            assert config_shard_grid is not None
            grid_size, num_cores_nhw = get_grid_size_and_num_cores_nhw_from_core_grid(
                conv_config.core_grid, conv_config.height_sharding
            )
            shard_scheme = (
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED
                if conv_config.height_sharding
                else ttnn.TensorMemoryLayout.BLOCK_SHARDED
            )
            shard_orientation = (
                ttnn.ShardOrientation.ROW_MAJOR if conv_config.height_sharding else ttnn.ShardOrientation.COL_MAJOR
            )
            parallel_config = ParallelConfig(grid_size.y, grid_size.x, num_cores_nhw, shard_scheme, shard_orientation)
    else:
        assert ttnn.is_sharded(input_tensor)
        grid_size, num_cores_nhw = get_grid_size_and_num_cores_nhw_from_core_grid(
            input_memory_config.shard_spec.grid,
            input_memory_config.memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        )
        parallel_config = ParallelConfig(
            grid_size.y,
            grid_size.x,
            num_cores_nhw,
            input_memory_config.memory_layout,
            input_memory_config.shard_spec.orientation,
        )

    if reshard_if_not_optimal:
        if parallel_config != optimal_parallel_config:
            parallel_config = optimal_parallel_config
            needs_reshard = True
    if needs_reshard:
        # not sure if reshard op works for all cases
        # copying to l1 interleaved first
        input_tensor = ttnn.to_memory_config(input_tensor, ttnn.L1_MEMORY_CONFIG)
        if input_tensor.shape[0] != 1 or input_tensor.shape[1] != 1:
            # reshape to [1, 1, N*H*W, C]
            input_tensor = ttnn.reshape(input_tensor, (1, 1, -1, input_tensor.shape[-1]))
        input_num_cores_nhw = parallel_config.num_cores_nhw
        input_tensor_sharded_memory_config = create_sharded_memory_config_from_parallel_config(
            input_tensor.shape, parallel_config, tile_size=32
        )
        input_tensor_height_snapped_to_tile = (
            input_tensor_sharded_memory_config.shard_spec.shape[0] * input_num_cores_nhw
        )
        assert input_tensor_height_snapped_to_tile >= input_tensor.shape[2]
        if input_tensor_height_snapped_to_tile != input_tensor.shape[2]:
            input_tensor = ttnn.pad(
                input_tensor,
                padding=((0, 0), (0, 0), (0, input_tensor_height_snapped_to_tile - input_tensor.shape[2]), (0, 0)),
                value=0,
            )
        input_tensor = ttnn.to_device(input_tensor, device=device, memory_config=input_tensor_sharded_memory_config)

    is_1x1_conv = kernel_size == (1, 1) and stride == (1, 1) and padding == (0, 0)
    if is_1x1_conv and input_tensor.layout != ttnn.TILE_LAYOUT:
        input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT)

    # Following code will be removed after op refactoring
    block_and_parallel_config_override = {}
    if conv_config.act_block_h is not None:
        block_and_parallel_config_override["act_block_h"] = conv_config.act_block_h
    assert parallel_config is not None
    block_and_parallel_config_override["grid_size"] = [parallel_config.grid_size.x, parallel_config.grid_size.y]
    block_and_parallel_config_override["num_cores_nhw"] = parallel_config.num_cores_nhw
    if is_grayskull():
        compute_kernel_config = ttnn.GrayskullComputeKernelConfig(
            math_fidelity=conv_config.math_fidelity,
            math_approx_mode=conv_config.math_approx_mode,
        )
    else:
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=conv_config.math_fidelity,
            math_approx_mode=conv_config.math_approx_mode,
            fp32_dest_acc_en=conv_config.fp32_dest_acc_en,
            packer_l1_acc=conv_config.packer_l1_acc,
        )
    # Build conv op object
    conv = ttnn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        dtype=conv_config.dtype,
        device=device,
        use_1d_systolic_array=parallel_config.shard_scheme == ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        reader_patterns_cache={},
        weight=weight_tensor,
        bias=bias_tensor,
        math_fidelity=conv_config.math_fidelity,
        weights_dtype=conv_config.weights_dtype,
        conv_blocking_and_parallelization_config_override=block_and_parallel_config_override,
        compute_kernel_config=compute_kernel_config,
        activation=conv_config.activation,
    )
    # Run conv
    return conv(input_tensor)


__all__ = []
