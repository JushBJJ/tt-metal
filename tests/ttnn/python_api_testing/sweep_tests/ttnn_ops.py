# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import tt_lib
from models.helper_funcs import Linear as tt_Linear


def layout_to_ttnn(layout):
    if layout == tt_lib.tensor.Layout.TILE:
        return ttnn.TILE_LAYOUT

    elif layout == tt_lib.tensor.Layout.ROW_MAJOR:
        return ttnn.ROW_MAJOR_LAYOUT

    else:
        assert False, "Unknown layout passed"


def dtype_to_ttnn(dtype):
    if dtype == tt_lib.tensor.DataType.BFLOAT16:
        return ttnn.bfloat16

    elif dtype == tt_lib.tensor.DataType.BFLOAT8_B:
        return ttnn.bfloat8_b

    else:
        assert False, "Unknown dtype passed"


def memory_config_to_ttnn(mem_config):
    if mem_config is None:
        return None

    if mem_config.buffer_type == tt_lib.tensor.BufferType.DRAM:
        return ttnn.DRAM_MEMORY_CONFIG

    elif mem_config.buffer_type == tt_lib.tensor.BufferType.L1:
        return ttnn.L1_MEMORY_CONFIG

    else:
        assert False, "Unknown memory passed"


def setup_ttnn_tensor(x, device, layout, input_mem_config, dtype):
    input_tensor = ttnn.from_torch(
        x,
        dtype=dtype_to_ttnn(dtype),
        layout=layout_to_ttnn(layout),
        device=device if input_mem_config is not None else None,
        memory_config=memory_config_to_ttnn(input_mem_config),
    )

    return input_tensor


def ttnn_tensor_to_torch(x, output_mem_config=None):
    output_tensor = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    # assert output_mem_config == tt_lib.tensor.MemoryConfig(tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM)
    return ttnn.to_torch(output_tensor)


def ones(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = ttnn.ones(
        x.shape,
        device=device if input_mem_config[0] is not None else None,
        dtype=dtype_to_ttnn(dtype[0]),
        layout=layout_to_ttnn(layout[0]),
        memory_config=memory_config_to_ttnn(output_mem_config),
    )

    return ttnn_tensor_to_torch(t0)


def ones_like(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.ones_like(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def full(
    x,
    *args,
    scalar,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = ttnn.full(
        x.shape,
        fill_value=scalar,
        device=device if input_mem_config[0] is not None else None,
        dtype=dtype_to_ttnn(dtype[0]),
        layout=layout_to_ttnn(layout[0]),
        memory_config=memory_config_to_ttnn(output_mem_config),
    )

    return ttnn_tensor_to_torch(t0)


def eltwise_hardswish(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.hardswish(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def eltwise_hardtanh(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.hardtanh(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def eltwise_heaviside(
    x,
    *args,
    scalar,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.heaviside(t0, scalar, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def eltwise_hypot(
    x,
    y,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = ttnn.hypot(t0, t1, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t2)


def eltwise_i0(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.i0(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def eltwise_isfinite(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.isfinite(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def eltwise_isinf(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.isinf(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def eltwise_isnan(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.isnan(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def eltwise_isneginf(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.isneginf(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def eltwise_isposinf(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.isposinf(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def eltwise_leaky_relu(
    x,
    *args,
    negative_slope,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.leaky_relu(t0, negative_slope, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def eltwise_lerp(
    x,
    y,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = ttnn.lerp(t0, t1, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t2)


def eltwise_add(
    x,
    y,
    *args,
    scalar,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.add(t0, t1, alpha=scalar)
    return ttnn_tensor_to_torch(t2, output_mem_config)


def eltwise_exp(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.exp(t0)
    return ttnn_tensor_to_torch(t1, output_mem_config)


def permute(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    permute_dims,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.permute(t0, permute_dims)
    return ttnn_tensor_to_torch(t1, output_mem_config)


def reshape(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    reshape_dims,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.reshape(t0, reshape_dims)
    return ttnn_tensor_to_torch(t1, output_mem_config)


def gelu(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])

    t1 = ttnn.gelu(t0)
    return ttnn_tensor_to_torch(t1, output_mem_config)


def eltwise_sub(
    x,
    y,
    *args,
    scalar,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.sub(t0, t1, alpha=scalar)
    return ttnn_tensor_to_torch(t2, output_mem_config)


def embeddings(x, y, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    x = x.int()
    x = torch.clamp(x, min=0, max=y.shape[0] - 1)

    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[1])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.embedding(t0, t1, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t2, output_mem_config)


def eltwise_tanh(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])

    t1 = ttnn.tanh(t0)
    return ttnn_tensor_to_torch(t1, output_mem_config)


def softmax(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])

    # t2 = ttnn.add(t0, t1, alpha=scalar)
    t1 = ttnn.softmax(t0, dim=-1)
    return ttnn_tensor_to_torch(t1, output_mem_config)


def mul(
    x,
    y,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.mul(t0, t1)
    return ttnn_tensor_to_torch(t2, output_mem_config)


def linear(x, weight, bias=None, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    # tensor preprocessing
    if bias is not None:
        bias = bias.repeat(1, 1, 32, 1)

    weight = torch.transpose(weight, 2, 3)
    batch_size = x.shape[0]
    num_cores_x = 12

    # ttnn setup
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    tt_weight = setup_ttnn_tensor(weight, device, layout[1], input_mem_config[1], dtype[1])

    if bias is not None:
        tt_bias = setup_ttnn_tensor(bias, device, layout[2], input_mem_config[2], dtype[2])
    else:
        tt_bias = None

    t1 = ttnn.linear(
        t0, tt_weight, bias=tt_bias, dtype=ttnn.bfloat16, core_grid=ttnn.CoreGrid(y=batch_size, x=num_cores_x)
    )
    return ttnn_tensor_to_torch(t1, output_mem_config)


def eltwise_softmax_in_place(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])

    t1 = ttnn.softmax(t0, -1)
    return ttnn_tensor_to_torch(t1, output_mem_config)


def matmul(
    x,
    y,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = t0 @ t1  # ttnn.matmul(t0, t1)
    return ttnn_tensor_to_torch(t2, output_mem_config)


def layernorm(
    x,
    y,
    z,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = setup_ttnn_tensor(z, device, layout[2], input_mem_config[2], dtype[2])

    t3 = ttnn.layer_norm(t0, weight=t1, bias=t2)

    return ttnn_tensor_to_torch(t3, output_mem_config)


def layernorm_residual(
    x,
    y,
    z,
    w,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = setup_ttnn_tensor(z, device, layout[2], input_mem_config[2], dtype[2])
    t3 = setup_ttnn_tensor(w, device, layout[3], input_mem_config[3], dtype[3])

    t4 = ttnn.layer_norm(t0, residual_input_tensor=t1, weight=t2, bias=t3)

    return ttnn_tensor_to_torch(t4, output_mem_config)


def layernorm_noweights(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.layer_norm(t0)

    return ttnn_tensor_to_torch(t1, output_mem_config)


def attention_softmax_nomask(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])

    t2 = ttnn.transformer.attention_softmax(t0, head_size=None, attention_mask=None)

    return ttnn_tensor_to_torch(t2, output_mem_config)


def attention_softmax(
    x,
    y,
    *args,
    scalar,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    y[y <= 0.50] = 0
    y[y > 0.50] = 1

    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    if scalar < 0:
        scalar = -scalar

    t2 = ttnn.transformer.attention_softmax(t0, head_size=scalar, attention_mask=t1)

    return ttnn_tensor_to_torch(t2, output_mem_config)


def rmsnorm(
    x,
    y,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[0], input_mem_config[0], dtype[0])

    t2 = ttnn.rms_norm(t0, t1)

    return ttnn_tensor_to_torch(t2, output_mem_config)


def transformer_concatenate_heads(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.transformer.concatenate_heads(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1, output_mem_config)


def abs(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.abs(t0)

    return ttnn_tensor_to_torch(t1)


def acos(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.acos(t0)

    return ttnn_tensor_to_torch(t1)


def acosh(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.acosh(t0)

    return ttnn_tensor_to_torch(t1)


def asin(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.asin(t0)

    return ttnn_tensor_to_torch(t1)


def asinh(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.asinh(t0)

    return ttnn_tensor_to_torch(t1)


def atan(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.atan(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def atan2(
    x,
    y,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[0], input_mem_config[0], dtype[0])

    t2 = ttnn.atan2(t0, t1, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t2)


def atanh(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.atanh(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def cos(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.cos(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def cosh(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.cosh(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def exp(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.exp(t0)

    return ttnn_tensor_to_torch(t1)


def exp2(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.exp2(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def expm1(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.expm1(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def elu(
    x,
    *args,
    alpha,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.elu(t0, alpha, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def erf(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.erf(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def erfc(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.erfc(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def erfinv(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.erfinv(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def hardsigmoid(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.hardsigmoid(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def deg2rad(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.deg2rad(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def hardshrink(
    x,
    *args,
    _lambda,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.hardshrink(t0, _lambda, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def clone(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.clone(t0, memory_config_to_ttnn(output_mem_config), dtype[0])

    return ttnn_tensor_to_torch(t1)


def cbrt(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.cbrt(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def digamma(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.digamma(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def clip(
    x,
    *args,
    low,
    high,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.clip(t0, low, high, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)
