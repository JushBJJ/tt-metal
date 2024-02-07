# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pathlib

MODEL_CACHE_PATH = pathlib.Path().home() / ".cache" / "tenstorrent"

from ttnn.types import (
    TILE_SIZE,
    Device,
    DataType,
    uint16,
    uint32,
    bfloat8_b,
    bfloat16,
    float32,
    MemoryConfig,
    MathFidelity,
    DRAM_MEMORY_CONFIG,
    L1_MEMORY_CONFIG,
    ShardStrategy,
    ShardOrientation,
    DEFAULT_SHARD_ORIENTATION,
    Layout,
    ROW_MAJOR_LAYOUT,
    TILE_LAYOUT,
    StorageType,
    DEVICE_STORAGE_TYPE,
    Shape,
    Tensor,
)

from ttnn.core import (
    has_storage_type_of,
    has_padding,
    is_sharded,
    get_memory_config,
    create_sharded_memory_config,
)

from ttnn.validation import validate_input_tensor

from ttnn.decorators import register_operation, disable_validate_decorator

from ttnn.device import open, close

from ttnn.program_cache import (
    enable_program_cache,
)

from ttnn.operations.core import (
    from_torch,
    to_torch,
    to_device,
    from_device,
    to_layout,
    reshape,
    to_memory_config,
    deallocate,
    reallocate,
    load_tensor,
    dump_tensor,
    unsqueeze_to_4D,
    squeeze,
)

from ttnn.operations.matmul import (
    matmul,
    linear,
)

from ttnn.operations.others import (
    embedding,
    pad_to_tile,
    unpad_from_tile,
    # fused operations
    softmax,
    # reduction operations
    mean,
    upsample,
)

from ttnn.operations.data_movement import (
    concat,
    pad,
    permute,
    split,
    repeat_interleave,
)

from ttnn.operations.unary import (
    exp,
    tanh,
    gelu,
    rsqrt,
    relu,
    silu,
    log,
    sin,
    cos,
    tan,
    asin,
    acos,
    atan,
)

from ttnn.operations.binary import (
    pow,
    add,
    sub,
    subtract,
    mul,
    multiply,
)

from ttnn.operations.relational import (
    gtz,
    ltz,
    gez,
    lez,
    nez,
    eqz,
    gt,
    gte,
    lt,
    lte,
    eq,
    ne,
)

from ttnn.operations.normalization import (
    layer_norm,
    rms_norm,
    group_norm,
)

from ttnn.operations import transformer
from ttnn.operations.conv import Conv2D
from ttnn.operations.pooling import (
    MaxPool2D,
    average_pool2d,
)

import ttnn.tracer
