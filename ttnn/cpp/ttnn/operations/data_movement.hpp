// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/types.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/concat/concat_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/repeat/repeat_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/composite/composite_ops.hpp"
#include "ttnn/operations/core/core.hpp"

#include <ranges>

namespace ttnn {
namespace operations {
namespace data_movement {


struct Repeat {
    static ttnn::Tensor operator()(
        const ttnn::Tensor& input_tensor,
        const ttnn::Shape& shape,
        std::optional<MemoryConfig> output_mem_config = std::nullopt) {
        MemoryConfig mem_config = output_mem_config.value_or(input_tensor.memory_config());
        auto output_tensor = tt::tt_metal::repeat(input_tensor, shape.value(), mem_config);
        return output_tensor;
    }
};

struct RepeatInterleave {

    // # This operation does not support the following cases:
    // #   - Shape([2[32], 2[32]]) -> repeats = 2, dim = 0
    // #   - Shape([2[32], 2[32]]) -> repeats = Tensor[1,2], dim = 1
    static ttnn::Tensor operator()(
        const ttnn::Tensor& input_tensor,
        uint32_t repeats,
        int32_t dim,
        std::optional<MemoryConfig> output_mem_config = std::nullopt) {
        MemoryConfig mem_config = output_mem_config.value_or(input_tensor.memory_config());
        auto output_tensor = tt::tt_metal::repeat_interleave(input_tensor, repeats, dim, mem_config);
        return output_tensor;
    }
};

}  // namespace data_movement
}  // namespace operations

constexpr auto repeat = ttnn::register_operation_with_auto_launch_op<"ttnn::repeat", ttnn::operations::data_movement::Repeat>();
constexpr auto repeat_interleave =
    ttnn::register_operation_with_auto_launch_op<"ttnn::repeat_interleave", ttnn::operations::data_movement::RepeatInterleave>();

}  // namespace ttnn
