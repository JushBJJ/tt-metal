// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/core/core.hpp"

#include "ttnn/run_operation.hpp"

#include "device/upsample_op.hpp"
#include "ttnn/operations/pool/upsample/device/upsample_op.hpp"

namespace ttnn {
namespace operations {
namespace data_movement {

struct ExecuteUpSample {
    static ttnn::Tensor operator()(
        const ttnn::Tensor& input_tensor,
        std::variant<int, std::array<int, 2>, std::array<int, 3>, std::array<int, 4>> scale_factor,
        std::optional<MemoryConfig> output_mem_config = std::nullopt);
};
} // data_movement
} // operations
// constexpr auto upsample = ttnn::
//     register_operation_with_auto_launch_op<"ttnn::upsample", ttnn::operations::data_movement::ExecuteUpsample>();
constexpr auto upsample = ttnn::register_operation_with_auto_launch_op<"ttnn::upsample", ttnn::operations::data_movement::ExecuteUpSample>();
} // data_movement
