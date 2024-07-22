// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/ccl/all_gather/device/all_gather_op.hpp"

#include <optional>

namespace ttnn {
namespace operations {
namespace ccl {

struct ExecuteAllGather {

    static ttnn::Tensor execute_on_main_thread(
        const ttnn::Tensor& input_tensor,
        const uint32_t dim,
        const uint32_t num_links = 1,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        const std::size_t num_workers = 0,
        const std::size_t max_channel_size = 0,
        const std::size_t buffers_per_channel = 1) {
        return ttnn::operations::ccl::all_gather(input_tensor, dim, num_links, memory_config, num_workers, max_channel_size, buffers_per_channel);
    }
};

}  // namespace ccl
}  // namespace operations

constexpr auto all_gather = ttnn::register_operation<ttnn::operations::ccl::ExecuteAllGather>("ttnn::all_gather");

}  // namespace ttnn
