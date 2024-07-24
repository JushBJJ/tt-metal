// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/ccl/all_gather/device/all_gather_op.hpp"
#include "ttnn/multi_device.hpp"

namespace ttnn {
namespace operations {
namespace ccl {

struct ExecuteAllGather {

    static ttnn::Tensor execute_on_main_thread(
        const QueueId queue_id,
        const ttnn::Tensor& input_tensor,
        const uint32_t dim,
        const uint32_t num_links = 1,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt) {
        return ttnn::operations::ccl::all_gather(input_tensor, dim, num_links, memory_config);
    }
};

}  // namespace ccl
}  // namespace operations

constexpr auto all_gather = ttnn::register_operation<ttnn::operations::ccl::ExecuteAllGather>("ttnn::all_gather");

}  // namespace ttnn
