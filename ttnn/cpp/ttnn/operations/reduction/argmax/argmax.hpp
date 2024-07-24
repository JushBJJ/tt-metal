// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/argmax_op.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn {
namespace operations::reduction {

struct ExecuteArgMax {
    static ttnn::Tensor execute_on_worker_thread(
        QueueId queue_id,
        const Tensor& input_tensor,
        const std::optional<int> dim = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt) {
        return operation::run(
                   ArgMax{tt::tt_metal::DataType::UINT32, dim, memory_config.value_or(input_tensor.memory_config())},
                   {input_tensor}, {}, {optional_output_tensor}, queue_id)
            .at(0);
    }
};

}  // namespace operations::reduction

constexpr auto argmax = ttnn::register_operation<ttnn::operations::reduction::ExecuteArgMax>("ttnn::argmax");

}  // namespace ttnn
