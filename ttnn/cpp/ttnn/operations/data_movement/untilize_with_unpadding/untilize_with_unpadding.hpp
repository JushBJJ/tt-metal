// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/untilize_with_unpadding_op.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/experimental/tt_dnn/op_library/run_operation.hpp"

namespace ttnn {
namespace operations::data_movement {

struct ExecuteUntilizeWithUnpadding {
    static ttnn::Tensor execute_on_worker_thread(
        uint8_t queue_id,
        const ttnn::Tensor &input_tensor,
        const tt::tt_metal::Shape &output_tensor_end,
        const std::optional<MemoryConfig> &memory_config,
        bool use_multicore = false,
        bool use_pack_untilize = true) {
        // MT: Currently only uint32 is moved to DST directly, fp32 is converted to fp16b
        bool fp32_dest_acc_en = input_tensor.get_dtype() == DataType::UINT32;

        return operation::run(
                   UntilizeWithUnpadding{
                       output_tensor_end,
                       memory_config.value_or(input_tensor.memory_config()),
                       use_multicore,
                       use_pack_untilize,
                       fp32_dest_acc_en},
                   {input_tensor},
                   {},
                   {},
                   queue_id)
            .at(0);
    }

    static ttnn::Tensor execute_on_worker_thread(
        const ttnn::Tensor &input_tensor,
        const tt::tt_metal::Shape &output_tensor_end,
        const std::optional<MemoryConfig> &memory_config,
        bool use_multicore = false,
        bool use_pack_untilize = true) {
        constexpr uint8_t DefaultQueueId = 0;
        return execute_on_worker_thread(
            DefaultQueueId, input_tensor, output_tensor_end, memory_config, use_multicore, use_pack_untilize);
    }
};

}  // namespace operations::data_movement

constexpr auto untilize_with_unpadding =
    ttnn::register_operation<ttnn::operations::data_movement::ExecuteUntilizeWithUnpadding>(
        "ttnn::untilize_with_unpadding");

}  // namespace ttnn
