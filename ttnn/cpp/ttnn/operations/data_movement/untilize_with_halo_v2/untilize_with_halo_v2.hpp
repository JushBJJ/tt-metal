// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/untilize_with_halo_v2_op.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/experimental/tt_dnn/op_library/run_operation.hpp"

namespace ttnn {
namespace operations::data_movement {

struct ExecuteUntilizeWithHaloV2 {
    static ttnn::Tensor execute_on_worker_thread(
        uint8_t queue_id,
        const ttnn::Tensor& input_tensor,
        const Tensor& padding_config,
        const Tensor& local_config,
        const Tensor& remote_config,
        const uint32_t pad_val,
        const uint32_t ncores_nhw,
        const uint32_t max_out_nsticks_per_core,
        const std::optional<MemoryConfig>& memory_config,
        const bool remote_read,
        const bool transpose_mcast) {
        TT_ASSERT(input_tensor.memory_config().is_sharded());
        TT_ASSERT(
            input_tensor.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED ||
            input_tensor.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED);

        return operation::run(
                   UntilizeWithHaloV2{
                       pad_val,
                       ncores_nhw,
                       max_out_nsticks_per_core,
                       memory_config.value_or(input_tensor.memory_config()),
                       remote_read,
                       transpose_mcast},
                   {input_tensor, padding_config, local_config, remote_config},
                   {},
                   {},
                   queue_id)
            .at(0);
    }

    static ttnn::Tensor execute_on_worker_thread(
        const ttnn::Tensor& input_tensor,
        const Tensor& padding_config,
        const Tensor& local_config,
        const Tensor& remote_config,
        const uint32_t pad_val,
        const uint32_t ncores_nhw,
        const uint32_t max_out_nsticks_per_core,
        const std::optional<MemoryConfig>& memory_config,
        const bool remote_read,
        const bool transpose_mcast) {
        constexpr uint8_t DefaultQueueId = 0;
        return execute_on_worker_thread(
            DefaultQueueId,
            input_tensor,
            padding_config,
            local_config,
            remote_config,
            pad_val,
            ncores_nhw,
            max_out_nsticks_per_core,
            memory_config,
            remote_read,
            transpose_mcast);
    }
};

}  // namespace operations::data_movement

constexpr auto untilize_with_halo_v2 =
    ttnn::register_operation<ttnn::operations::data_movement::ExecuteUntilizeWithHaloV2>("ttnn::untilize_with_halo_v2");

}  // namespace ttnn
