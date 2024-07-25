// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/core.hpp"
#include "ttnn/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/conv2d/conv2d.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/sliding_window_op_infra/sliding_window.hpp"

#include "device/max_pool2d_device_op.hpp"



namespace ttnn {
namespace operations::pool {

struct MaxPoolNewOp {

    static Tensor operator()(uint8_t queue_id, const Tensor& input_tensor, uint32_t batch_size, uint32_t input_h, uint32_t input_w, uint32_t channels, std::array<uint32_t, 2> kernel_size, std::array<uint32_t, 2> stride, std::array<uint32_t, 2> padding, std::array<uint32_t, 2> dilation, Device& device) {
        MemoryConfig memory_config = input_tensor.memory_config();
        const auto shard_grid = memory_config.shard_spec.value().grid;
        const auto shard_scheme = memory_config.memory_layout;
        const auto shard_orientation = memory_config.shard_spec.value().orientation;

        TT_FATAL(shard_scheme == TensorMemoryLayout::HEIGHT_SHARDED, "Only height sharded tensors are supported.");
        TT_FATAL(shard_orientation == ShardOrientation::ROW_MAJOR, "Only row major orientation is supported.");

        ParallelConfig parallel_config = conv2d::determine_parallel_config(
                                            shard_scheme == TensorMemoryLayout::HEIGHT_SHARDED,
                                            batch_size,
                                            0,          // in_channels -- not used
                                            input_h,
                                            input_w,
                                            0,          // out_channels -- not used
                                            device,
                                            shard_orientation);
        uint32_t num_cores_nhw = conv2d::get_num_cores_nhw_from_parallel_config(parallel_config);

        tt::tt_metal::SlidingWindowConfig sliding_window_config = tt::tt_metal::SlidingWindowConfig(
                                                                        batch_size,
                                                                        input_h, input_w,
                                                                        kernel_size.at(0), kernel_size.at(1),
                                                                        stride.at(0), stride.at(1),
                                                                        padding.at(0), padding.at(1),
                                                                        dilation.at(0), dilation.at(1),
                                                                        num_cores_nhw,
                                                                        parallel_config.grid);
        // call the halo uop
        uint32_t neg_inf_pad_val = 0xf7ff;
        auto haloed_tensor = ttnn::operations::halo::halo_op(input_tensor, sliding_window_config, neg_inf_pad_val, false, parallel_config.shard_orientation == ShardOrientation::COL_MAJOR, 0, memory_config);

        MaxPoolNew::operation_attributes_t op_attr{
            .sliding_window_config_ = sliding_window_config,
            .memory_config_ = memory_config};

        // and then call the maxpool uop
        return ttnn::device_operation::run<MaxPoolNew>(
            queue_id,
            op_attr,
            MaxPoolNew::tensor_args_t{.input_tensor_ = haloed_tensor});
    }
};

MaxPoolNew::MultiCore::cached_program_t max_pool_2d_multi_core_sharded_with_halo_v2_new(
                                                                const Tensor &input,
                                                                Tensor& output,
                                                                const SlidingWindowConfig& sliding_window_config,
                                                                const MemoryConfig& out_mem_config);

}  // namespace operations::pool

constexpr auto max_pool2d_new = ttnn::register_operation<"ttnn::max_pool2d_new", operations::pool::MaxPoolNewOp>();

}  // namespace ttnn
