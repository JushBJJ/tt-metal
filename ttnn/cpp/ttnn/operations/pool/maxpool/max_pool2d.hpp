// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/core.hpp"
#include "ttnn/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"

#include "ttnn/operations/conv2d/conv2d.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/sliding_window_op_infra/sliding_window.hpp"
#include "device/max_pool2d_device_op.hpp"


namespace ttnn {
namespace operations::pool {

struct MaxPoolNewOp {

    static Tensor operator()(uint8_t queue_id, const Tensor& input_tensor, uint32_t batch_size, uint32_t input_h, uint32_t input_w, uint32_t channels, std::array<uint32_t, 2> kernel_size, std::array<uint32_t, 2> stride, std::array<uint32_t, 2> padding, std::array<uint32_t, 2> dilation, Device& device) {

        tt::tt_metal::SlidingWindowConfig sliding_window_config = tt::tt_metal::SlidingWindowConfig(
                                                                        batch_size,
                                                                        input_h, input_w,
                                                                        kernel_size.at(0), kernel_size.at(1),
                                                                        stride.at(0), stride.at(1),
                                                                        padding.at(0), padding.at(1),
                                                                        dilation.at(0), dilation.at(1));
        auto output_shape = sliding_window_config.get_output_shape();
        auto input_tensor_tmp = input_tensor;

        ParallelConfig parallel_config;

        MemoryConfig memory_config = input_tensor.memory_config();
        if (!memory_config.shard_spec.has_value()) {
            parallel_config = conv2d::determine_parallel_config(
                                                true,
                                                batch_size,
                                                0,          // in_channels -- not used
                                                output_shape[1],
                                                output_shape[2],
                                                0,          // out_channels -- not used
                                                device,
                                                ShardOrientation::ROW_MAJOR,
                                                false);

            // log_debug("MaxPoolNewOp: Shard spec not found in input tensor. Executing sharding.");
            auto sharded_mem_config = conv2d::create_sharded_memory_config_from_parallel_config(input_tensor.shape(), parallel_config, 1);
            auto input_tensor_sharded = ttnn::to_memory_config(input_tensor_tmp, sharded_mem_config, std::nullopt);
            ttnn::operations::core::deallocate(input_tensor_tmp);
            input_tensor_tmp = ttnn::operations::core::reallocate(input_tensor_sharded, input_tensor_sharded.memory_config());
            memory_config = input_tensor_tmp.memory_config();
        } else {
            // input is already sharded, use it as is
            const auto shard_grid = memory_config.shard_spec.value().grid;
            const auto shard_scheme = memory_config.memory_layout;
            const auto shard_orientation = memory_config.shard_spec.value().orientation;

            TT_FATAL(shard_scheme == TensorMemoryLayout::HEIGHT_SHARDED, "Only height sharded tensors are supported.");
            TT_FATAL(shard_orientation == ShardOrientation::ROW_MAJOR, "Only row major orientation is supported.");

            parallel_config.grid = shard_grid;
            parallel_config.shard_scheme = shard_scheme;
            parallel_config.shard_orientation = shard_orientation;
        }

        uint32_t num_cores_nhw = conv2d::get_num_cores_nhw_from_parallel_config(parallel_config);

        // tt::tt_metal::SlidingWindowConfig sliding_window_config = tt::tt_metal::SlidingWindowConfig(
        sliding_window_config = tt::tt_metal::SlidingWindowConfig(
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
        auto haloed_tensor = ttnn::operations::halo::halo_op(input_tensor_tmp, sliding_window_config, neg_inf_pad_val, false, parallel_config.shard_orientation == ShardOrientation::COL_MAJOR, 0, memory_config);

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
