// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/core.hpp"
#include "ttnn/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/device_operation.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/buffers/circular_buffer_types.hpp"
#include "tt_metal/impl/kernels/kernel_types.hpp"

#include "ttnn/operations/conv2d/conv2d.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/sliding_window_op_infra/sliding_window.hpp"


namespace ttnn::operations {
namespace pool {

inline uint32_t ceil_multiple_of(uint32_t n, uint32_t m) {
    return (uint32_t) std::ceil((float) n / m) * m;
}

// new maxpool uop -- called from the macro-op
struct MaxPoolNew {
    struct operation_attributes_t {
        tt::tt_metal::SlidingWindowConfig sliding_window_config_;
        MemoryConfig out_mem_config_;
    };

    struct tensor_args_t {
        const Tensor& input_tensor_;
    };

    using shape_return_value_t = ttnn::Shape;
    using tensor_return_value_t = Tensor;

    struct MultiCore {
        struct shared_variables_t {
            bool something_to_add_later_;
            KernelHandle reader0_kernel;
            KernelHandle reader1_kernel;
            CBHandle raw_in_cb;
            CBHandle in_reader_indices_cb;
            CBHandle cb_out;
            uint32_t ncores;
            uint32_t ncores_w;
        };

        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        cached_program_t create(const operation_attributes_t& operation_attributes,
                                const tensor_args_t& tensor_args,
                                Tensor& output_tensor);
        static void override_runtime_arguments(cached_program_t& cached_program,
                                               const operation_attributes_t& operation_attributes,
                                               const tensor_args_t& tensor_args,
                                               Tensor& output_tensor);
    };

    using program_factory_t = std::variant<MultiCore>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static Shape compute_output_shapes(const operation_attributes_t&, const tensor_args_t&);
    static Tensor create_output_tensors(const operation_attributes_t& operation_attributes, const tensor_args_t&);

    // call old funcs from the above
    static void validate(const Tensor& input, const tt::tt_metal::SlidingWindowConfig& sliding_window_config, const MemoryConfig& out_mem_config);
    static std::vector<Shape> compute_output_shapes(const Tensor& input, const tt::tt_metal::SlidingWindowConfig& sliding_window_config, const MemoryConfig& out_mem_config);
    static std::vector<Tensor> create_output_tensors(const Tensor &input, const tt::tt_metal::SlidingWindowConfig& sliding_window_config, const MemoryConfig& out_mem_config);
    // operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    // static operation::OpPerformanceModel create_op_performance_model(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors, const std::vector<std::optional<Tensor>>& optional_output_tensors, const std::vector<Tensor> &output_tensors);
    static operation::OpPerformanceModel create_op_performance_model(const operation_attributes_t&, const tensor_args_t&, const Tensor&);

    // static constexpr auto attribute_names = std::make_tuple(
    //     "sliding_window_config",
    //     "out_mem_config");
    // const auto attribute_values() const {
    //     return std::make_tuple(
    //         std::cref(this->sliding_window_config_),
    //         std::cref(this->out_mem_config_));
    // }
};

MaxPoolNew::MultiCore::cached_program_t max_pool_2d_multi_core_sharded_with_halo_v2_new(
                                                                const Tensor &input,
                                                                Tensor& output,
                                                                const SlidingWindowConfig& sliding_window_config,
                                                                const MemoryConfig& out_mem_config);

// // maxpool micro-op
// Tensor max_pool2d_uop(const Tensor &input,
//                         const SlidingWindowConfig& sliding_window_config,
//                         uint32_t in_c,
//                         const MemoryConfig& out_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);


// // maxpool macro-op
// inline Tensor max_pool2d_new(const Tensor& input_tensor, uint32_t batch_size, uint32_t input_h, uint32_t input_w, uint32_t channels, std::array<uint32_t, 2> kernel_size, std::array<uint32_t, 2> stride, std::array<uint32_t, 2> padding, std::array<uint32_t, 2> dilation, Device& device) {
//     MemoryConfig memory_config = input_tensor.memory_config();
//     const auto shard_grid = memory_config.shard_spec.value().grid;
//     const auto shard_scheme = memory_config.memory_layout;
//     const auto shard_orientation = memory_config.shard_spec.value().orientation;

//     TT_FATAL(shard_scheme == TensorMemoryLayout::HEIGHT_SHARDED, "Only height sharded tensors are supported.");
//     TT_FATAL(shard_orientation == ShardOrientation::ROW_MAJOR, "Only row major orientation is supported.");

//     ParallelConfig parallel_config = conv2d::determine_parallel_config(
//                                         shard_scheme == TensorMemoryLayout::HEIGHT_SHARDED,
//                                         batch_size,
//                                         0,          // in_channels -- not used
//                                         input_h,
//                                         input_w,
//                                         0,          // out_channels -- not used
//                                         device,
//                                         shard_orientation);
//     uint32_t num_cores_nhw = conv2d::get_num_cores_nhw_from_parallel_config(parallel_config);

//     SlidingWindowConfig sliding_window_config = SlidingWindowConfig(batch_size,
//                                                                     input_h, input_w,
//                                                                     kernel_size.at(0), kernel_size.at(1),
//                                                                     stride.at(0), stride.at(1),
//                                                                     padding.at(0), padding.at(1),
//                                                                     dilation.at(0), dilation.at(1),
//                                                                     num_cores_nhw,
//                                                                     parallel_config.grid);
//     uint32_t neg_inf_pad_val = 0xf7ff;

//     auto haloed_tensor = ttnn::operations::halo::halo_op(input_tensor, sliding_window_config, neg_inf_pad_val, false, parallel_config.shard_orientation == ShardOrientation::COL_MAJOR, 0, memory_config);
//     return max_pool2d_uop(haloed_tensor, sliding_window_config, channels, memory_config);
// }

}  // namespace pool
}  // namespace ttnn::operations
