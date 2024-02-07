// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>

#include "tt_dnn/op_library/upsample/upsample_op.hpp"
#include "tt_dnn/op_library/pool/max_pool.hpp"
#include "tt_dnn/op_library/reduce/reduce_op.hpp"   // for reduce_op_utils
#include "tt_dnn/op_library/work_split.hpp"
#include "tt_metal/host_api.hpp"
#include "tensor/tensor_utils.hpp"
#include "tensor/owned_buffer_functions.hpp"
#include "detail/util.hpp"

namespace tt {
namespace tt_metal {

void UpSample::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to copy need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr , "Operands to copy need to be allocated in buffers on device!");
    // TT_FATAL(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);
    TT_FATAL(input_tensor_a.layout() == Layout::ROW_MAJOR, "Input tensor layout should be ROW_MAJOR");
    TT_FATAL(input_tensor_a.dtype() == DataType::BFLOAT16, "Input tensor data type should be BFLOAT16");
    if (input_tensor_a.memory_config().is_sharded()) {
        TT_FATAL(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED, "Input tensor memory layout should be HEIGHT_SHARDED");
        TT_FATAL(input_tensor_a.buffer()->buffer_type() == tt_metal::BufferType::L1, "Input buffer should be sharded in L1");
    }
}

std::vector<Shape> UpSample::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    // NOTE1: data is packed in { N, H , W, C }
    // NOTE2: Mapping it into in 2D format should be {N*H*W, C}
    // NOTE3: Assuming output data type is same as input
    const auto& input = input_tensors.at(0);
    const auto input_shape = input.shape().without_padding();

    uint32_t out_n = input_shape[0];
    uint32_t out_h = input_shape[1] * scale_factor_h_;
    uint32_t out_w = input_shape[2] * scale_factor_w_;
    uint32_t out_c = input_shape[3];
    const auto out_dims = std::vector<uint32_t>({ out_n, out_h, out_w, out_c }); //in the NHWC format
    auto out_shape = Shape{out_dims};

    return {out_shape};
}

std::vector<Tensor> UpSample::create_output_tensors(const std::vector<Tensor> &inputs) const {
    const auto& input = inputs.at(0);
    if (output_mem_config_.is_sharded()) {
        if (input.memory_config().is_sharded()) {
            auto mem_config = output_mem_config_;
            auto input_shard_spec = input.memory_config().shard_spec.value();
            auto ncores = input_shard_spec.num_cores();
            auto output_shape = compute_output_shapes(inputs).at(0);
            array<uint32_t, 2> output_shard_shape = {output_shape[0] * output_shape[1] * output_shape[2] / ncores, output_shape[-1]};
            auto output_shard_spec = input_shard_spec;
            output_shard_spec.shape = output_shard_shape;
            mem_config.shard_spec = output_shard_spec;
            return {create_sharded_device_tensor(output_shape, input.dtype(), input.layout(), input.device(), mem_config)};
        } else {
            TT_FATAL(false, "Output memory config is sharded but input memory config is not sharded");
        }
    } else {
        return operation::generic_create_output_tensors(*this, inputs, input.dtype(), input.layout(), output_mem_config_);
    }
}

 operation::ProgramWithCallbacks UpSample::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const Tensor& input_tensor_0 = input_tensors.at(0);
    Tensor& output_tensor_0 = output_tensors.at(0);
    switch (get_parallelization_strategy(input_tensors)) {
        case UpSampleParallelizationStrategy::MULTI_CORE:
            return upsample_multi_core(input_tensor_0, output_tensor_0, scale_factor_h_, scale_factor_w_);
        case UpSampleParallelizationStrategy::SINGLE_CORE:
            return upsample_single_core(input_tensor_0, output_tensor_0, scale_factor_h_, scale_factor_w_);
    };
    return upsample_single_core(input_tensor_0, output_tensor_0, scale_factor_h_, scale_factor_w_);
}

UpSampleParallelizationStrategy UpSample::get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const {
    auto input = input_tensors.at(0);
    if (input.memory_config().is_sharded()) {
        return UpSampleParallelizationStrategy::MULTI_CORE;
    }
    return UpSampleParallelizationStrategy::SINGLE_CORE;
}

Tensor upsample(const Tensor &input,
                int scale_factor_h,
                int scale_factor_w,
                const MemoryConfig& out_mem_config) {
    return operation::run_without_autoformat(UpSample{scale_factor_h,
                                                      scale_factor_w,
                                                      out_mem_config},
                                              {input}).at(0);
}

} // namespace tt_metal
} // namespace tt
