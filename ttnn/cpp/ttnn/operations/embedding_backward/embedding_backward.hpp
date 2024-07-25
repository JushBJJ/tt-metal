// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/embedding_backward/device/embedding_backward_device_operation.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn {

namespace operations {

namespace embedding_backward {

struct EmbeddingBackwardOperation {
    static inline Tensor operator()(
        uint8_t queue_id,
        const Tensor& input_tensor_arg,
        const Tensor& weight_tensor_arg,
        const Tensor& output_gradient_tensor_arg,
        const std::optional<const DataType> dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt) {

        auto num_embeddings = weight_tensor_arg.get_shape()[-2];

        auto batch_size = input_tensor_arg.get_shape()[0];
        auto sentence_size = input_tensor_arg.get_shape()[-1];
        auto input_tensor = ttnn::reshape(input_tensor_arg, ttnn::Shape{{batch_size, 1, 1, sentence_size}});

        auto input_gradient = operation::run(
                              EmbeddingBackward{
                                  .output_mem_config = memory_config.value_or(output_gradient_tensor_arg.memory_config()),
                                  .output_dtype = dtype.value_or(output_gradient_tensor_arg.get_dtype()),
                                  .num_embeddings = num_embeddings},
                              {input_tensor, output_gradient_tensor_arg})
                              .at(0);

        return input_gradient;
    }

    static inline auto operator()(
        const Tensor& input_tensor_arg,
        const Tensor& weight_tensor_arg,
        const Tensor& output_gradient_tensor_arg,
        const std::optional<const DataType> dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt
        ) {
            constexpr auto DefaultQueueId = 0;
            return operator()(DefaultQueueId, input_tensor_arg, weight_tensor_arg, output_gradient_tensor_arg, dtype, memory_config, optional_output_tensor);
        }
};

}  // namespace embedding_backward
}  // namespace operations

constexpr auto embedding_bw = ttnn::register_operation_with_auto_launch_op<"ttnn::embedding_bw", ttnn::operations::embedding_backward::EmbeddingBackwardOperation>();

}  // namespace ttnn
