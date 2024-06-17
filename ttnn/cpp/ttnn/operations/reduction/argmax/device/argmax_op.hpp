// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/run_operation.hpp"

namespace ttnn::operations::reduction {

constexpr uint8_t DefaultQueueId = 0;

struct ArgMax {
    const DataType output_dtype;
    const std::optional<int> dim;
    const MemoryConfig output_mem_config;

    void validate_with_output_tensors(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;
    static constexpr auto attribute_names = std::forward_as_tuple("output_dtype", "dim", "output_mem_config");
    const auto attribute_values() const {
        return std::forward_as_tuple(this->output_dtype, this->dim, this->output_mem_config);
    }
};


}  // namespace ttnn::operations::reduction
