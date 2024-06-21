// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/run_operation.hpp"

namespace ttnn::operations::reduction {

struct TopK {
    const uint16_t k;
    const int8_t dim;
    const bool largest;
    const bool sorted;
    const MemoryConfig output_mem_config;

    void validate_with_output_tensors(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;
    static constexpr auto attribute_names = std::forward_as_tuple("k", "dim", "largest", "sorted", "output_mem_config");
    const auto attribute_values() const {
        return std::forward_as_tuple(this->k, this->dim, this->largest, this->sorted, this->output_mem_config);
    }
};

} // namespace ttnn::operations::reduction
