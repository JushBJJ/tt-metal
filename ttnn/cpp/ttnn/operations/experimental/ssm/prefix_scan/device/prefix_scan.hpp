// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::ssm {

struct PrefixScan {
    MemoryConfig memory_config;
    DataType dtype;
    MathFidelity math_fidelity;

    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;
};

inline Tensor prefix_scan(
    const Tensor& a,
    const Tensor& bx,
    const Tensor& h,
    const MemoryConfig& memory_config,
    std::optional<const DataType> dtype = std::nullopt,
    MathFidelity math_fidelity = MathFidelity::HiFi4) {
    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({a, bx, h}))};
    operation::launch_op(
        [memory_config, dtype, math_fidelity](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            const auto& a = input_tensors.at(0);
            const auto& bx = input_tensors.at(1);
            const auto& h = input_tensors.at(2);
            return operation::run(
                PrefixScan{memory_config, dtype.value_or(a.get_dtype()), math_fidelity}, input_tensors);
        },
        {a, bx, h},
        output_tensors);
    return output_tensors.at(0);
}
}  // namespace ttnn::operations::experimental::ssm
