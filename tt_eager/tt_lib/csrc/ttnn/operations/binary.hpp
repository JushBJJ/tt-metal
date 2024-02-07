// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>

#include "tt_dnn/op_library/eltwise_binary/eltwise_binary_op.hpp"

namespace py = pybind11;

namespace ttnn {

static const auto DRAM_MEMORY_CONFIG = tt::tt_metal::MemoryConfig{
    .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED, .buffer_type = tt::tt_metal::BufferType::DRAM};
static const auto L1_MEMORY_CONFIG = tt::tt_metal::MemoryConfig{
    .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED, .buffer_type = tt::tt_metal::BufferType::L1};

ttnn::Tensor reshape(const ttnn::Tensor& tensor, const ttnn::Shape& shape) {
    return ttnn::Tensor(tensor.reshape(shape.value()));
}

ttnn::Tensor unsqueeze_to_4D(const ttnn::Tensor& tensor) {
    const auto tensor_shape = tensor.ttnn_shape();
    const auto rank = tensor_shape.rank();
    if (rank == 4) {
        return tensor;
    }
    if (rank > 4) {
        TT_THROW("Tensor rank is greater than 4");
    }

    const auto tensor_shape_4D = tensor_shape.to_rank<4>();
    return ttnn::reshape(tensor, tensor_shape_4D);
}

namespace operations {
namespace binary {

void py_module(py::module& m_binary) {
    m_binary.def(
        "add",
        [](const ttnn::Tensor& input_tensor_a_arg,
           const ttnn::Tensor& input_tensor_b_arg,
           const tt::tt_metal::MemoryConfig& memory_config) {
            auto&& [input_tensor_a, input_tensor_b] = [](const auto& input_tensor_a_arg,
                                                         const auto& input_tensor_b_arg) {
                // Swap tensors if input_tensor_a needs to be broadcasted to input_tensor_b
                if (tt::tt_metal::compute_volume(input_tensor_a_arg.ttnn_shape()) <
                    tt::tt_metal::compute_volume(input_tensor_b_arg.ttnn_shape())) {
                    return std::make_tuple(input_tensor_b_arg, input_tensor_a_arg);
                    }
                return std::make_tuple(input_tensor_a_arg, input_tensor_b_arg);
            }(input_tensor_a_arg, input_tensor_b_arg);

            const auto original_shape = input_tensor_a.ttnn_shape();
            const auto input_shape_b = input_tensor_b.ttnn_shape();

            std::size_t height_b{};
            std::size_t width_b{};
            if (input_shape_b.rank() == 1) {
                height_b = 1;
                width_b = input_shape_b[-1];
            } else {
                height_b = input_shape_b[-2];
                width_b = input_shape_b[-1];
            }

            auto input_tensor_a_4D = ttnn::unsqueeze_to_4D(input_tensor_a);
            auto input_tensor_b_4D = ttnn::unsqueeze_to_4D(input_tensor_b);

            if (height_b == 1 or width_b == 1) {
                tt::tt_metal::BcastOpDim bcast_op_dim;
                if (height_b == 1 and width_b == 1) {
                    bcast_op_dim = tt::tt_metal::BcastOpDim::HW;
                } else if (height_b == 1) {
                    bcast_op_dim = tt::tt_metal::BcastOpDim::H;
                } else if (width_b == 1) {
                    bcast_op_dim = tt::tt_metal::BcastOpDim::W;
                } else {
                    TT_THROW("Invalid broadcasting dimensions");
                }
                auto output = tt::tt_metal::bcast(
                    input_tensor_a_4D, input_tensor_b_4D, tt::tt_metal::BcastOpMath::ADD, bcast_op_dim, memory_config);
                return ttnn::reshape(output, original_shape);
            } else {
                auto output = tt::tt_metal::add(input_tensor_a_4D, input_tensor_b_4D, std::nullopt, memory_config);
                return ttnn::reshape(output, original_shape);
            }
        },
        py::arg("input_tensor_a"),
        py::arg("input_tensor_b"),
        py::kw_only(),
        py::arg("memory_config") = DRAM_MEMORY_CONFIG

    );

    m_binary.def(
        "add",
        [](const ttnn::Tensor& input_tensor_a,
           const float input_tensor_b,
           const tt::tt_metal::MemoryConfig& memory_config) {
            const auto original_shape = input_tensor_a.ttnn_shape();

            auto input_tensor_a_4D = ttnn::unsqueeze_to_4D(input_tensor_a);

            auto output = tt::tt_metal::add_unary(input_tensor_a_4D, input_tensor_b, memory_config);
            return ttnn::reshape(output, original_shape);
        },
        py::arg("input_tensor_a"),
        py::arg("input_tensor_b"),
        py::kw_only(),
        py::arg("memory_config") = DRAM_MEMORY_CONFIG);
}

}  // namespace binary
}  // namespace operations
}  // namespace ttnn
