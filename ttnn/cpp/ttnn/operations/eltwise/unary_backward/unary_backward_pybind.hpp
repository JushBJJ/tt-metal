// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operations/eltwise/unary_backward/unary_backward.hpp"
#include "ttnn/types.hpp"

namespace py = pybind11;

namespace ttnn {
namespace operations {
namespace unary_backward {

namespace detail {

template <typename unary_backward_operation_t>
void bind_unary_backward(py::module& module, const unary_backward_operation_t& operation, const std::string& description) {
    auto doc = fmt::format(
R"doc({0}(grad_tensor: ttnn.Tensor, input_tensor: ttnn.Tensor *, memory_config: ttnn.MemoryConfig) -> std::vector<Tensor>

{2}

Args:
    * :attr:`grad_tensor`
    * :attr:`input_tensor`

Keyword args:
    * :attr:`memory_config` (Optional[ttnn.MemoryConfig]): memory config for the output tensor

Example:

    >>> grad_tensor = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
    >>> input = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
>>> output = {1}(grad_tensor, input)
)doc",
        operation.name(),
        operation.python_fully_qualified_name(),
        description);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const unary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               const std::optional<ttnn::MemoryConfig>& memory_config) -> std::vector<ttnn::Tensor> {
                auto output_memory_config = memory_config.value_or(input_tensor.memory_config());
                return self(grad_tensor, input_tensor, output_memory_config);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt},


        ttnn::pybind_overload_t{
            [](const unary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor,
               const float alpha,
               const std::optional<ttnn::MemoryConfig>& memory_config) -> std::vector<ttnn::Tensor> {
                return self(grad_tensor, input_tensor, alpha, memory_config);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor"),
            py::arg("alpha"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt},

        ttnn::pybind_overload_t{
            [](const unary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               const std::optional<ttnn::MemoryConfig>& memory_config) -> std::vector<ttnn::Tensor> {
                return self(grad_tensor, input_tensor_a, input_tensor_b, memory_config);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt});

        // ttnn::pybind_overload_t{
        //     [](const unary_backward_operation_t& self,
        //        const ttnn::Tensor& grad_tensor,
        //        const ttnn::Tensor& input_tensor_a,
        //        const ttnn::Tensor& input_tensor_b,
        //        const std::optional<ttnn::MemoryConfig>& memory_config,
        //        const std::vector<bool>& are_required_outputs,
        //        const std::optional<ttnn::Tensor>& input_a_grad,
        //        const std::optional<ttnn::Tensor>& input_b_grad,
        //        const uint8_t& queue_id) -> std::vector<optional<ttnn::Tensor>> {
        //         return self(queue_id, grad_tensor, input_tensor_a, input_tensor_b, memory_config, are_required_outputs, input_a_grad, input_b_grad);
        //     },
        //     py::arg("grad_tensor"),
        //     py::arg("input_tensor_a"),
        //     py::arg("input_tensor_b"),
        //     py::kw_only(),
        //     py::arg("memory_config") = std::nullopt,
        //     py::arg("are_required_outputs") = std::vector<bool>{true, true},
        //     py::arg("input_a_grad") = std::nullopt,
        //     py::arg("input_b_grad") = std::nullopt,
        //     py::arg("queue_id") = 0},

        // ttnn::pybind_overload_t{
        //     [](const unary_backward_operation_t& self,
        //        const ttnn::Tensor& grad_tensor,
        //        const ttnn::Tensor& input_tensor_a,
        //        const ttnn::Tensor& input_tensor_b,
        //        const std::optional<ttnn::MemoryConfig>& memory_config,
        //        const std::vector<bool>& are_required_outputs,
        //        const std::optional<ttnn::Tensor>& input_a_grad,
        //        const std::optional<ttnn::Tensor>& input_b_grad) -> std::vector<optional<ttnn::Tensor>> {
        //         return self(grad_tensor, input_tensor_a, input_tensor_b, memory_config, are_required_outputs, input_a_grad, input_b_grad);
        //     },
        //     py::arg("grad_tensor"),
        //     py::arg("input_tensor_a"),
        //     py::arg("input_tensor_b"),
        //     py::kw_only(),
        //     py::arg("memory_config") = std::nullopt,
        //     py::arg("are_required_outputs") = std::vector<bool>{true, true},
        //     py::arg("input_a_grad") = std::nullopt,
        //     py::arg("input_b_grad") = std::nullopt});

}

}  // namespace detail


void py_module(py::module& module) {
    detail::bind_unary_backward(
        module,
        ttnn::mul_bw,
        R"doc(Performs backward operations for multiply on :attr:`input_tensor`, :attr:`alpha` with given :attr:`grad_tensor`.)doc");

    detail::bind_unary_backward(
        module,
        ttnn::clamp_min_bw,
        R"doc(Performs backward operations for clamp min value on :attr:`input_tensor`, :attr:`alpha` with given :attr:`grad_tensor`.)doc");

}

}  // namespace binary_backward
}  // namespace operations
}  // namespace ttnn
