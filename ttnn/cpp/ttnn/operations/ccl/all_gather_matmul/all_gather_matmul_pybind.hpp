// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operations/ccl/all_gather_matmul/all_gather_matmul_op.hpp"
#include "ttnn/types.hpp"

namespace py = pybind11;

namespace ttnn {
namespace operations {
namespace ccl {

namespace detail {

template <typename ccl_operation_t>
void bind_all_gather_matmul(py::module& module, const ccl_operation_t& operation, const char* doc) {
    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const ccl_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::Tensor& weight_tensor,
               const uint32_t dim,
               const uint32_t num_links,
               const std::optional<ttnn::MemoryConfig>& memory_config
               const bool transpose_a = false,
               const bool transpose_b = false,
               const std::optional<const DataType> dtype,
               const std::optional<const ttnn::MatmulProgramConfig> program_config,
               const std::optional<const std::string>& activation,
               const std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
               const std::optional<const ttnn::CoreGrid> core_grid) -> ttnn::Tensor {
                return self(input_tensor, weight_tensor, dim, num_links, memory_config, transpose_a, transpose_b, dtype, program_config, activation, compute_kernel_config, core_grid);
            },
            py::arg("input_tensor"),
            py::arg("weight_tensor"),
            py::arg("dim"),
            py::kw_only(),
            py::arg("num_links") = 1,
            py::arg("memory_config") = std::nullopt
            py::arg("transpose_a") = false,
            py::arg("transpose_b") = false,
            py::arg("dtype") = std::nullopt,
            py::arg("program_config") = std::nullopt,
            py::arg("activation") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt,
            py::arg("core_grid") = std::nullopt});
}

}  // namespace detail

// TODO: Update with all_gather_matmul docs
void py_bind_all_gather_matmul(py::module& module) {
    detail::bind_all_gather_matmul(
        module,
        ttnn::all_gather_matmul,
        R"doc(all_gather_matmul(input_tensor: ttnn.Tensor, weight_tensor: ttnn.Tensor, dim: int, *, num_links: int = 1, memory_config: Optional[ttnn.MemoryConfig] = None) -> (ttnn.Tensor, ttnn.Tensor)

        Performs an all-gather operation on multi-device :attr:`input_tensor` across all devices.

        Args:
            * :attr:`input_tensor` (ttnn.Tensor): multi-device tensor
            * :attr:`weight_tensor` (ttnn.Tensor): multi-device tensor
            * :attr:`dim` (int)

        Keyword Args:
            * :attr:`num_links` (int): Number of links to use for the all-gather operation.
            * :attr:`memory_config` (Optional[ttnn.MemoryConfig]): Memory configuration for the operation.
            * :attr:`transpose_a` (bool)
            * :attr:`transpose_b` (bool)
            * :attr:`dtype` (Optional[DataType])
            * :attr:`program_config` (Optional[ttnn.MatmulProgramConfig])
            * :attr:`activation` (Optional[str])
            * :attr:`compute_kernel_config` (Optional[DeviceComputeKernelConfig])
            * :attr:`core_grid` (Optional[ttnn.CoreGrid])

        Example:

            >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> weight_tensor = ttnn.from_torch(torch.tensor((2, 1), dtype=torch.bfloat16), device=device)
            >>> all_gathered_mm_in, mm_out = ttnn.all_gather_matmul(tensor, weight_tensor, dim=0)

        )doc");
}

}  // namespace ccl
}  // namespace operations
}  // namespace ttnn
