// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "prefix_scan.hpp"
#include "ttnn/cpp/pybind11/decorators.hpp"

namespace ttnn::operations::experimental::ssm::detail {
namespace py = pybind11;

void bind_ssm_prefix_scan(py::module& module) {
    ttnn::bind_registered_operation(
        module,
        ttnn::prefix_scan,
        R"doc(Performs a prefix scan to produce the SSM hidden states across an entire sequence. All input and output tensors are expected to be shape [1, 1, L, 2EN] where E = 2560 and N = 32. L can be any multiple of 32.)doc",
        ttnn::pybind_arguments_t{
            py::arg("a"),
            py::arg("bx"),
            py::arg("h_prev"),
            py::arg("memory_config") = std::nullopt,
            py::arg("dtype") = std::nullopt,
            py::arg("math_fidelity") = std::nullopt});
}

}  // namespace ttnn::operations::experimental::ssm::detail
