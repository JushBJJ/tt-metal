// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "ttnn/operations/data_movement.hpp"

namespace py = pybind11;

namespace ttnn {
namespace operations {
namespace data_movement {
void py_module(py::module& module) {

    module.def("permute", &permute,
        py::arg("input_tensor"),
        py::arg("order"),
        R"doc(
Permutes :attr:`input_tensor` using :attr:`order`.

Args:
    * :attr:`input_tensor`: the input tensor
    * :attr:`order`: the desired ordering of dimensions.

Example::

    >>> tensor = ttnn.to_device(ttnn.from_torch(torch.zeros((1, 1, 64, 32), dtype=torch.bfloat16)), device)
    >>> output = ttnn.permute(tensor, (0, 1, 3, 2))
    >>> print(output.shape)
    [1, 1, 32, 64]

    )doc");

    module.def("concat", &concat,
        py::arg("input_tensor"),
        py::arg("dim") = 0,
        py::kw_only(),
        py::arg("memory_config") = std::nullopt,
        R"doc(
Concats :attr:`tensors` in the given :attr:`dim`.

Args:
    * :attr:`tensors`: the tensors to be concatenated.
    * :attr:`dim`: the concatenating dimension.

Keyword Args:
    * :attr:`memory_config`: the memory configuration to use for the operation

Example::

    >>> tensor = ttnn.concat(ttnn.from_torch(torch.zeros((1, 1, 64, 32), ttnn.from_torch(torch.zeros((1, 1, 64, 32), dim=3)), device)

    >>> tensor1 = ttnn.from_torch(torch.zeros((1, 1, 64, 32), dtype=torch.bfloat16), device=device)
    >>> tensor2 = ttnn.from_torch(torch.zeros((1, 1, 64, 32), dtype=torch.bfloat16), device=device)
    >>> output = ttnn.concat([tensor1, tensor2], dim=4)
    >>> print(output.shape)
    [1, 1, 32, 64]

    )doc");

    ttnn::bind_registered_operation(
        module,
        ttnn::upsample,
        R"doc(
Upsamples a given multi-channel 2D (spatial) data.
The input data is assumed to be of the form [N, H, W, C].

The algorithms available for upsampling are 'nearest' for now.

Args:
    * :attr:`input_tensor`: the input tensor
    * :attr:`scale_factor`: multiplier for spatial size. Has to match input size if it is a tuple.
    )doc",
        ttnn::pybind_arguments_t{
            py::arg("input_tensor"),
            py::arg("scale_factor"),
            py::arg("memory_config") = std::nullopt
        }
    );
}

}  // namespace data_movement
}  // namespace operations
}  // namespace ttnn
