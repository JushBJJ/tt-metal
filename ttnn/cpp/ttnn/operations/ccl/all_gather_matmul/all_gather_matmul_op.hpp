// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/ccl/all_gather_matmul/device/all_gather_matmul_op.hpp"
#include "ttnn/cpp/ttnn/multi_device.hpp"

namespace ttnn {
namespace operations {
namespace ccl {

struct ExecuteAllGatherMatmul {

    static std::vector<ttnn::Tensor> execute_on_main_thread(
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& weight_tensor,
        const uint32_t dim,
        const CoreCoord all_gather_core_grid_offset,
        const uint32_t num_links = 1,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        const bool transpose_a = false,
        const bool transpose_b = false,
        const std::optional<const DataType> dtype = std::nullopt,
        const std::optional<const tt::operations::primary::MatmulProgramConfig> program_config = std::nullopt,
        const std::optional<const std::string>& activation = std::nullopt,
        const std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
        const std::optional<const ttnn::CoreGrid> core_grid = std::nullopt) {
        return ttnn::operations::ccl::all_gather_matmul(input_tensor, weight_tensor, dim, all_gather_core_grid_offset, num_links, memory_config, transpose_a, transpose_b, dtype, program_config, activation, compute_kernel_config, core_grid);
    }
};

}  // namespace ccl
}  // namespace operations

constexpr auto all_gather_matmul = ttnn::register_operation<ttnn::operations::ccl::ExecuteAllGatherMatmul>("ttnn::all_gather_matmul");

}  // namespace ttnn
