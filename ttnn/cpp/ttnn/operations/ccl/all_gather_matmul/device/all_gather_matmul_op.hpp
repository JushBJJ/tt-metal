// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "common/core_coord.h"
#include "impl/buffers/buffer.hpp"
#include "tensor/tensor.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/ccl_common.hpp"

#include "ttnn/cpp/ttnn/run_operation.hpp"

/* Fusion includes */
#include "ttnn/cpp/ttnn/operations/ccl/all_gather/device/all_gather_op.hpp"
#include "ttnn/cpp/ttnn/operations/matmul/device/matmul_op.hpp"

#include <optional>
#include <vector>
#include <tuple>
#include <algorithm>

namespace ttnn {

struct AllGatherMatmul {

    /* All Gather Params */
    const ttnn::AllGather all_gather_struct;

    /* Matmul Params */
    const tt::operations::primary::Matmul matmul_struct;

    /* General */
    void validate(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        std::vector<Tensor> &output_tensors
    ) const;
};

operation::ProgramWithCallbacks all_gather_matmul_multi_core_with_workers(

    /* General Params */
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    Tensor& all_gather_output_tensor,

    /* All Gather Params */
    const uint32_t dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    const std::optional<chip_id_t> receiver_device_id,
    const std::optional<chip_id_t> sender_device_id,
    all_gather_op::Topology topology

    /* Matmul Params */
    // const std::optional<const Tensor> bias,
    // Tensor &mm_output_tensor,
    // bool bcast_batch,
    // CoreCoord compute_with_storage_grid_size,
    // DeviceComputeKernelConfig compute_kernel_config,
    // uint32_t in0_block_w,
    // uint32_t out_subblock_h,
    // uint32_t out_subblock_w,
    // uint32_t per_core_M,
    // uint32_t per_core_N,
    // bool fuse_batch,
    // bool transpose_mcast,
    // std::optional<UnaryWithParam> fused_activation,
    // bool untilize_out

);


namespace operations {
namespace ccl {

std::vector<Tensor> all_gather_matmul(
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    const uint32_t dim,
    const uint32_t num_links = 1,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const bool transpose_a = false,
    const bool transpose_b = false,
    const std::optional<const DataType> dtype = std::nullopt,
    const std::optional<const tt::operations::primary::MatmulProgramConfig> program_config = std::nullopt,
    const std::optional<const std::string>& activation = std::nullopt,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    const std::optional<const ttnn::CoreGrid> core_grid = std::nullopt);

} // namespace ccl
} // namespace operations

}  // namespace ttnn
