// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/base_types.hpp"
#include "common/core_coord.h"
#include "tensor/types.hpp"
#include "tt_eager/tensor/tensor.hpp"

#include "tt_dnn/op_library/operation.hpp"

#include "tt_dnn/op_library/compute_kernel_config.hpp"

namespace tt {
namespace operations {
namespace primary {

using namespace tt_metal;

namespace transformers {
struct SDPADefaultProgramConfig{
    tt::stl::reflection::Attributes attributes() const { return {}; };
};

struct SDPAProgramConfig {
    CoreCoord compute_with_storage_grid_size;
    std::size_t chunk_size;

    tt::stl::reflection::Attributes attributes() const {
        return {
            {"compute_with_storage_grid_size", compute_with_storage_grid_size},
            {"chunk_size", chunk_size}
        };
    };
};

using SDPAProgramConfig = std::variant<
    SDPADefaultProgramConfig,
    SDPAProgramConfig
>;

}  // namespace transformers

struct ScaledDotProductAttention {
    const std::optional<float> scale;
    const MemoryConfig output_mem_config;
    const tt::operations::primary::transformers::SDPAProgramConfig program_config;
    const bool is_causal;
    const DeviceComputeKernelConfig compute_kernel_config;

    void validate(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        std::vector<Tensor> &output_tensors
    ) const;
    tt::stl::reflection::Attributes attributes() const;

    const operation::Hash compute_program_hash(
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors) const;
};

operation::ProgramWithCallbacks sdpa_multi_core(
    const Tensor &input_tensor_q,
    const Tensor &input_tensor_kt,
    const Tensor &input_tensor_v,
    const Tensor &output_tensor,
    const std::optional<const Tensor> attn_mask,
    std::optional<float> scale,
    bool is_causal,
    DeviceComputeKernelConfig compute_kernel_config
);

namespace transformers {

Tensor scaled_dot_product_attention(Tensor& input_tensor_q, Tensor& input_tensor_k, Tensor& input_tensor_v, std::optional<const Tensor> attn_mask = std::nullopt, const bool is_causal = false, std::optional<float> scale = std::nullopt, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, const SDPAProgramConfig& program_config = SDPADefaultProgramConfig{}, std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

}   // namespace transformers

}   // namespace primary
}   // namespace operations
}   // namespace tt
