// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>
#include "ttnn/tensor/tensor.hpp"
#include "third_party/magic_enum/magic_enum.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::binary_backward {

enum class BinaryBackwardOpType {
    ATAN2_BW,
    EMBEDDING_BW,
    ADDALPHA_BW,
    SUBALPHA_BW,
    SUB_BW,
    XLOGY_BW,
    HYPOT_BW,
    LDEXP_BW,
    LOGADDEXP_BW,
    LOGADDEXP2_BW,
    SQUARED_DIFFERENCE_BW,
    ADD_BW,
    EQ_BW,
    ASSIGN_BW,
    CONCAT_BW,
    LE_BW,
    RSUB_BW,
    BIAS_GELU_BW,
    GT_BW,
    LT_BW,
    NE_BW,
    GE_BW,
    MIN_BW,
    MAX_BW,
    DIV_BW,
    LERP_BW,
    MUL_BW,
};
struct BinaryBackwardFunction{
static std::function<std::vector<ttnn::Tensor>(const Tensor&, const Tensor&, const Tensor&, const MemoryConfig&)> get_function_type1(BinaryBackwardOpType OpType); //get_function_binary_bw
static std::function<std::vector<ttnn::Tensor>(const Tensor&, const Tensor&, const Tensor&, std::string, const MemoryConfig&)> get_function_type1_w_string(BinaryBackwardOpType OpType);
static std::function<std::vector<std::optional<ttnn::Tensor>>(uint8_t , const Tensor&, const Tensor&, const Tensor&, const MemoryConfig&, const std::vector<bool>&, std::optional<Tensor>, std::optional<Tensor>)> get_function_type3(BinaryBackwardOpType OpType);
static std::function<std::vector<std::optional<ttnn::Tensor>>(const Tensor&, const Tensor&, const Tensor&, const MemoryConfig&, const std::vector<bool>&, std::optional<Tensor>, std::optional<Tensor>)> get_function_type3_wo_qid(BinaryBackwardOpType OpType);
};

//OpHandler_binary_bw : get_function_binary_bw
std::vector<Tensor> _atan2_bw( const Tensor& grad, const Tensor& input, const Tensor& other, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _rsub_bw( const Tensor& grad, const Tensor& input, const Tensor& other, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _embedding_bw( const Tensor& grad, const Tensor& input, const Tensor& other, const std::optional<MemoryConfig>& output_mem_config);

//OpHandler_binary_bw_float_default : get_function_binary_bw_float_default
std::vector<ttnn::Tensor> _subalpha_bw( const Tensor& grad, const Tensor& input, const Tensor& other, float alpha = 1.0f, const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

//OpHandler_binary_bw_float : get_function_binary_bw_float
std::vector<ttnn::Tensor> _lerp_bw( const Tensor& grad, const Tensor& input, const Tensor& other, float weight , const std::optional<MemoryConfig>& output_mem_config);

//OpHandler_binary_bw_int_default : get_function_binary_bw_int_default
std::vector<ttnn::Tensor> _concat_bw( const Tensor& grad, const Tensor& input, const Tensor& other, int dim = 0, const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

//OpHandler_binary_bw_opt_float_default : get_function_binary_bw_opt_float_default
std::vector<std::optional<ttnn::Tensor>> _addalpha_bw( QueueId queue_id, const Tensor& grad, const Tensor& input, const Tensor& other, float alpha = 1.0f, const std::optional<MemoryConfig>& output_mem_config = std::nullopt, const std::vector<bool>& are_required_outputs = std::vector<bool>{true, true}, std::optional<Tensor> input_grad = std::nullopt, std::optional<Tensor> other_grad = std::nullopt);

// OpHandler struct template
template <BinaryBackwardOpType OpType>
struct OpHandler_binary_bw;

template <BinaryBackwardOpType OpType>
struct OpHandler_binary_bw_opt_float_default;

template <BinaryBackwardOpType OpType>
struct OpHandler_binary_bw_float_default;

template <BinaryBackwardOpType OpType>
struct OpHandler_binary_bw_int_default;

template <BinaryBackwardOpType OpType>
struct OpHandler_binary_bw_float;

template <>
struct OpHandler_binary_bw<BinaryBackwardOpType::ATAN2_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, const Tensor& other, const std::optional<MemoryConfig>& output_mem_config ) {
        return _atan2_bw(grad, input, other, output_mem_config);
    }
};

template <>
struct OpHandler_binary_bw<BinaryBackwardOpType::RSUB_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, const Tensor& other, const std::optional<MemoryConfig>& output_mem_config ) {
        return _rsub_bw(grad, input, other, output_mem_config);
    }
};

template <>
struct OpHandler_binary_bw_opt_float_default<BinaryBackwardOpType::ADDALPHA_BW> {
    static std::vector<std::optional<ttnn::Tensor>> handle( QueueId queue_id, const Tensor& grad, const Tensor& input, const Tensor& other, float alpha, const MemoryConfig& output_mem_config, const std::vector<bool>& are_required_outputs, std::optional<Tensor> input_grad, std::optional<Tensor> other_grad ) {
        return _addalpha_bw( queue_id, grad, input, other, alpha, output_mem_config, are_required_outputs, input_grad, other_grad);
    }
};

template <>
struct OpHandler_binary_bw<BinaryBackwardOpType::EMBEDDING_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, const Tensor& other, const std::optional<MemoryConfig>& output_mem_config ) {
        return _embedding_bw(grad, input, other, output_mem_config);
    }
};

template <>
struct OpHandler_binary_bw_float_default<BinaryBackwardOpType::SUBALPHA_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, const Tensor& other, float alpha, const std::optional<MemoryConfig>& output_mem_config ) {
        return _subalpha_bw(grad, input, other, alpha, output_mem_config);
    }
};

template <>
struct OpHandler_binary_bw_int_default<BinaryBackwardOpType::CONCAT_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, const Tensor& other, int dim, const std::optional<MemoryConfig>& output_mem_config ) {
        return _concat_bw(grad, input, other, dim, output_mem_config);
    }
};

template <>
struct OpHandler_binary_bw_float<BinaryBackwardOpType::LERP_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, const Tensor& other, float weight, const std::optional<MemoryConfig>& output_mem_config ) {
        return _lerp_bw(grad, input, other, weight, output_mem_config);
    }
};

// Template functions to get the function pointers
template <BinaryBackwardOpType OpType>
auto get_function_binary_bw() {
    return &OpHandler_binary_bw<OpType>::handle;
}

template <BinaryBackwardOpType OpType>
auto get_function_binary_bw_opt_float_default() {
    return &OpHandler_binary_bw_opt_float_default<OpType>::handle;
}

template <BinaryBackwardOpType OpType>
auto get_function_binary_bw_float_default() {
    return &OpHandler_binary_bw_float_default<OpType>::handle;
}

template <BinaryBackwardOpType OpType>
auto get_function_binary_bw_int_default() {
    return &OpHandler_binary_bw_int_default<OpType>::handle;
}

template <BinaryBackwardOpType OpType>
auto get_function_binary_bw_float() {
    return &OpHandler_binary_bw_float<OpType>::handle;
}
}  // namespace ttnn::operations::binary_backward
