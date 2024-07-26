// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>
#include "ttnn/tensor/tensor.hpp"
#include "third_party/magic_enum/magic_enum.hpp"
#include "ttnn/operations/eltwise/binary_backward/device/binary_backward_op.hpp"

namespace ttnn::operations::unary_backward {

constexpr uint8_t DefaultQueueId = 0;
enum class UnaryBackwardOpType {
    MUL_BW,
    CLAMP_BW,
    HARDTANH_BW,
    THRESHOLD_BW,
    SOFTPLUS_BW,
    DIV_BW,
    RDIV_BW,
    BIAS_GELU_BW,
    POW_BW,
    TANH_BW,
    EXP_BW,
    SQRT_BW,
    ASSIGN_BW,
    MULTIGAMMALN_BW,
    ADD_BW,
    EQ_BW,
    GT_BW,
    LT_BW,
    LE_BW,
    GE_BW,
    NE_BW,
    LGAMMA_BW,
    FILL_BW,
    HARDSIGMOID_BW,
    COS_BW,
    ACOSH_BW,
    ACOS_BW,
    ATAN_BW,
    RAD2DEG_BW,
    SUB_BW,
    FRAC_BW,
    TRUNC_BW,
    LOG_SIGMOID_BW,
    FILL_ZERO_BW,
    I0_BW,
    TAN_BW,
    SIGMOID_BW,
    RSQRT_BW,
    NEG_BW,
    RELU_BW,
    LOGIT_BW,
    HARDSHRINK_BW,
    SOFTSHRINK_BW,
    LEAKY_RELU_BW,
    ELU_BW,
    CELU_BW,
    RPOW_BW,
    FLOOR_BW,
    ROUND_BW,
    LOG_BW,
    RELU6_BW,
    ABS_BW,
    SILU_BW,
    SELU_BW,
    SQUARE_BW,
    HARDSWISH_BW,
    TANHSHRINK_BW,
    ATANH_BW,
    ASIN_BW,
    ASINH_BW,
    SIN_BW,
    SINH_BW,
    LOG10_BW,
    LOG1P_BW,
    ERFC_BW,
    CEIL_BW,
    SOFTSIGN_BW,
    COSH_BW,
    LOGITEPS_BW,
    LOG2_BW,
    SIGN_BW,
    FMOD_BW,
    REMAINDER_BW,
    DIV_NO_NAN_BW,
    EXP2_BW,
    EXPM1_BW,
    RECIPROCAL_BW,
    DIGAMMA_BW,
    ERFINV_BW,
    ERF_BW,
    DEG2RAD_BW,
    POLYGAMMA_BW,
    GELU_BW,
    REPEAT_BW,
    PROD_BW,
};

struct UnaryBackwardFunction{
    //TODO: Use get_function_unary_optional , get_function_unary_optional_float after optional tensor support
    static std::function<std::vector<ttnn::Tensor>(const Tensor&, const Tensor&, const MemoryConfig&)> get_function_type1(UnaryBackwardOpType OpType);
    static std::function<std::vector<ttnn::Tensor>(const Tensor&, const Tensor&, float, const MemoryConfig&)> get_function_type1_w_float(UnaryBackwardOpType OpType);
};

std::vector<Tensor> _threshold_bw( const Tensor& grad, const Tensor& input, float threshold, float value, const std::optional<MemoryConfig>& output_mem_config);

std::vector<Tensor> _remainder_bw( const Tensor& grad, const Tensor& input, float scalar, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _fmod_bw( const Tensor& grad, const Tensor& input, float scalar, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _sub_bw( const Tensor& grad, const Tensor& input, float scalar, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _gt_bw( const Tensor& grad, const Tensor& input, float other, const std::optional<MemoryConfig>& output_mem_config);

std::vector<Tensor> _assign_bw( const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _multigammaln_bw( const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _lgamma_bw( const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _fill_bw( const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _hardsigmoid_bw( const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _cos_bw( const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _acosh_bw( const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config);

std::vector<Tensor> _softplus_bw( const Tensor& grad, const Tensor& input, float beta = 1.0, float threshold = 20.0, const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
std::vector<Tensor> _hardtanh_bw( const Tensor& grad, const Tensor& input, float min = -1.0, float max = 1.0, const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> _add_bw( const Tensor& grad, const Tensor& input, float alpha, const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
std::vector<Tensor> _mul_bw( const Tensor& grad, const Tensor& input, float scalar, const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
std::vector<Tensor> _eq_bw( const Tensor& grad, const Tensor& input, float other, const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> _clamp_bw( const Tensor& grad, const Tensor& input, std::optional<float> min = std::nullopt, std::optional<float> max = std::nullopt, const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> _div_bw( const Tensor& grad, const Tensor& input, float scalar, string round_mode = "None", const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
std::vector<Tensor> _rdiv_bw( const Tensor& grad, const Tensor& input, float scalar, string round_mode = "None", const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
std::vector<Tensor> _bias_gelu_bw( const Tensor& grad, const Tensor& input, float bias, string approximate = "none", const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> _gelu_bw( const Tensor& grad, const Tensor& input, string approximate = "none", const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> _repeat_bw(const Tensor& grad, const Tensor& input, const tt::tt_metal::Shape& shape, const std::optional<MemoryConfig>& output_mem_config);

std::vector<std::optional<Tensor>> _pow_bw(uint8_t queue_id, const Tensor& grad, const Tensor& input, float exponent, const MemoryConfig& output_mem_config , const std::vector<bool>& are_required_outputs, std::optional<Tensor> input_grad);

std::vector<std::optional<Tensor>> _exp_bw(uint8_t queue_id, const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config, const std::vector<bool>& are_required_outputs, std::optional<Tensor> input_grad);
std::vector<std::optional<Tensor>> _tanh_bw(uint8_t queue_id, const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config, const std::vector<bool>& are_required_outputs, std::optional<Tensor> input_grad);
std::vector<std::optional<Tensor>> _sqrt_bw(uint8_t queue_id, const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config, const std::vector<bool>& are_required_outputs, std::optional<Tensor> input_grad);

std::vector<Tensor> _prod_bw( const Tensor& grad, const Tensor& input, bool all_dimensions = true, int64_t dim = 0, const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
Tensor change_layout_to_tile(const Tensor& temp, const MemoryConfig& output_mem_config);

// OpHandler struct template
template <UnaryBackwardOpType OpType>
struct OpHandler;

template <>
struct OpHandler<UnaryBackwardOpType::CLAMP_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, std::optional<float> min, std::optional<float> max, const std::optional<MemoryConfig>& output_mem_config ) {
        return _clamp_bw(grad, input, min, max, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::HARDTANH_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, float min, float max, const std::optional<MemoryConfig>& output_mem_config ) {
        return _hardtanh_bw(grad, input, min, max, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::THRESHOLD_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, float threshold, float value, const std::optional<MemoryConfig>& output_mem_config ) {
        return _threshold_bw(grad, input, threshold, value, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::REMAINDER_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, float scalar, const std::optional<MemoryConfig>& output_mem_config ) {
        return _remainder_bw(grad, input, scalar, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::FMOD_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, float scalar, const std::optional<MemoryConfig>& output_mem_config ) {
        return _fmod_bw(grad, input, scalar, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::GT_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, float other, const std::optional<MemoryConfig>& output_mem_config ) {
        return _gt_bw(grad, input, other, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::ASSIGN_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config ) {
        return _assign_bw(grad, input, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::MULTIGAMMALN_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config ) {
        return _multigammaln_bw(grad, input, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::LGAMMA_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config ) {
        return _lgamma_bw(grad, input, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::FILL_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config ) {
        return _fill_bw(grad, input, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::HARDSIGMOID_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config ) {
        return _hardsigmoid_bw(grad, input, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::COS_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config ) {
        return _cos_bw(grad, input, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::ACOSH_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config ) {
        return _acosh_bw(grad, input, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::SUB_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, float scalar, const std::optional<MemoryConfig>& output_mem_config ) {
        return _sub_bw(grad, input, scalar, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::SOFTPLUS_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, float beta, float threshold, const std::optional<MemoryConfig>& output_mem_config ) {
        return _softplus_bw(grad, input, beta, threshold, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::DIV_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, float scalar, string round_mode, const std::optional<MemoryConfig>& output_mem_config ) {
        return _div_bw(grad, input, scalar, round_mode, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::RDIV_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, float scalar, string round_mode, const std::optional<MemoryConfig>& output_mem_config ) {
        return _rdiv_bw(grad, input, scalar, round_mode, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::BIAS_GELU_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, float bias, string approximate, const std::optional<MemoryConfig>& output_mem_config ) {
        return _bias_gelu_bw(grad, input, bias, approximate, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::POW_BW> {
    static std::vector<std::optional<Tensor>> handle( uint8_t queue_id, const Tensor& grad, const Tensor& input, float exponent, const MemoryConfig& output_mem_config, const std::vector<bool>& are_required_outputs, std::optional<Tensor> input_grad ) {
        return _pow_bw(queue_id, grad, input, exponent, output_mem_config, are_required_outputs, input_grad);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::EXP_BW> {
    static std::vector<std::optional<Tensor>> handle( uint8_t queue_id, const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config, const std::vector<bool>& are_required_outputs, std::optional<Tensor> input_grad ) {
        return _exp_bw(queue_id, grad, input, output_mem_config, are_required_outputs, input_grad);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::TANH_BW> {
    static std::vector<std::optional<Tensor>> handle( uint8_t queue_id, const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config, const std::vector<bool>& are_required_outputs, std::optional<Tensor> input_grad ) {
        return _tanh_bw(queue_id, grad, input, output_mem_config, are_required_outputs, input_grad);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::SQRT_BW> {
    static std::vector<std::optional<Tensor>> handle( uint8_t queue_id, const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config, const std::vector<bool>& are_required_outputs, std::optional<Tensor> input_grad ) {
        return _sqrt_bw(queue_id, grad, input, output_mem_config, are_required_outputs, input_grad);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::GELU_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, string approximate, const std::optional<MemoryConfig>& output_mem_config ) {
        return _gelu_bw(grad, input, approximate, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::REPEAT_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, const tt::tt_metal::Shape& shape, const std::optional<MemoryConfig>& output_mem_config ) {
        return _repeat_bw(grad, input, shape, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::PROD_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, bool all_dimensions, int64_t dim, const std::optional<MemoryConfig>& output_mem_config ) {
        return _prod_bw(grad, input, all_dimensions, dim, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::ADD_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, float alpha, const std::optional<MemoryConfig>& output_mem_config ) {
        return _add_bw(grad, input, alpha, output_mem_config);
    }
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input_a, const Tensor& input_b, const std::optional<MemoryConfig>& output_mem_config ) {
        return ttnn::operations::binary_backward::_add_bw_inter(grad, input_a, input_b, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::MUL_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, float scalar, const std::optional<MemoryConfig>& output_mem_config ) {
        return _mul_bw(grad, input, scalar, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::EQ_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, float other, const std::optional<MemoryConfig>& output_mem_config ) {
        return _eq_bw(grad, input, other, output_mem_config);
    }
};

}  // namespace ttnn::operations::unary_backward
