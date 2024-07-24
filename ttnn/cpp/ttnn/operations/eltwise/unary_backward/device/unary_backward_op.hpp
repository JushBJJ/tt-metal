// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>
#include "ttnn/tensor/tensor.hpp"
#include "third_party/magic_enum/magic_enum.hpp"

namespace ttnn::operations::unary_backward {

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
    static std::function<std::vector<ttnn::Tensor>(const QueueId, const Tensor&, const Tensor&, const MemoryConfig&)> get_function_type1(UnaryBackwardOpType OpType);
    static std::function<std::vector<ttnn::Tensor>(const QueueId, const Tensor&, const Tensor&, float, const MemoryConfig&)> get_function_type1_w_float(UnaryBackwardOpType OpType);
};

//OpHandler_two_float : get_function_type1_w_two_float
std::vector<Tensor> _threshold_bw(const QueueId queue_id, const Tensor& grad, const Tensor& input, float threshold, float value, const std::optional<MemoryConfig>& output_mem_config);

//OpHandler_two_float_with_default : get_function_type1_w_two_float_with_default
std::vector<Tensor> _softplus_bw(const QueueId queue_id, const Tensor& grad, const Tensor& input, float beta = 1.0, float threshold = 20.0, const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
std::vector<Tensor> _hardtanh_bw(const QueueId queue_id, const Tensor& grad, const Tensor& input, float min = -1.0, float max = 1.0, const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

//OpHandler_optional_float_params_with_default : get_function_optional_float_params_with_default
std::vector<Tensor> _clamp_bw(const QueueId queue_id, const Tensor& grad, const Tensor& input, std::optional<float> min = std::nullopt, std::optional<float> max = std::nullopt, const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

//OpHandler_float_string_default : get_function_type1_float_string_default
std::vector<Tensor> _div_bw(const QueueId queue_id, const Tensor& grad, const Tensor& input, float scalar, string round_mode = "None", const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
std::vector<Tensor> _rdiv_bw(const QueueId queue_id, const Tensor& grad, const Tensor& input, float scalar, string round_mode = "None", const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
std::vector<Tensor> _bias_gelu_bw(const QueueId queue_id, const Tensor& grad, const Tensor& input, float bias, string approximate = "none", const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

//OpHandler_string_default : get_function_type1_string_default
std::vector<Tensor> _gelu_bw(const QueueId queue_id, const Tensor& grad, const Tensor& input, string approximate = "none", const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

//OpHandler_shape : get_function_type1_shape
std::vector<Tensor> _repeat_bw(const QueueId queue_id, const Tensor& grad, const Tensor& input, const tt::tt_metal::Shape& shape, const std::optional<MemoryConfig>& output_mem_config);

//OpHandler_unary_optional_float : get_function_unary_optional_float
std::vector<std::optional<Tensor>> _pow_bw(const QueueId queue_id, const Tensor& grad, const Tensor& input, float exponent, const MemoryConfig& output_mem_config , const std::vector<bool>& are_required_outputs, std::optional<Tensor> input_grad);

//OpHandler_unary_optional : get_function_unary_optional
std::vector<std::optional<Tensor>> _exp_bw(const QueueId queue_id, const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config, const std::vector<bool>& are_required_outputs, std::optional<Tensor> input_grad);
std::vector<std::optional<Tensor>> _tanh_bw(const QueueId queue_id, const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config, const std::vector<bool>& are_required_outputs, std::optional<Tensor> input_grad);
std::vector<std::optional<Tensor>> _sqrt_bw(const QueueId queue_id, const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config, const std::vector<bool>& are_required_outputs, std::optional<Tensor> input_grad);

//OpHandler_prod_bw : get_function_prod_bw
std::vector<Tensor> _prod_bw(const QueueId queue_id, const Tensor& grad, const Tensor& input, bool all_dimensions = true, int64_t dim = 0, const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
Tensor change_layout_to_tile(const Tensor& temp, const MemoryConfig& output_mem_config);

// OpHandler struct template
template <UnaryBackwardOpType OpType>
struct OpHandler_two_float;

template <UnaryBackwardOpType OpType>
struct OpHandler_two_float_with_default;

template <UnaryBackwardOpType OpType>
struct OpHandler_float_string_default;

template <UnaryBackwardOpType OpType>
struct OpHandler_string_default;

template <UnaryBackwardOpType OpType>
struct OpHandler_shape;

template <UnaryBackwardOpType OpType>
struct OpHandler_unary_optional_float;

template <UnaryBackwardOpType OpType>
struct OpHandler_unary_optional;

template <UnaryBackwardOpType OpType>
struct OpHandler_prod_bw;

template <UnaryBackwardOpType OpType>
struct OpHandler_optional_float_params_with_default;

template <>
struct OpHandler_optional_float_params_with_default<UnaryBackwardOpType::CLAMP_BW> {
    static std::vector<Tensor> handle(const QueueId queue_id, const Tensor& grad, const Tensor& input, std::optional<float> min, std::optional<float> max, const std::optional<MemoryConfig>& output_mem_config ) {
        return _clamp_bw(queue_id, grad, input, min, max, output_mem_config);
    }
};

template <>
struct OpHandler_two_float_with_default<UnaryBackwardOpType::HARDTANH_BW> {
    static std::vector<Tensor> handle(const QueueId queue_id, const Tensor& grad, const Tensor& input, float min, float max, const std::optional<MemoryConfig>& output_mem_config ) {
        return _hardtanh_bw(queue_id, grad, input, min, max, output_mem_config);
    }
};

template <>
struct OpHandler_two_float<UnaryBackwardOpType::THRESHOLD_BW> {
    static std::vector<Tensor> handle(const QueueId queue_id, const Tensor& grad, const Tensor& input, float threshold, float value, const std::optional<MemoryConfig>& output_mem_config ) {
        return _threshold_bw(queue_id, grad, input, threshold, value, output_mem_config);
    }
};

template <>
struct OpHandler_two_float_with_default<UnaryBackwardOpType::SOFTPLUS_BW> {
    static std::vector<Tensor> handle(const QueueId queue_id, const Tensor& grad, const Tensor& input, float beta, float threshold, const std::optional<MemoryConfig>& output_mem_config ) {
        return _softplus_bw(queue_id, grad, input, beta, threshold, output_mem_config);
    }
};

template <>
struct OpHandler_float_string_default<UnaryBackwardOpType::DIV_BW> {
    static std::vector<Tensor> handle(const QueueId queue_id, const Tensor& grad, const Tensor& input, float scalar, string round_mode, const std::optional<MemoryConfig>& output_mem_config ) {
        return _div_bw(queue_id, grad, input, scalar, round_mode, output_mem_config);
    }
};

template <>
struct OpHandler_float_string_default<UnaryBackwardOpType::RDIV_BW> {
    static std::vector<Tensor> handle(const QueueId queue_id, const Tensor& grad, const Tensor& input, float scalar, string round_mode, const std::optional<MemoryConfig>& output_mem_config ) {
        return _rdiv_bw(queue_id, grad, input, scalar, round_mode, output_mem_config);
    }
};

template <>
struct OpHandler_float_string_default<UnaryBackwardOpType::BIAS_GELU_BW> {
    static std::vector<Tensor> handle(const QueueId queue_id, const Tensor& grad, const Tensor& input, float bias, string approximate, const std::optional<MemoryConfig>& output_mem_config ) {
        return _bias_gelu_bw(queue_id, grad, input, bias, approximate, output_mem_config);
    }
};

template <>
struct OpHandler_unary_optional_float<UnaryBackwardOpType::POW_BW> {
    static std::vector<std::optional<Tensor>> handle(const QueueId queue_id, const Tensor& grad, const Tensor& input, float exponent, const MemoryConfig& output_mem_config, const std::vector<bool>& are_required_outputs, std::optional<Tensor> input_grad ) {
        return _pow_bw(queue_id, grad, input, exponent, output_mem_config, are_required_outputs, input_grad);
    }
};

template <>
struct OpHandler_unary_optional<UnaryBackwardOpType::EXP_BW> {
    static std::vector<std::optional<Tensor>> handle(const QueueId queue_id, const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config, const std::vector<bool>& are_required_outputs, std::optional<Tensor> input_grad ) {
        return _exp_bw(queue_id, grad, input, output_mem_config, are_required_outputs, input_grad);
    }
};

template <>
struct OpHandler_unary_optional<UnaryBackwardOpType::TANH_BW> {
    static std::vector<std::optional<Tensor>> handle(const QueueId queue_id, const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config, const std::vector<bool>& are_required_outputs, std::optional<Tensor> input_grad ) {
        return _tanh_bw(queue_id, grad, input, output_mem_config, are_required_outputs, input_grad);
    }
};

template <>
struct OpHandler_unary_optional<UnaryBackwardOpType::SQRT_BW> {
    static std::vector<std::optional<Tensor>> handle(const QueueId queue_id, const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config, const std::vector<bool>& are_required_outputs, std::optional<Tensor> input_grad ) {
        return _sqrt_bw(queue_id, grad, input, output_mem_config, are_required_outputs, input_grad);
    }
};

template <>
struct OpHandler_string_default<UnaryBackwardOpType::GELU_BW> {
    static std::vector<Tensor> handle(const QueueId queue_id, const Tensor& grad, const Tensor& input, string approximate, const std::optional<MemoryConfig>& output_mem_config ) {
        return _gelu_bw(queue_id, grad, input, approximate, output_mem_config);
    }
};

template <>
struct OpHandler_shape<UnaryBackwardOpType::REPEAT_BW> {
    static std::vector<Tensor> handle(const QueueId queue_id, const Tensor& grad, const Tensor& input, const tt::tt_metal::Shape& shape, const std::optional<MemoryConfig>& output_mem_config ) {
        return _repeat_bw(queue_id, grad, input, shape, output_mem_config);
    }
};

template <>
struct OpHandler_prod_bw<UnaryBackwardOpType::PROD_BW> {
    static std::vector<Tensor> handle(const QueueId queue_id, const Tensor& grad, const Tensor& input, bool all_dimensions, int64_t dim, const std::optional<MemoryConfig>& output_mem_config ) {
        return _prod_bw(queue_id, grad, input, all_dimensions, dim, output_mem_config);
    }
};

// Template functions to get the function pointers
template <UnaryBackwardOpType OpType>
auto get_function_type1_w_two_float() {
    return &OpHandler_two_float<OpType>::handle;
}

template <UnaryBackwardOpType OpType>
auto get_function_type1_w_two_float_with_default() {
    return &OpHandler_two_float_with_default<OpType>::handle;
}

template <UnaryBackwardOpType OpType>
auto get_function_type1_float_string_default() {
    return &OpHandler_float_string_default<OpType>::handle;
}

template <UnaryBackwardOpType OpType>
auto get_function_type1_string_default() {
    return &OpHandler_string_default<OpType>::handle;
}

template <UnaryBackwardOpType OpType>
auto get_function_type1_shape() {
    return &OpHandler_shape<OpType>::handle;
}

template <UnaryBackwardOpType OpType>
auto get_function_unary_optional_float() {
    return &OpHandler_unary_optional_float<OpType>::handle;
}

template <UnaryBackwardOpType OpType>
auto get_function_unary_optional() {
    return &OpHandler_unary_optional<OpType>::handle;
}

template <UnaryBackwardOpType OpType>
auto get_function_prod_bw() {
    return &OpHandler_prod_bw<OpType>::handle;
}

template <UnaryBackwardOpType OpType>
auto get_function_optional_float_params_with_default() {
    return &OpHandler_optional_float_params_with_default<OpType>::handle;
}

}  // namespace ttnn::operations::unary_backward
