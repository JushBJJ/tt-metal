// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/op_library/composite/composite_ops.hpp"

#include "ttnn/deprecated/tt_dnn/op_library/auto_format.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/concat/concat_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/copy/copy_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/math.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/optimizer/optimizer_ops.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/reduce/reduce_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/reshape/reshape_op.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "tt_numpy/functions.hpp"

#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/copy.hpp"
#include "ttnn/operations/matmul/matmul.hpp"

namespace tt {

namespace tt_metal {

Tensor mk_zero_tensor_like(
    uint8_t queue_id,
    const Tensor& reference_tensor,
    const MemoryConfig& output_mem_config,
    std::optional<Tensor> output_tensor = std::nullopt) {
    const DataType& dtype =
        output_tensor.has_value() ? output_tensor.value().get_dtype() : reference_tensor.get_dtype();
    Tensor zero = ttnn::operations::creation::create_scalar(0.0f, dtype, Layout::TILE, reference_tensor.device());
    return ttnn::multiply(queue_id, reference_tensor, zero, std::nullopt, output_mem_config, output_tensor);
}

Tensor mk_zero_tensor_like(
    const Tensor& reference_tensor,
    const MemoryConfig& output_mem_config,
    std::optional<Tensor> output_tensor = std::nullopt) {
    uint8_t default_queue_id = 0;
    return mk_zero_tensor_like(default_queue_id, reference_tensor, output_mem_config, output_tensor);
}

// TODO: enable zeroes(), ones() and eye() type functions on-device using this type of logic
template <typename T>
Tensor mk_filled_tensor_like(
    const Tensor& reference_tensor,
    T val,
    const MemoryConfig& output_mem_config,
    std::optional<Tensor> output_tensor = std::nullopt,
    uint8_t queue_id = 0) {
    const DataType& dtype =
        output_tensor.has_value() ? output_tensor.value().get_dtype() : reference_tensor.get_dtype();
    Tensor k = ttnn::operations::creation::create_scalar(val, dtype, Layout::TILE, reference_tensor.device());
    Tensor zero_like = mk_zero_tensor_like(reference_tensor, output_mem_config);
    if (output_tensor.has_value()) {
        return ttnn::add(queue_id, zero_like, k, std::nullopt, output_mem_config, output_tensor);
    } else {
        return ttnn::add(queue_id, zero_like, k, std::nullopt, output_mem_config);
    }
}

// Function: softshrink
// Ref: https://pytorch.org/docs/stable/generated/torch.nn.Softshrink.html
Tensor _softshrink(const Tensor& a, float param, const MemoryConfig& output_mem_config) {
    TT_ASSERT(param >= 0);
    Tensor t_a_plus_param = ttnn::add(a, param, std::nullopt, output_mem_config);
    Tensor t1 = ttnn::multiply(ttnn::ltz(t_a_plus_param, output_mem_config), t_a_plus_param, std::nullopt, output_mem_config);
    t_a_plus_param.deallocate();
    Tensor t_a_minus_param = ttnn::subtract(a, param, std::nullopt, output_mem_config);
    Tensor t2 =
        ttnn::multiply(ttnn::gtz(t_a_minus_param, output_mem_config), t_a_minus_param, std::nullopt, output_mem_config);
    t_a_minus_param.deallocate();
    return ttnn::add(t1, t2, std::nullopt, output_mem_config);
}
Tensor softshrink(const Tensor& a, float param, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _softshrink)(a, param, output_mem_config);
}

// Function: hardshrink
// Ref: https://pytorch.org/docs/stable/generated/torch.nn.Hardshrink.html
Tensor _hardshrink(const Tensor& a, float param, const MemoryConfig& output_mem_config) {
    TT_ASSERT(param >= 0);
    Tensor t1 = ttnn::multiply(ttnn::ltz(ttnn::add(a, param)), a, std::nullopt, output_mem_config);
    Tensor t2 = ttnn::multiply(ttnn::gtz(ttnn::subtract(a, param)), a, std::nullopt, output_mem_config);
    return ttnn::add(t1, t2, std::nullopt, output_mem_config);
}
Tensor hardshrink(const Tensor& a, float param, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _hardshrink)(a, param, output_mem_config);
}

// Function: bias gelu
// Ref: http://www.xavierdupre.fr/app/mlprodict/helpsphinx/onnxops/onnx_commicrosoft_BiasGelu.html
Tensor _bias_gelu_unary(const Tensor& a, float bias, const MemoryConfig& output_mem_config) {
    return ttnn::gelu(ttnn::add(a, bias), true, output_mem_config);
}
Tensor bias_gelu_unary(const Tensor& a, float bias, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _bias_gelu_unary)(a, bias, output_mem_config);
}

// Function: softsign
// Ref: https://pytorch.org/docs/stable/generated/torch.nn.Softsign.html
Tensor _softsign(const Tensor& a, const MemoryConfig& output_mem_config) {
    return ttnn::multiply(
        a,
        ttnn::reciprocal(ttnn::add(ttnn::abs(a, output_mem_config), 1.0f, std::nullopt, output_mem_config), output_mem_config),
        std::nullopt,
        output_mem_config);
}
Tensor softsign(const Tensor& a, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _softsign)(a, output_mem_config);
}

Tensor _swish(const Tensor& a, const MemoryConfig& output_mem_config) {
    // x / (1.0f + exp(-x))
    return ttnn::silu(a, output_mem_config);
}
Tensor swish(const Tensor& a, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _swish)(a, output_mem_config);
}


// tanhshrink(x) = x - tanh(x)
Tensor _tanhshrink(const Tensor& x, const MemoryConfig& output_mem_config) {
    Tensor tan_x = ttnn::tanh(x, output_mem_config);
    Tensor result = ttnn::subtract(x, tan_x, std::nullopt, output_mem_config);
    return result;
}
Tensor tanhshrink(const Tensor& a, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _tanhshrink)(a, output_mem_config);
}

// Theano defines this differently...
/**
 *
 *   alpha = 1.6732632423543772848170429916717
 *    scale = 1.0507009873554804934193349852946
 *    return scale * elu(x, alpha)
 *
 */
// Function Selu - scaled exponential linear
// use transformation y = scale *(max(0,x)) + min(0,alpha * (exp(X)-1)) by broadcast
// Ref: https://pytorch.org/docs/stable/generated/torch.nn.SELU.html
Tensor _selu(const Tensor& x, const float scale, const float alpha, const MemoryConfig& output_mem_config) {
    // term 2
    Tensor x_Exp = ttnn::exp(x, false, output_mem_config);
    Tensor minus_one = ttnn::operations::creation::create_scalar(-1.0f, x.get_dtype(), Layout::TILE, x.device());
    Tensor x_Exp_minus_1 = ttnn::add(x_Exp, minus_one,std::nullopt, output_mem_config);
    x_Exp.deallocate();
    minus_one.deallocate();
    Tensor t_alpha = ttnn::operations::creation::create_scalar(alpha, x.get_dtype(), Layout::TILE, x.device());
    Tensor result_t2_ = ttnn::multiply(x_Exp_minus_1, t_alpha, std::nullopt, output_mem_config);
    x_Exp_minus_1.deallocate();
    t_alpha.deallocate();
    Tensor result_term2 =
        ttnn::multiply(ttnn::gtz(result_t2_, output_mem_config), result_t2_, std::nullopt, output_mem_config);
    result_t2_.deallocate();

    // term 1
    Tensor t_scale = ttnn::operations::creation::create_scalar(scale, x.get_dtype(), Layout::TILE, x.device());
    Tensor x_relu = ttnn::relu(x, output_mem_config);
    Tensor result_term1 = ttnn::multiply(x_relu, t_scale, std::nullopt, output_mem_config);
    t_scale.deallocate();
    x_relu.deallocate();
    Tensor result_selu = ttnn::add(result_term1, result_term2, std::nullopt, output_mem_config);

    return result_selu;
}
Tensor selu(const Tensor& x, const float scale, const float alpha, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _selu)(x, scale, alpha, output_mem_config);
}

// ELU :
//  Theano defines it as,
//  return tensor.switch(x > 0, x, alpha * tensor.expm1(x))

// rpow: y = k**(a) = exp( a**log(k) )
Tensor rpow(const Tensor& a, float k, const MemoryConfig& output_mem_config) {
    TT_ASSERT(k > 0.0, "rpow cannot be calcualted for non-positive numbers");
    float log_k = logf(k);

    Tensor scalar = ttnn::operations::creation::create_scalar(log_k, a.get_dtype(), Layout::TILE, a.device());
    Tensor result = ttnn::multiply(a, scalar, std::nullopt, output_mem_config);
    scalar.deallocate();
    return ttnn::exp(result, false, output_mem_config);
}

// Function Clip
// use clip y = min( max( x, min_value), max_value) by broadcast
// Ref: https://pytorch.org/docs/stable/generated/torch.clamp.html#torch.clamp
Tensor _clip(const Tensor& a, float low, float high, const MemoryConfig& output_mem_config) {
    const Tensor h_const = full_like(a, high);
    Tensor a_max = tt::tt_metal::min(a, h_const, output_mem_config);
    if (low == 0.0f) {
        return ttnn::relu(a_max, output_mem_config);
    } else {
        const Tensor l_const = full_like(a, low);
        return tt::tt_metal::max(a_max, l_const, output_mem_config);
    }
}
Tensor clip(const Tensor& a, float low, float high, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _clip)(a, low, high, output_mem_config);
}

// compute polyval by Horner's rule
Tensor _polyval(const Tensor& input_tensor, std::vector<float> coeffs, const MemoryConfig& output_mem_config) {
    TT_ASSERT(coeffs.size() != 0 && "coeffs should be 1 or more coefficients");
    if (coeffs.size() == 1) {
        return mk_filled_tensor_like(input_tensor, coeffs[0], output_mem_config);
    }

    Tensor scalar = ttnn::operations::creation::create_scalar(
        coeffs[0], input_tensor.get_dtype(), Layout::TILE, input_tensor.device());
    Tensor result = ttnn::multiply(input_tensor, scalar, std::nullopt, output_mem_config);
    scalar.deallocate();
    for (int idx = 1; idx < coeffs.size() - 1; idx++) {
        Tensor scalar = ttnn::operations::creation::create_scalar(
            coeffs[idx], input_tensor.get_dtype(), Layout::TILE, input_tensor.device());
        result = ttnn::add(result, scalar, std::nullopt, output_mem_config);
        scalar.deallocate();
        result = ttnn::multiply(input_tensor, result, std::nullopt, output_mem_config);
    }
    Tensor last_coeffs = ttnn::operations::creation::create_scalar(
        coeffs.back(), input_tensor.get_dtype(), Layout::TILE, input_tensor.device());
    Tensor final_tensor = ttnn::add(result, last_coeffs, std::nullopt, output_mem_config);
    last_coeffs.deallocate();
    return final_tensor;
}
Tensor polyval(const Tensor& input_tensor, std::vector<float> coeffs, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _polyval)(input_tensor, coeffs, output_mem_config);
}

// Function: MAC
// compute multiply-accumulate: y = a * b + c,  over various 8 combinations of a, b, c
// being a scalar or tensor
Tensor _mac(const Tensor& a, const Tensor& b, const Tensor& c, const MemoryConfig& output_mem_config) {
    bool a_is_scalar = a.intended_volume() == 1;
    bool b_is_scalar = b.intended_volume() == 1;
    bool c_is_scalar = c.intended_volume() == 1;

    if (!a_is_scalar && !b_is_scalar && !c_is_scalar) {
        // all tensors
        return ttnn::add(ttnn::multiply(a, b, std::nullopt, output_mem_config), c, std::nullopt, output_mem_config);
    } else if (!a_is_scalar && !b_is_scalar && c_is_scalar) {
        // a - tensor, b - tensor, c - is scalar
        return ttnn::add(
            ttnn::multiply(a, b, std::nullopt, output_mem_config), c, std::nullopt, output_mem_config);
    } else if (!a_is_scalar && b_is_scalar && !c_is_scalar) {
        // a - tensor, b - scalar, c - is tensor
        return ttnn::add(ttnn::multiply(a, b, std::nullopt, output_mem_config), c, std::nullopt, output_mem_config);
    } else if (!a_is_scalar && b_is_scalar && c_is_scalar) {
        // a - tensor, b - scalar, c - is scalar
        return ttnn::add(
            ttnn::multiply(a, b, std::nullopt, output_mem_config), c, std::nullopt, output_mem_config);
    } else if (a_is_scalar && !b_is_scalar && !c_is_scalar) {
        // a - scalar, b - tensor, c - tensor
        return ttnn::add(ttnn::multiply(b, a, std::nullopt, output_mem_config), c, std::nullopt, output_mem_config);
    } else if (a_is_scalar && !b_is_scalar && c_is_scalar) {
        // a - scalar, b - tensor, c - is scalar
        return ttnn::add(
            ttnn::multiply(b, a, std::nullopt, output_mem_config), c, std::nullopt, output_mem_config);
    } else if (a_is_scalar && b_is_scalar && !c_is_scalar) {
        // a - scalar, b - scalar, c - is tensor
        return ttnn::add(
            c, ttnn::multiply(a, b, std::nullopt, output_mem_config), std::nullopt, output_mem_config);
    }

    // all scalars
    // a - scalar, b - scalar, c - is scalar
    TT_ASSERT(a_is_scalar && b_is_scalar && c_is_scalar);
    return ttnn::add(ttnn::multiply(a, b), c);
}
Tensor mac(const Tensor& a, const Tensor& b, const Tensor& c, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _mac)(a, b, c, output_mem_config);
}

Tensor _mac_overload(const Tensor& a, float b, float c, const MemoryConfig& output_mem_config) {
    Tensor t_b = ttnn::operations::creation::create_scalar(b, a.get_dtype(), Layout::TILE, a.device());
    Tensor t_c = ttnn::operations::creation::create_scalar(c, a.get_dtype(), Layout::TILE, a.device());
    Tensor return_tensor = mac(a, t_b, t_c, output_mem_config);
    t_b.deallocate();
    t_c.deallocate();
    return return_tensor;
}
Tensor mac(const Tensor& input_a, float b, float c, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _mac_overload)(input_a, b, c, output_mem_config);
}

// min(a,b) = a - (a - b > 0 )*(a-b)
Tensor _min(const Tensor& input_a, const Tensor& input_b, const MemoryConfig& output_mem_config) {
    Tensor t_diff = ttnn::subtract(input_a, input_b, std::nullopt, output_mem_config);
    Tensor result = where(t_diff, input_b, input_a, output_mem_config);
    return result;
}
Tensor min(const Tensor& input_a, const Tensor& input_b, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _min)(input_a, input_b, output_mem_config);
}

// max(a,b) = a + (b - a > 0 )*(b-a)
Tensor _max(const Tensor& input_a, const Tensor& input_b, const MemoryConfig& output_mem_config) {
    Tensor t_diff = ttnn::subtract(input_b, input_a, std::nullopt, output_mem_config);
    Tensor result = where(t_diff, input_b, input_a, output_mem_config);
    return result;
}
Tensor max(const Tensor& input_a, const Tensor& input_b, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _max)(input_a, input_b, output_mem_config);
}

Tensor _logical_andi(const Tensor& input_a, float immediate, const MemoryConfig& output_mem_config) {
    if (std::fpclassify(immediate) == FP_ZERO) {
        return full_like(input_a, immediate, output_mem_config);
    } else {
        return ttnn::nez(input_a);
    }
}
Tensor logical_andi(const Tensor& input_a, float immediate, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _logical_andi)(input_a, immediate, output_mem_config);
}

// sinh[x] = (exp[x] - exp[-x])/2
Tensor _sinh(const Tensor& input_a, const MemoryConfig& output_mem_config) {
    Tensor e_pos_x = ttnn::exp(input_a, false, output_mem_config);
    Tensor e_neg_x = ttnn::exp(ttnn::neg(input_a, output_mem_config), false, output_mem_config);
    Tensor nr_term = ttnn::subtract(e_pos_x, e_neg_x, std::nullopt, output_mem_config);
    e_pos_x.deallocate();
    e_neg_x.deallocate();
    Tensor scalar =
        ttnn::operations::creation::create_scalar(0.5f, input_a.get_dtype(), Layout::TILE, input_a.device());
    Tensor result = ttnn::multiply(nr_term, scalar, std::nullopt, output_mem_config);
    scalar.deallocate();
    return result;
}
Tensor sinh(const Tensor& input_a, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _sinh)(input_a, output_mem_config);
}

// cosh[x] = (exp[x] + exp[-x])/2
Tensor _cosh(const Tensor& input_a, const MemoryConfig& output_mem_config) {
    Tensor e_pos_x = ttnn::exp(input_a, false, output_mem_config);
    Tensor e_neg_x = ttnn::exp(ttnn::neg(input_a, output_mem_config), false, output_mem_config);
    Tensor nr_term = ttnn::add(e_pos_x, e_neg_x, std::nullopt, output_mem_config);
    e_pos_x.deallocate();
    e_neg_x.deallocate();
    Tensor scalar =
        ttnn::operations::creation::create_scalar(0.5f, input_a.get_dtype(), Layout::TILE, input_a.device());
    Tensor result = ttnn::multiply(nr_term, scalar, std::nullopt, output_mem_config);
    scalar.deallocate();
    return result;
}
Tensor cosh(const Tensor& input_a, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _cosh)(input_a, output_mem_config);
}

// asinh(x) = log(x + sqrt(x^2 + 1))
Tensor _asinh(const Tensor& input_a, const MemoryConfig& output_mem_config) {
    Tensor ln_res(input_a);
    {
        Tensor x_abs = ttnn::abs(input_a, output_mem_config);
        Tensor x_sq_p1(input_a);
        {
            Tensor x_sq = ttnn::square(input_a, output_mem_config);
            x_sq_p1 = ttnn::add(x_sq, 1.0f, std::nullopt, output_mem_config);
        }
        ln_res =
            ttnn::log(ttnn::add(x_abs, ttnn::sqrt(x_sq_p1, output_mem_config), std::nullopt, output_mem_config), output_mem_config);
    }
    // input is negative, output is -asinh(input)
    Tensor result = where(input_a, ln_res, ttnn::neg(ln_res, output_mem_config), output_mem_config);
    return result;
}
Tensor asinh(const Tensor& input_a, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _asinh)(input_a, output_mem_config);
}

// acosh(x) = log(x + sqrt(x^2 - 1))
Tensor _acosh(const Tensor& input_a, const MemoryConfig& output_mem_config) {
    Tensor t_one = ones_like(input_a, output_mem_config);
    Tensor t_result(input_a);
    {
        Tensor ln_res(input_a);
        {
            Tensor x_abs = ttnn::abs(input_a, output_mem_config);
            Tensor x_sq_m1(input_a);
            {
                Tensor x_sq = ttnn::square(x_abs, output_mem_config);
                x_sq_m1 = ttnn::subtract(x_sq, 1.0f, std::nullopt, output_mem_config);
            }
            ln_res = ttnn::log(
                ttnn::add(x_abs, ttnn::sqrt(x_sq_m1, output_mem_config), std::nullopt, output_mem_config), output_mem_config);
        }
        // To handle inputs <= 1
        // input < 1, output is nan
        // input > 1, output is acosh(input)
        Tensor scalar = ttnn::operations::creation::create_scalar(
            std::nanf(""), input_a.get_dtype(), Layout::TILE, input_a.device());
        Tensor nan_res = ttnn::multiply(
            ttnn::le(input_a, t_one, std::nullopt, output_mem_config), scalar, std::nullopt, output_mem_config);
        scalar.deallocate();
        t_result = ttnn::multiply(
            ttnn::gt(input_a, t_one, std::nullopt, output_mem_config), ln_res, std::nullopt, output_mem_config);
        t_result = ttnn::add(nan_res, t_result, std::nullopt, output_mem_config);
    }
    // input == 1, output is 0
    Tensor result = where(ttnn::eq(input_a, t_one, std::nullopt, output_mem_config), 0.0f, t_result, output_mem_config);
    return result;
}
Tensor acosh(const Tensor& input_a, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _acosh)(input_a, output_mem_config);
}

// atanh[x] = 0.5 * ln((1 + x) / (1 - x))
Tensor _atanh(const Tensor& input_a, const MemoryConfig& output_mem_config) {
    Tensor comp_result(input_a);
    {
        Tensor nr_term(input_a);
        {
            Tensor pos_x = ttnn::add(input_a, 1.0f, std::nullopt, output_mem_config);
            Tensor neg_x = ttnn::subtract(input_a, 1.0f, std::nullopt, output_mem_config);
            nr_term = ttnn::log(
                ttnn::multiply(
                    pos_x, ttnn::reciprocal(ttnn::neg(neg_x, output_mem_config), output_mem_config), std::nullopt, output_mem_config),
                output_mem_config);
        }
        comp_result = ttnn::multiply(nr_term, 0.5f, std::nullopt, output_mem_config);
    }
    // Input is -1 > value > 1, output is nan
    // Input is -1 < value < 1, output is atanh(input)
    float t_nan = std::nanf("");
    Tensor abs_temp = ttnn::subtract(ttnn::abs(input_a, output_mem_config), 1.0f, std::nullopt, output_mem_config);
    Tensor result = where(ttnn::ltz(abs_temp, output_mem_config), comp_result, t_nan, output_mem_config);
    return result;
}
Tensor atanh(const Tensor& input_a, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _atanh)(input_a, output_mem_config);
}

// lerp(input, end, weight) = start + weight * (end - start)
Tensor _lerp(const Tensor& input_a, const Tensor& input_b, float value, const MemoryConfig& output_mem_config) {
    Tensor t_value =
        ttnn::operations::creation::create_scalar(value, input_a.get_dtype(), Layout::TILE, input_a.device());
    Tensor t_diff = ttnn::subtract(input_b, input_a, std::nullopt, output_mem_config);
    Tensor t_mul = ttnn::multiply(t_diff, t_value, std::nullopt, output_mem_config);
    Tensor result = ttnn::add(input_a, t_mul, std::nullopt, output_mem_config);
    return result;
}
Tensor lerp(const Tensor& input_a, const Tensor& input_b, float value, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _lerp)(input_a, input_b, value, output_mem_config);
}

Tensor _atan2(const Tensor& input_a, const Tensor& input_b, const MemoryConfig& output_mem_config) {
    Tensor result(input_a);
    {
        Tensor atan_input = ttnn::multiply(
            ttnn::abs(input_b, output_mem_config),
            ttnn::reciprocal(ttnn::abs(input_a, output_mem_config), output_mem_config),
            std::nullopt,
            output_mem_config);
        result = ttnn::atan(atan_input, output_mem_config);
    }
    Tensor res(result);
    {
        Tensor ib_gtz = ttnn::gtz(input_b, output_mem_config);
        Tensor ib_gt = ttnn::gtz(input_b, output_mem_config);
        Tensor ib_lt = ttnn::ltz(input_b, output_mem_config);
        float pi_2 = M_PI_2;
        Tensor neg_result = ttnn::neg(result, output_mem_config);

        res = where(
            ttnn::gtz(input_a, output_mem_config),
            where(ib_gtz, result, neg_result, output_mem_config),
            where(
                ttnn::ltz(input_a, output_mem_config),
                where(
                    ib_gt,
                    ttnn::add(neg_result, M_PI, std::nullopt, output_mem_config),
                    where(ib_lt, ttnn::subtract(result, M_PI, std::nullopt, output_mem_config), M_PI, output_mem_config),
                    output_mem_config),
                where(ib_gt, pi_2, where(ib_lt, -pi_2, 0.0f, output_mem_config), output_mem_config),
                output_mem_config));
    }
    return res;
}
Tensor atan2(const Tensor& input_a, const Tensor& input_b, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _atan2)(input_a, input_b, output_mem_config);
}

// lerp(input, end, weight) = start + weight * (end - start)
Tensor _lerp_overload(
    const Tensor& input_a, const Tensor& input_b, const Tensor& input_c, const MemoryConfig& output_mem_config) {
    Tensor t_diff = ttnn::multiply(
        ttnn::subtract(input_b, input_a, std::nullopt, output_mem_config), input_c, std::nullopt, output_mem_config);
    Tensor result = ttnn::add(input_a, t_diff, std::nullopt, output_mem_config);
    return result;
}
Tensor lerp(
    const Tensor& input_a, const Tensor& input_b, const Tensor& input_c, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _lerp_overload)(input_a, input_b, input_c, output_mem_config);
}

// ldexp(input,other)=input * (2^other)
Tensor _ldexp(const Tensor& input_a, const Tensor& input_b, const MemoryConfig& output_mem_config) {
    Tensor result = ttnn::multiply(input_a, ttnn::exp2(input_b, output_mem_config), std::nullopt, output_mem_config);
    return result;
}

Tensor _logical_ori(const Tensor& input_a, float immediate, const MemoryConfig& output_mem_config) {
    if (std::fpclassify(immediate) == FP_ZERO) {
        return ttnn::nez(input_a, output_mem_config);
    } else {
        return full_like(input_a, 1, output_mem_config);
    }
}
Tensor logical_ori(const Tensor& input_a, float immediate, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _logical_ori)(input_a, immediate, output_mem_config);
}

Tensor _logical_noti(const Tensor& input_a, float immediate, const MemoryConfig& output_mem_config) {
    Tensor t_imm = full_like(input_a, immediate, output_mem_config);
    Tensor result = ttnn::logical_not(t_imm, output_mem_config);
    return result;
}
Tensor logical_noti(const Tensor& input_a, float immediate, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _logical_noti)(input_a, immediate, output_mem_config);
}

// subalpha(input,other,alpha)=input-alpha*other
Tensor _subalpha(const Tensor& input_a, const Tensor& input_b, float alpha, const MemoryConfig& output_mem_config) {
    Tensor result = ttnn::add(
        ttnn::neg(ttnn::multiply(input_b, alpha, std::nullopt, output_mem_config), output_mem_config), input_a, std::nullopt, output_mem_config);
    return result;
}
Tensor subalpha(const Tensor& input_a, const Tensor& input_b, float alpha, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _subalpha)(input_a, input_b, alpha, output_mem_config);
}

// addalpha(input, other, alpha) = input + (alpha * other)
Tensor _addalpha(
    uint8_t cq_id,
    const Tensor& input_a,
    const Tensor& input_b,
    float alpha,
    const MemoryConfig& output_mem_config,
    std::optional<Tensor> output_tensor) {
    if (output_tensor.has_value()) {
        ttnn::add(cq_id, ttnn::multiply(cq_id, input_b, alpha, std::nullopt, output_mem_config), input_a, std::nullopt, std::nullopt, output_tensor);
        return output_tensor.value();
    }

    return ttnn::add(cq_id, ttnn::multiply(cq_id, input_b, alpha, std::nullopt, output_mem_config), input_a, std::nullopt, output_mem_config);
}

Tensor addalpha(
    const Tensor& input_a,
    const Tensor& input_b,
    float alpha,
    const MemoryConfig& output_mem_config,
    std::optional<Tensor> output_tensor) {
    uint8_t default_queue_id = 0;
    return operation::decorate_as_composite(__func__, _addalpha)(
        default_queue_id, input_a, input_b, alpha, output_mem_config, output_tensor);
}

Tensor addalpha(
    uint8_t cq_id,
    const Tensor& input_a,
    const Tensor& input_b,
    float alpha,
    const MemoryConfig& output_mem_config,
    std::optional<Tensor> output_tensor) {
    return operation::decorate_as_composite(__func__, _addalpha)(
        cq_id, input_a, input_b, alpha, output_mem_config, output_tensor);
}


// addcmul(input,tensor1,tensor2,value)=input+value×tensor1×tensor2
Tensor _addcmul(
    const Tensor& input_a,
    const Tensor& input_b,
    const Tensor& input_c,
    float value,
    const MemoryConfig& output_mem_config) {
    Tensor t_value =
        ttnn::operations::creation::create_scalar(value, input_a.get_dtype(), Layout::TILE, input_a.device());
    Tensor t_mul = ttnn::multiply(input_b, input_c, std::nullopt, output_mem_config);
    Tensor t_factor = ttnn::multiply(t_mul, t_value, std::nullopt, output_mem_config);
    t_mul.deallocate();
    t_value.deallocate();
    Tensor result = ttnn::add(input_a, t_factor, std::nullopt, output_mem_config);
    return result;
}
Tensor addcmul(
    const Tensor& input_a,
    const Tensor& input_b,
    const Tensor& input_c,
    float value,
    const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _addcmul)(input_a, input_b, input_c, value, output_mem_config);
}

// addcdiv(input,tensor1,tensor2,value)=input+value×tensor1/tensor2
Tensor _addcdiv(
    const Tensor& input_a,
    const Tensor& input_b,
    const Tensor& input_c,
    float value,
    const MemoryConfig& output_mem_config) {
    Tensor t_value =
        ttnn::operations::creation::create_scalar(value, input_a.get_dtype(), Layout::TILE, input_a.device());
    Tensor t_div = ttnn::multiply(input_b, ttnn::reciprocal(input_c, output_mem_config), std::nullopt, output_mem_config);
    Tensor t_factor = ttnn::multiply(t_div, t_value, std::nullopt, output_mem_config);
    t_div.deallocate();
    t_value.deallocate();
    Tensor result = ttnn::add(input_a, t_factor, std::nullopt, output_mem_config);
    Tensor t_inf = full_like(input_a, std::numeric_limits<float>::infinity(), output_mem_config);
    Tensor t_nan = full_like(input_a, std::nanf(""), output_mem_config);
    return where(
        ttnn::eqz(input_c, output_mem_config),
        (value == 0) ? t_nan
                     : where(
                           ttnn::eqz(input_b, output_mem_config),
                           t_nan,
                           ttnn::multiply(t_inf, ttnn::sign(input_b, output_mem_config), std::nullopt, output_mem_config),
                           output_mem_config),
        result,
        output_mem_config);
}
Tensor addcdiv(
    const Tensor& input_a,
    const Tensor& input_b,
    const Tensor& input_c,
    float value,
    const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _addcdiv)(input_a, input_b, input_c, value, output_mem_config);
}

Tensor _div(const Tensor& input_a, const Tensor& input_b, bool accurate_mode, string round_mode,  const MemoryConfig& output_mem_config) {
    TT_FATAL((round_mode == "None" || round_mode == "trunc" || round_mode == "floor") && "Incorrect rounding mode (expected 'None', 'trunc', or 'floor')");
    Tensor result = ttnn::divide(input_a, input_b);
    if(round_mode == "trunc"){
        result = trunc(result);
    }
    else if(round_mode == "floor"){
        result = ttnn::floor(result);
    }

    if (accurate_mode == false) {  // If input_b is non-zero tensor
        return result;
    }

    Tensor t_inf = full_like(input_a, std::numeric_limits<float>::infinity(), output_mem_config);
    Tensor t_nan = full_like(input_a, std::nanf(""), output_mem_config);
    return where(
        ttnn::eqz(input_b, output_mem_config),
        where(
            ttnn::eqz(input_a, output_mem_config),
            t_nan,
            ttnn::multiply(t_inf, ttnn::sign(input_a, output_mem_config), std::nullopt, output_mem_config),
            output_mem_config),
        result,
        output_mem_config);
}
Tensor div(const Tensor& input_a, const Tensor& input_b, bool accurate_mode, string round_mode, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _div)(input_a, input_b, accurate_mode, round_mode, output_mem_config);
}

Tensor _div_overload(const Tensor& input_a, float scalar, bool accurate_mode, string round_mode,  const MemoryConfig& output_mem_config) {
    TT_FATAL((round_mode == "None" || round_mode == "trunc" || round_mode == "floor") && "Incorrect rounding mode (expected 'None', 'trunc', or 'floor')");
    Tensor result = ttnn::multiply(input_a, (1.0f/scalar));

    if(round_mode == "trunc"){
        result = trunc(result);
    }
    else if(round_mode == "floor"){
        result = ttnn::floor(result);
    }

    return result;
}
Tensor div(const Tensor& input_a, float scalar, bool accurate_mode, string round_mode, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _div_overload)(input_a, scalar, accurate_mode, round_mode, output_mem_config);
}

Tensor _trunc(const Tensor& input, const MemoryConfig& output_mem_config) {
    auto arch = input.device()->arch();
    TT_FATAL(arch == tt::ARCH::WORMHOLE_B0, "Op is only supported on Wormhole");
    Tensor floor_res = ttnn::floor(input, output_mem_config);
    Tensor trunc_res = where(ttnn::ne(input, floor_res), ttnn::add(floor_res, 1.0f), floor_res, output_mem_config);
    Tensor result = where(ttnn::gtz(input, output_mem_config), floor_res, trunc_res, output_mem_config);
    return result;
}
Tensor trunc(const Tensor& input, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _trunc)(input, output_mem_config);
}

Tensor _frac(const Tensor& input, const MemoryConfig& output_mem_config) {
    auto arch = input.device()->arch();
    TT_FATAL(arch == tt::ARCH::WORMHOLE_B0, "Op is only supported on Wormhole");
    Tensor trunc_res = trunc(input, output_mem_config);
    Tensor result = ttnn::subtract(input, trunc_res, std::nullopt, output_mem_config);
    return result;
}
Tensor frac(const Tensor& input, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _frac)(input, output_mem_config);
}

Tensor _div_trunc(
    const Tensor& input_a,
    const Tensor& input_b,
    const MemoryConfig& output_mem_config) {
    auto arch = input_a.device()->arch();
    TT_FATAL(arch == tt::ARCH::WORMHOLE_B0, "Op is only supported on Wormhole");
    Tensor result = div(input_a, input_b, true);
    return trunc(result);
}
Tensor div_trunc(
    const Tensor& input_a,
    const Tensor& input_b,
    const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _div_trunc)(input_a, input_b, output_mem_config);
}

Tensor _div_trunc_overload(
    const Tensor& input,
    float value,
    const MemoryConfig& output_mem_config) {
    auto arch = input.device()->arch();
    TT_FATAL(arch == tt::ARCH::WORMHOLE_B0, "Op is only supported on Wormhole");
    Tensor result = ttnn::multiply(input, (1 / value));
    return trunc(result);
}
Tensor div_trunc(
    const Tensor& input,
    float value,
    const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _div_trunc_overload)(input, value, output_mem_config);
}

Tensor _unary_rdiv_trunc(
    float value,
    const Tensor& input,
    const MemoryConfig& output_mem_config) {
    auto arch = input.device()->arch();
    TT_FATAL(arch == tt::ARCH::WORMHOLE_B0, "Op is only supported on Wormhole");
    Tensor result = ttnn::multiply(ttnn::full_like(input, value), ttnn::reciprocal(input));
    return trunc(result);
}
Tensor unary_rdiv_trunc(
    float value,
    const Tensor& input,
    const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _unary_rdiv_trunc)(value, input, output_mem_config);
}

Tensor is_odd(const Tensor& input, const MemoryConfig& output_mem_config) {
    Tensor result = ttnn::multiply(input, (1.0f/2.0f));
    Tensor floor_res = ttnn::floor(result);
    return ttnn::ne(result, floor_res);
}

Tensor _round(const Tensor& input, int64_t decimals, const MemoryConfig& output_mem_config) {
    auto arch = input.device()->arch();
    TT_FATAL(arch == tt::ARCH::WORMHOLE_B0, "Op is only supported on Wormhole");
    Tensor floor_res = ttnn::floor(input, output_mem_config);
    if (decimals != 0) {  // TODO: For decimal value!=0
        Tensor power_10 =
            pow(full_like(input, 10.0f, output_mem_config), static_cast<float>(decimals), output_mem_config);
        Tensor rounded_non_half = ttnn::floor(
            ttnn::add(ttnn::multiply(input, power_10, std::nullopt, output_mem_config), 0.5, std::nullopt, output_mem_config),
            output_mem_config);
        rounded_non_half = div(rounded_non_half, power_10);
        return rounded_non_half;
    } else {  // Bankers' Rounding
        Tensor rounded_non_half = ttnn::floor(
            ttnn::add(
                input,
                where(ttnn::logical_and(ttnn::ge(input, 0.4), ttnn::le(input, 0.5)), 0.4f, 0.5f, output_mem_config),
                std::nullopt,
                output_mem_config),
            output_mem_config);
        Tensor fractional_part = ttnn::subtract(input, floor_res, std::nullopt, output_mem_config);
        Tensor is_half = ttnn::eq(fractional_part, 0.5);
        Tensor rounded_half =
            ttnn::add(floor_res, tt::tt_metal::is_odd(floor_res, output_mem_config), std::nullopt, output_mem_config);
        return where(is_half, rounded_half, rounded_non_half, output_mem_config);
    }
}

Tensor round(const Tensor& input, int64_t decimals, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _round)(input, decimals, output_mem_config);
}

Tensor _floor_div(const Tensor& input_a, const Tensor& input_b, const MemoryConfig& output_mem_config) {
    auto arch = input_a.device()->arch();
    TT_FATAL(arch == tt::ARCH::WORMHOLE_B0, "Op is only supported on Wormhole");
    Tensor temp = div(input_a, input_b, true);
    // floor(nan, inf, -inf) = nan, inf, -inf
    return where(
        ttnn::logical_or(
            ttnn::eq(temp, std::nanf("")),
            ttnn::logical_or(
                ttnn::eq(temp, std::numeric_limits<float>::infinity()),
                ttnn::eq(temp, -std::numeric_limits<float>::infinity()))),
        temp,
        ttnn::floor(temp, output_mem_config));
}
Tensor floor_div(const Tensor& input_a, const Tensor& input_b, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _floor_div)(input_a, input_b, output_mem_config);
}

Tensor _floor_div_overload(const Tensor& input, float value, const MemoryConfig& output_mem_config) {
    if (value == 0) {
        Tensor t_inf = full_like(input, std::numeric_limits<float>::infinity(), output_mem_config);
        Tensor t_nan = full_like(input, std::nanf(""), output_mem_config);
        return where(
            ttnn::eqz(input, output_mem_config),
            t_nan,
            ttnn::multiply(t_inf, ttnn::sign(input, output_mem_config), std::nullopt, output_mem_config),
            output_mem_config);

    }
    Tensor temp = ttnn::multiply(input, (1.0f/value));
    return ttnn::floor(temp);
}
Tensor floor_div(const Tensor& input_a, float value, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _floor_div_overload)(input_a, value, output_mem_config);
}

Tensor _rfloor_div(float value, const Tensor& input, const MemoryConfig& output_mem_config) {
    Tensor result = ttnn::multiply(ttnn::full_like(input, value), ttnn::reciprocal(input));
    return ttnn::floor(result, output_mem_config);
}
Tensor rfloor_div(float value, const Tensor& input, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _rfloor_div)(value, input, output_mem_config);
}

Tensor _div_no_nan(const Tensor& input_a, const Tensor& input_b, const MemoryConfig& output_mem_config) {
    Tensor div_result = div(input_a, input_b);
    return where(ttnn::eqz(input_b, output_mem_config), 0, div_result);
}
Tensor div_no_nan(const Tensor& input_a, const Tensor& input_b, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _div_no_nan)(input_a, input_b, output_mem_config);
}

Tensor _div_no_nan_overload(const Tensor& input_a, float value, const MemoryConfig& output_mem_config) {
    if (value == 0)
        return full_like(input_a, 0.0f, output_mem_config);
    else
        return ttnn::multiply(input_a, (1.0f/value));
}
Tensor div_no_nan(const Tensor& input_a, float value, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _div_no_nan_overload)(input_a, value, output_mem_config);
}

Tensor _remainder(const Tensor& input_a, const Tensor& input_b, const MemoryConfig& output_mem_config) {
    DataType input_dtype = input_a.get_dtype();
    Tensor a = ttnn::typecast(input_a, DataType::FLOAT32);
    Tensor b = ttnn::typecast(input_b, DataType::FLOAT32);
    Tensor result = ttnn::subtract(a, ttnn::multiply(b, floor_div(input_a, input_b, output_mem_config), std::nullopt, output_mem_config), std::nullopt, output_mem_config);
    result = where(ttnn::ge(result, b), ttnn::subtract(result, b), result);
    result = where(ttnn::ltz(b), ttnn::add(result, b), result);
    result = where(ttnn::eq(a, b, std::nullopt, output_mem_config), full_like(input_a, 0.0f, output_mem_config), result, output_mem_config);
    return ttnn::typecast(result, input_dtype);
}
Tensor remainder(const Tensor& input_a, const Tensor& input_b, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _remainder)(input_a, input_b, output_mem_config);
}

Tensor _fmod(const Tensor& input_a, const Tensor& input_b, const MemoryConfig& output_mem_config) {
    DataType input_dtype = input_a.get_dtype();
    Tensor a = ttnn::typecast(input_a, DataType::FLOAT32);
    Tensor b = ttnn::typecast(input_b, DataType::FLOAT32);
    Tensor result = ttnn::subtract(a, ttnn::multiply(div(input_a, input_b, true, "trunc", output_mem_config), b, std::nullopt, output_mem_config), std::nullopt, output_mem_config);
    result = where(ttnn::eq(a, b, std::nullopt, output_mem_config), full_like(input_a, 0.0f, output_mem_config), result, output_mem_config);
    return ttnn::typecast(result, input_dtype);
}
Tensor fmod(const Tensor& input_a, const Tensor& input_b, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _fmod)(input_a, input_b, output_mem_config);
}

// logit(input, eps)=log(input / 1 - input)
Tensor _logit(const Tensor& input_a, float eps, const MemoryConfig& output_mem_config) {
    Tensor t_eps = full_like(input_a, eps, output_mem_config);
    Tensor t1m_eps = full_like(input_a, (1 - eps), output_mem_config);
    Tensor logit_input = where(
        ttnn::ltz(t_eps, output_mem_config),
        input_a,
        where(
            ttnn::lt(input_a, t_eps, std::nullopt, output_mem_config),
            t_eps,
            where(ttnn::gt(input_a, t1m_eps, std::nullopt, output_mem_config), t1m_eps, input_a, output_mem_config),
            output_mem_config),
        output_mem_config);
    t_eps.deallocate();
    t1m_eps.deallocate();
    Tensor linput_m1 = ttnn::rsub(logit_input, 1.0, output_mem_config);
    Tensor log_input =
        ttnn::multiply(logit_input, ttnn::reciprocal(linput_m1, output_mem_config), std::nullopt, output_mem_config);
    linput_m1.deallocate();
    Tensor t_inf =
        ttnn::multiply(ttnn::sign(input_a, output_mem_config), std::numeric_limits<float>::infinity(), std::nullopt, output_mem_config);
    Tensor logit_result = where(
        ttnn::eq(logit_input, 1.0, std::nullopt, output_mem_config),
        t_inf,
        where(ttnn::ltz(log_input, output_mem_config), std::nanf(" "), ttnn::log(log_input, output_mem_config), output_mem_config),
        output_mem_config);
    return logit_result;
}
Tensor logit(const Tensor& input_a, float eps, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _logit)(input_a, eps, output_mem_config);
}

// polygamma support for the range of input(1, 10) and n(1, 10)
Tensor _polygamma(const Tensor& input_a, uint32_t k, const MemoryConfig& output_mem_config) {
    float k_der = 1.0f + k;
    float fact_val = std::tgamma(k_der);
    float pos_neg = 1.0f;
    if (k == 2 || k == 4 || k == 6 || k == 8 || k == 10) {
        pos_neg = -1.0f;
    }
    Tensor temp(input_a);
    {
        Tensor z1 = ttnn::reciprocal(ttnn::power(input_a, k_der, output_mem_config), output_mem_config);
        temp = z1;
        for (int idx = 1; idx < 11; idx++) {
            z1 = ttnn::reciprocal(ttnn::power(ttnn::add(input_a, idx, std::nullopt, output_mem_config), k_der, output_mem_config), output_mem_config);
            temp = ttnn::add(temp, z1, std::nullopt, output_mem_config);
        }
    }
    fact_val *= pos_neg;
    return ttnn::multiply(temp, fact_val, std::nullopt, output_mem_config);
}
Tensor polygamma(const Tensor& input_a, uint32_t value, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _polygamma)(input_a, value, output_mem_config);
}

// logical_xori
Tensor _logical_xori(const Tensor& input_a, float value, const MemoryConfig& output_mem_config) {
    if (std::fpclassify(value) == FP_ZERO) {
        return ttnn::nez(input_a);
    } else {
        return ttnn::eqz(input_a);  // eqz( input_a ) = not( nez( input_a ) )
    }
}
Tensor logical_xori(const Tensor& input_a, float value, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _logical_xori)(input_a, value, output_mem_config);
}

// xlogy(x,y))=x*log(y)
Tensor _xlogy(const Tensor& input_a, const Tensor& input_b, const MemoryConfig& output_mem_config) {
    Tensor t_nan = full_like(input_b, std::nanf(" "), output_mem_config);
    Tensor result = ttnn::multiply(input_a, ttnn::log(input_b, output_mem_config), std::nullopt, output_mem_config);
    result = where(
        ttnn::logical_or(
            ttnn::ltz(input_b, output_mem_config),
            ttnn::eq(input_b, t_nan, std::nullopt, output_mem_config),
            std::nullopt,
            output_mem_config),
        t_nan,
        result,
        output_mem_config);
    return result;
}
Tensor xlogy(const Tensor& input_a, const Tensor& input_b, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _xlogy)(input_a, input_b, output_mem_config);
}

// Celu
// torch.where(x > 0, x, alpha * (torch.exp(x / alpha) - 1))
Tensor _celu(const Tensor& input_a, float alpha, const MemoryConfig& output_mem_config) {
    float recip_val = 1.0f / alpha;
    using ttnn::operations::unary::UnaryWithParam;
    using ttnn::operations::unary::UnaryOpType;
    std::vector<UnaryWithParam> ops_chain = {
    UnaryWithParam{UnaryOpType::MUL_UNARY_SFPU, recip_val},
    UnaryWithParam{UnaryOpType::EXP, 1.0f},
    UnaryWithParam{UnaryOpType::SUB_UNARY_SFPU, 1.0f}, UnaryWithParam{UnaryOpType::MUL_UNARY_SFPU, alpha} };

    Tensor result = ttnn::unary_chain(input_a, ops_chain, output_mem_config);
    result = where(ttnn::gtz(input_a, output_mem_config), input_a, result, output_mem_config);
    return result;
}
Tensor celu(const Tensor& input_a, float alpha, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _celu)(input_a, alpha, output_mem_config);
}


Tensor _variance_impl(
    const Tensor& y, const Tensor& mean_y, Tensor& y_minus_mean_y, const MemoryConfig& output_mem_config) {
    constexpr float correction = 0.0f;
    auto shape_wh = y.get_legacy_shape();
    float scale = 1.0f / ((float)(shape_wh[3] * shape_wh[2]) - correction);
    Tensor sqr_y_minus_mean_y = ttnn::square(y_minus_mean_y, output_mem_config);
    Tensor sum_sqr_y_minus_mean_y =
        reduce(sqr_y_minus_mean_y, ReduceOpMath::SUM, ReduceOpDim::HW, scale, output_mem_config);
    return sum_sqr_y_minus_mean_y;  // var
}
Tensor _variance_impl(const Tensor& y, const Tensor& mean_y, const MemoryConfig& output_mem_config) {
    Tensor y_minus_mean_y = ttnn::subtract(y, mean_y);
    return _variance_impl(y, mean_y, y_minus_mean_y, output_mem_config);
}
Tensor _variance(const Tensor& y, const MemoryConfig& output_mem_config) {
    Tensor mean_y = mean_hw(y);
    return _variance_impl(y, mean_y, output_mem_config);
}
Tensor var_hw(const Tensor& y, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _variance)(y, output_mem_config);
}

// Function std
// compute standard deviation of tensor y = sqrt( E((y-<y>)^2)/ y.volume() )
//  Ref: torch.std
Tensor _std(const Tensor& y, const Tensor& mean_y, const MemoryConfig& output_mem_config) {
    return ttnn::sqrt(_variance_impl(y, mean_y, output_mem_config));
}
Tensor _std(const Tensor& y, const Tensor& mean_y, Tensor& y_minus_mean_y, const MemoryConfig& output_mem_config) {
    return ttnn::sqrt(_variance_impl(y, mean_y, y_minus_mean_y, output_mem_config));
}
Tensor _std_overload(const Tensor& y, const MemoryConfig& output_mem_config) {
    return ttnn::sqrt(_variance(y, output_mem_config));
}
Tensor std_hw(const Tensor& y, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, tt::tt_metal::_std_overload)(y, output_mem_config);
}

// Function normalize
// use transformation y = (y - mean(y))/std(y) by broadcast
Tensor _normalize(const Tensor& y, const MemoryConfig& output_mem_config) {
    Tensor mean_y = mean_hw(y);
    Tensor y_minus_mean_y = ttnn::subtract(y, mean_y);
    Tensor std_y = tt::tt_metal::_std(y, mean_y, y_minus_mean_y, output_mem_config);
    Tensor recip_std_y = ttnn::reciprocal(std_y, output_mem_config);
    Tensor z = ttnn::multiply(y_minus_mean_y, recip_std_y);
    return z;
}
Tensor normalize_hw(const Tensor& y, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _normalize)(y, output_mem_config);
}

using HWFunctionT = std::function<Tensor(const Tensor& y, const MemoryConfig&)>;
Tensor _make_global_from_hw_impl(
    HWFunctionT fn, const Tensor& y, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    const Shape s_orig = y.get_legacy_shape();
    TT_FATAL(s_orig.rank() == 4, "Cannot support non-rank 4 Tensor");

    // format to HW
    Tensor y_hw = reshape(y, 1, 1, s_orig[2], s_orig[3] * s_orig[1] * s_orig[0], output_mem_config);

    // compute @fn
    Tensor z_0 = fn(y_hw, output_mem_config);
    TT_FATAL(y_hw.get_legacy_shape() == z_0.get_legacy_shape(), "shape match");
    y_hw.deallocate();

    // reformat
    Tensor z_1 = reshape(z_0, s_orig[0], s_orig[1], s_orig[2], s_orig[3], output_mem_config);
    z_0.deallocate();

    return z_1;
}

// Global Norm
Tensor _normalize_global(const Tensor& y, const MemoryConfig& output_mem_config) {
    return _make_global_from_hw_impl(normalize_hw, y, output_mem_config);
}
Tensor normalize_global(const Tensor& y, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _normalize_global)(y, output_mem_config);
}

Tensor _scatter(const Tensor& input_a, const Tensor& input_b, const MemoryConfig& output_mem_config) {
    tt::tt_metal::Array4D start_index = {0, 0, 0, 0};
    ttnn::Tensor input_tensor_4D = ttnn::unsqueeze_to_4D(input_a);

    Tensor index = ttnn::pad(0, ones_like(input_tensor_4D, output_mem_config), input_b.get_legacy_shape().to_array_4D(), start_index, 0, false, std::nullopt);
    Tensor temp_a = ttnn::pad(0, input_tensor_4D,input_b.get_legacy_shape().to_array_4D(), start_index, 0, false, std::nullopt);
    return where(index, temp_a, input_b, output_mem_config);
}
Tensor scatter(const Tensor& input_a, const Tensor& input_b, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _scatter)(input_a, input_b, output_mem_config);
}

// threshold(a,t,v) = (a <= t)*v + (a > t)*a
Tensor _threshold(const Tensor& input_tensor, float threshold, float value, const MemoryConfig& output_mem_config) {
    Tensor t_threshold = ttnn::operations::creation::create_scalar(
        threshold, input_tensor.get_dtype(), Layout::TILE, input_tensor.device());
    Tensor t0 = ttnn::subtract(input_tensor, t_threshold, std::nullopt, output_mem_config);
    t_threshold.deallocate();
    Tensor t_value =
        ttnn::operations::creation::create_scalar(value, input_tensor.get_dtype(), Layout::TILE, input_tensor.device());
    Tensor t1 = ttnn::multiply(ttnn::lez(t0), t_value, std::nullopt, output_mem_config);
    t_value.deallocate();
    Tensor t2 = ttnn::multiply(ttnn::gtz(t0, output_mem_config), input_tensor, std::nullopt, output_mem_config);
    return ttnn::add(t1, t2, std::nullopt, output_mem_config);
}
Tensor threshold(const Tensor& input_tensor, float threshold, float value, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _threshold)(input_tensor, threshold, value, output_mem_config);
}

// TODO: In future will uplift the op once the floor and tan has supported.
// digamma support for the range of (1, inf)
Tensor _digamma(const Tensor& input_a, const MemoryConfig& output_mem_config) {
    Tensor t_log_out = ttnn::log(input_a, output_mem_config);  // negative log is not useful here

    // 1/2(z)
    Tensor output = ttnn::multiply(ttnn::reciprocal(input_a, output_mem_config), 0.5f, std::nullopt, output_mem_config);
    Tensor tmp = ttnn::square(ttnn::reciprocal(input_a, output_mem_config), output_mem_config);
    Tensor val_square = tmp;
    // (1/12) * x^2
    output = ttnn::subtract(output, ttnn::multiply(tmp, 0.083333333f, std::nullopt, output_mem_config), std::nullopt, output_mem_config);

    // (1/120) * x^4
    tmp = ttnn::multiply(tmp, val_square, std::nullopt, output_mem_config);
    output =
        ttnn::add(output, ttnn::multiply(tmp, 0.008333333333333333f, std::nullopt, output_mem_config), std::nullopt, output_mem_config);

    //(1/252) * x^6
    tmp = ttnn::multiply(tmp, val_square, std::nullopt, output_mem_config);
    output = ttnn::subtract(
        output, ttnn::multiply(tmp, 0.003968253968253968f, std::nullopt, output_mem_config), std::nullopt, output_mem_config);

    // (1/240) *x^8
    tmp = ttnn::multiply(tmp, val_square, std::nullopt, output_mem_config);
    output =
        ttnn::add(output, ttnn::multiply(tmp, 0.004166666666666667f, std::nullopt, output_mem_config), std::nullopt, output_mem_config);

    //(1/132) * x^10
    tmp = ttnn::multiply(tmp, val_square, std::nullopt, output_mem_config);
    output = ttnn::subtract(
        output, ttnn::multiply(tmp, 0.007575757575757576, std::nullopt, output_mem_config), std::nullopt, output_mem_config);

    //(691/32760) * x^12
    tmp = ttnn::multiply(tmp, val_square, std::nullopt, output_mem_config);
    output =
        ttnn::add(output, ttnn::multiply(tmp, 0.021092796092796094, std::nullopt, output_mem_config), std::nullopt, output_mem_config);

    //(1/12) * x^14
    tmp = ttnn::multiply(tmp, val_square, std::nullopt, output_mem_config);
    output =
        ttnn::subtract(output, ttnn::multiply(tmp, 0.08333333333333333, std::nullopt, output_mem_config), std::nullopt, output_mem_config);

    return ttnn::subtract(t_log_out, output, std::nullopt, output_mem_config);
}
Tensor digamma(const Tensor& input_a, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _digamma)(input_a, output_mem_config);
}

// cbrt(a) = pow(a,1/3) or (cbrt(a))**3 = a.
//         = exp[ (1/3)*log[a] ]
Tensor _cbrt(const Tensor& input_tensor, const MemoryConfig& output_mem_config) {
    constexpr float scale = (float)(1.0 / 3.0);
    Tensor t_scale =
        ttnn::operations::creation::create_scalar(scale, input_tensor.get_dtype(), Layout::TILE, input_tensor.device());
    Tensor t_ln_input =
        ttnn::log(ttnn::abs(input_tensor, output_mem_config), output_mem_config);  // negative log is not useful here
    Tensor t1 = ttnn::multiply(t_ln_input, t_scale, std::nullopt, output_mem_config);
    t_scale.deallocate();
    t_ln_input.deallocate();
    Tensor t2 = ttnn::exp(t1, false, output_mem_config);
    t1.deallocate();
    Tensor t3 = ttnn::multiply(t2, ttnn::sign(input_tensor, output_mem_config), std::nullopt, output_mem_config);
    return t3;
}
Tensor cbrt(const Tensor& input_a, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _cbrt)(input_a, output_mem_config);
}

// where - ternary operator y = (predicate) ? value_true : value_false; elementwise
//            y = (predicate >= 0)*value_true + (predicate < 0)*value_false
Tensor _where(
    uint8_t queue_id,
    const Tensor& predicate,
    const Tensor& value_true,
    const Tensor& value_false,
    const MemoryConfig& output_mem_config,
    std::optional<Tensor> output_tensor) {

    Tensor t2 = ttnn::multiply(queue_id, ttnn::gtz(queue_id, predicate, output_mem_config), value_true, std::nullopt, output_mem_config);
    if(output_tensor.has_value())
    {
        ttnn::multiply(queue_id, ttnn::lez(queue_id, predicate, output_mem_config), value_false, std::nullopt, operation::DEFAULT_OUTPUT_MEMORY_CONFIG, output_tensor);
        ttnn::add(queue_id, t2, output_tensor.value(), std::nullopt, operation::DEFAULT_OUTPUT_MEMORY_CONFIG, output_tensor);
    }
    else
    {
        Tensor t1 = ttnn::multiply(queue_id, ttnn::lez(queue_id, predicate, output_mem_config), value_false, std::nullopt, output_mem_config);
        output_tensor = ttnn::add(queue_id, t2, t1, std::nullopt, output_mem_config);
    }
    return output_tensor.value();
}
Tensor _where_v1(
    uint8_t queue_id, const Tensor& predicate, const float value_true, const Tensor& value_false, const MemoryConfig& output_mem_config, std::optional<Tensor> output_tensor) {

    Tensor t2 = ttnn::multiply(queue_id, ttnn::gtz(queue_id, predicate, output_mem_config), value_true, std::nullopt, output_mem_config);

    if(output_tensor.has_value()){
        ttnn::multiply(queue_id, ttnn::lez(queue_id, predicate, output_mem_config), value_false, std::nullopt, operation::DEFAULT_OUTPUT_MEMORY_CONFIG , output_tensor);
        ttnn::add(queue_id, t2, output_tensor.value(), std::nullopt, operation::DEFAULT_OUTPUT_MEMORY_CONFIG, output_tensor);
    }
    else
    {
        Tensor t1 = ttnn::multiply(queue_id, ttnn::lez(queue_id, predicate, output_mem_config), value_false, std::nullopt, output_mem_config);
        output_tensor = ttnn::add(queue_id, t2, t1, std::nullopt, output_mem_config);
    }
    return output_tensor.value();
}
Tensor _where_v2(
    uint8_t queue_id, const Tensor& predicate, const Tensor& value_true, float value_false, const MemoryConfig& output_mem_config, std::optional<Tensor> output_tensor) {

    Tensor t1 = ttnn::multiply(queue_id, ttnn::lez(queue_id, predicate, output_mem_config), value_false, std::nullopt, output_mem_config);

    if(output_tensor.has_value()){
        ttnn::multiply(queue_id, ttnn::gtz(queue_id, predicate, output_mem_config), value_true, std::nullopt, operation::DEFAULT_OUTPUT_MEMORY_CONFIG, output_tensor);
        ttnn::add(queue_id, output_tensor.value(), t1, std::nullopt, operation::DEFAULT_OUTPUT_MEMORY_CONFIG, output_tensor);
    }
    else
    {
        Tensor t2 = ttnn::multiply(queue_id, ttnn::gtz(queue_id, predicate, output_mem_config), value_true, std::nullopt, output_mem_config);
        output_tensor = ttnn::add(queue_id, t2, t1, std::nullopt, output_mem_config);
    }
    return output_tensor.value();
}
Tensor _where_v3(
    uint8_t queue_id, const Tensor& predicate, const float value_true, const float value_false, const MemoryConfig& output_mem_config, std::optional<Tensor> output_tensor) {
    Tensor t2 = ttnn::multiply(queue_id, ttnn::gtz(queue_id, predicate, output_mem_config), value_true, std::nullopt, output_mem_config);
    Tensor t1 = ttnn::multiply(queue_id, ttnn::lez(queue_id, predicate, output_mem_config), value_false, std::nullopt, output_mem_config);
    if(output_tensor.has_value()){
        ttnn::add(queue_id, t2, t1, std::nullopt, operation::DEFAULT_OUTPUT_MEMORY_CONFIG, output_tensor);
    } else {
        output_tensor = ttnn::add(queue_id, t2, t1, std::nullopt, output_mem_config);
    }
    return output_tensor.value();
}

Tensor where(
    const Tensor& predicate,
    const Tensor& value_true,
    const Tensor& value_false,
    const MemoryConfig& output_mem_config,
    std::optional<Tensor> output_tensor) {
    uint8_t default_queue_id = 0;
    return operation::decorate_as_composite(__func__, _where)(
        default_queue_id, predicate, value_true, value_false, output_mem_config, output_tensor);
}
Tensor where(
    const Tensor& predicate,
    const float value_true,
    const Tensor& value_false,
    const MemoryConfig& output_mem_config,
    std::optional<Tensor> output_tensor) {
    uint8_t default_queue_id = 0;
    return operation::decorate_as_composite(__func__, _where_v1)(
        default_queue_id, predicate, value_true, value_false, output_mem_config, output_tensor);
}
Tensor where(
    const Tensor& predicate,
    const Tensor& value_true,
    const float value_false,
    const MemoryConfig& output_mem_config,
    std::optional<Tensor> output_tensor) {
    uint8_t default_queue_id = 0;
    return operation::decorate_as_composite(__func__, _where_v2)(
        default_queue_id, predicate, value_true, value_false, output_mem_config, output_tensor);
}
Tensor where(
    const Tensor& predicate,
    const float value_true,
    const float value_false,
    const MemoryConfig& output_mem_config,
    std::optional<Tensor> output_tensor) {
    uint8_t default_queue_id = 0;
    return operation::decorate_as_composite(__func__, _where_v3)(
        default_queue_id, predicate, value_true, value_false, output_mem_config, output_tensor);
}

Tensor where(
    uint8_t queue_id,
    const Tensor& predicate,
    const Tensor& value_true,
    const Tensor& value_false,
    const MemoryConfig& output_mem_config,
    std::optional<Tensor> output_tensor) {
    return operation::decorate_as_composite(__func__, _where)(
        queue_id, predicate, value_true, value_false, output_mem_config, output_tensor);
}
Tensor where(
    uint8_t queue_id,
    const Tensor& predicate,
    const float value_true,
    const Tensor& value_false,
    const MemoryConfig& output_mem_config,
    std::optional<Tensor> output_tensor) {
    return operation::decorate_as_composite(__func__, _where_v1)(
        queue_id, predicate, value_true, value_false, output_mem_config, output_tensor);
}
Tensor where(
    uint8_t queue_id,
    const Tensor& predicate,
    const Tensor& value_true,
    const float value_false,
    const MemoryConfig& output_mem_config,
    std::optional<Tensor> output_tensor) {
    return operation::decorate_as_composite(__func__, _where_v2)(
        queue_id, predicate, value_true, value_false, output_mem_config, output_tensor);
}
Tensor where(
    uint8_t queue_id,
    const Tensor& predicate,
    const float value_true,
    const float value_false,
    const MemoryConfig& output_mem_config,
    std::optional<Tensor> output_tensor) {
    return operation::decorate_as_composite(__func__, _where_v3)(
        queue_id, predicate, value_true, value_false, output_mem_config, output_tensor);
}

// on-device tensor creation 0s like @reference_tensor
Tensor zeros_like(
    uint8_t queue_id,
    const Tensor& reference_tensor,
    const MemoryConfig& output_mem_config,
    std::optional<Tensor> output_tensor) {
    return mk_zero_tensor_like(reference_tensor, output_mem_config, output_tensor);
}
Tensor zeros_like(
    const Tensor& reference_tensor, const MemoryConfig& output_mem_config, std::optional<Tensor> output_tensor) {
    uint8_t default_queue_id = 0;
    return mk_zero_tensor_like(default_queue_id, reference_tensor, output_mem_config, output_tensor);
}

// on-device tensor creation 1s like @reference_tensor
Tensor ones_like(const Tensor& reference_tensor, const MemoryConfig& output_mem_config) {
    return mk_filled_tensor_like(reference_tensor, 1.0f, output_mem_config);
}

// on-device tensor creation with value like @reference_tensor
Tensor full_like(
    const Tensor& reference_tensor,
    float value,
    const MemoryConfig& output_mem_config,
    std::optional<Tensor> output_tensor) {
    uint8_t default_queue_id = 0;
    return mk_filled_tensor_like(reference_tensor, value, output_mem_config, output_tensor, default_queue_id);
}
Tensor full_like(
    uint8_t queue_id,
    const Tensor& reference_tensor,
    float value,
    const MemoryConfig& output_mem_config,
    std::optional<Tensor> output_tensor) {
    return mk_filled_tensor_like(reference_tensor, value, output_mem_config, output_tensor, queue_id);
}

// hardtanh
Tensor _hardtanh(
    const Tensor& a, float low /* = -1.0f */, float high /* = +1.0f */, const MemoryConfig& output_mem_config) {
    return clip(a, low, high, output_mem_config);
}
Tensor hardtanh(
    const Tensor& a, float low /* = -1.0f */, float high /* = +1.0f */, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _hardtanh)(a, low, high, output_mem_config);
}

// clamp
Tensor _clamp(const Tensor& a, float low, float high, const MemoryConfig& output_mem_config) {
    return clip(a, low, high, output_mem_config);
}
Tensor clamp(const Tensor& a, float low, float high, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _clamp)(a, low, high, output_mem_config);
}

// on-device tensor creation 0s with shape
Tensor zeros(
    const Shape shape, DataType data_type, Layout layout, Device* device, const MemoryConfig& output_mem_config) {
    return tt::numpy::zeros(shape, data_type, layout, device, output_mem_config);
}

// on-device tensor creation 1s with shape
Tensor ones(
    const Shape shape, DataType data_type, Layout layout, Device* device, const MemoryConfig& output_mem_config) {
    return tt::numpy::ones(shape, data_type, layout, device, output_mem_config);
}

// on-device tensor creation with shape and filled with value
Tensor full(
    const Shape shape,
    float value,
    DataType data_type,
    Layout layout,
    Device* device,
    const MemoryConfig& output_mem_config) {
    return tt::numpy::full(shape, value, data_type, layout, device, output_mem_config);
}

/**
 * outer product = matrix multiply when a = [1,1,N,1] and b = [1,1,1,M]
 * and result is of size [1,1,N,M].
 * - implementation supports any 1D "squeezable tensor" at input operands
 *   by running reshape.
 */
Tensor _outer(Tensor& a, Tensor& b, const MemoryConfig& output_mem_config) {
    const Shape s_a = a.get_legacy_shape();
    const Shape s_b = b.get_legacy_shape();

    auto num_ones = [](const Shape& s) -> uint32_t {
        uint32_t num1s = 0;
        for (uint32_t idx = 0; idx < 4; idx++) num1s += (uint32_t)(s[idx] == 1);
        return num1s;
    };

    // check if 3 dimensions are 1
    TT_ASSERT(!(num_ones(s_a) < 3), "3 dimensions are required to be 1 for use with outer product");
    TT_ASSERT(!(num_ones(s_b) < 3), "3 dimensions are required to be 1 for use with outer product");

    const bool skip_reshape_a = (s_a[0] == 1 && s_a[1] == 1 && s_a[2] >= 1 && s_a[3] == 1);
    const bool skip_reshape_b = (s_b[0] == 1 && s_b[1] == 1 && s_b[2] == 1 && s_b[3] >= 1);

    Tensor a_slim = a;
    Tensor b_slim = b;

    if (!skip_reshape_a) {
        a_slim = reshape(a, 1, 1, a.volume(), 1, output_mem_config);
    }
    if (!skip_reshape_b) {
        b_slim = reshape(b, 1, 1, 1, b.volume(), output_mem_config);
    }
    a_slim = ttnn::to_layout(a_slim, ttnn::TILE_LAYOUT, std::nullopt, std::nullopt, (Device*)nullptr);
    b_slim = ttnn::to_layout(b_slim, ttnn::TILE_LAYOUT, std::nullopt, std::nullopt, (Device*)nullptr);
    Device* device = AutoFormat::GetDefaultDevice();
    if (device != nullptr) {
        if (a_slim.storage_type() != tt::tt_metal::StorageType::DEVICE) {
            a_slim = AutoFormat::move_tensor_to_device(a_slim, device);
        }
        if (b_slim.storage_type() != tt::tt_metal::StorageType::DEVICE) {
            b_slim = AutoFormat::move_tensor_to_device(b_slim, device);
        }
    }

    return ttnn::operations::matmul::matmul(
            a_slim,
            b_slim,
            /*bias=*/std::nullopt,
            tt::operations::primary::Matmul{
            /*program_config=*/std::nullopt,
            /*bcast_batch=*/std::nullopt,
            output_mem_config}
            );
}
Tensor outer(Tensor& a, Tensor& b, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _outer)(a, b, output_mem_config);
}

std::vector<Tensor> split_tensor_for_glu(const Tensor& input_a, int32_t dim, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> t_split;
    Shape inshape = input_a.get_legacy_shape();
    TT_FATAL(((inshape[dim] / 2) % TILE_WIDTH == 0), "Split tensor dimension should be in full tile");
    std::vector<uint32_t> s_a = {0, 0, 0, 0};
    std::vector<uint32_t> e_a = {inshape[0] - 1, inshape[1] - 1, inshape[2] - 1, inshape[3] / 2 - 1};

    std::vector<uint32_t> s_b = {0, 0, 0, inshape[3] / 2};
    std::vector<uint32_t> e_b = {inshape[0] - 1, inshape[1] - 1, inshape[2] - 1, inshape[3] - 1};

    Tensor t_a = ttnn::slice(0, input_a, s_a, e_a, output_mem_config);
    Tensor t_b = ttnn::slice(0, input_a, s_b, e_b, output_mem_config);

    t_split.emplace_back(t_a);
    t_split.emplace_back(t_b);

    return t_split;
}


// on-device tensor creation with shape and filled with value
Tensor _sfpu_eps(const Shape shape, Layout layout, Device* device, const MemoryConfig& output_mem_config) {
    float value = device->sfpu_eps();
    return tt::numpy::full(shape, value, DataType::BFLOAT16, layout, device, output_mem_config);
}
Tensor sfpu_eps(const Shape shape, Layout layout, Device* device, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _sfpu_eps)(shape, layout, device, output_mem_config);
}

// tril : select lower triangular region of input matrix
Tensor _tril(const Tensor& input_a, int32_t diag, const MemoryConfig& output_mem_config) {
    Tensor index_l = tt::numpy::index_tril<bfloat16>(
        input_a.get_legacy_shape(), diag, DataType::BFLOAT16, Layout::TILE, input_a.device(), output_mem_config);
    return ttnn::multiply(input_a, index_l, std::nullopt, output_mem_config);
}
Tensor tril(
    const Tensor& input_a,
    int32_t dim /* = -1 */,
    const MemoryConfig& output_mem_config /* = operation::DEFAULT_OUTPUT_MEMORY_CONFIG */) {
    return operation::decorate_as_composite(__func__, _tril)(input_a, dim, output_mem_config);
}

// triu : select upper triangular region of input matrix
Tensor _triu(const Tensor& input_a, int32_t diag, const MemoryConfig& output_mem_config) {
    Tensor index_u = tt::numpy::index_triu<bfloat16>(
        input_a.get_legacy_shape(), diag, DataType::BFLOAT16, Layout::TILE, input_a.device(), output_mem_config);
    return ttnn::multiply(input_a, index_u, std::nullopt, output_mem_config);
}
Tensor triu(
    const Tensor& input_a,
    int32_t dim /* = -1 */,
    const MemoryConfig& output_mem_config /* = operation::DEFAULT_OUTPUT_MEMORY_CONFIG */) {
    return operation::decorate_as_composite(__func__, _triu)(input_a, dim, output_mem_config);
}

Tensor _power_fp(uint8_t queue_id, const Tensor& input_a, float exponent, const MemoryConfig& output_mem_config, std::optional<Tensor> output_tensor) {
    TT_FATAL(exponent >= 0.0f, "works for positive exponents only");
    const uint32_t exponent_floor = static_cast<uint32_t>(std::floor(exponent));
    if (static_cast<float>(exponent_floor) == exponent) {
        if(output_tensor.has_value()){
            ttnn::power(queue_id,input_a, exponent_floor, output_mem_config, output_tensor);
            return output_tensor.value();
        }
        return ttnn::power(queue_id, input_a, exponent_floor, output_mem_config);
    }
    const float exponent_trunc = exponent - static_cast<float>(exponent_floor);
    Tensor pow_trunc_log = ttnn::multiply(queue_id, ttnn::log(queue_id, input_a, output_mem_config), exponent_trunc, std::nullopt, output_mem_config);
    Tensor pow_frac = ttnn::exp(queue_id, pow_trunc_log, false, output_mem_config);
    pow_trunc_log.deallocate();
    float t_nan = std::nanf("");
    Tensor result = ttnn::multiply(queue_id, ttnn::power(queue_id, input_a, exponent_floor, output_mem_config), pow_frac, std::nullopt, output_mem_config);
    // To handle negative inputs:
    // in torch For -ve inputs with float exponent power returns nan
    if(output_tensor.has_value()){
        where(queue_id, ttnn::ltz(queue_id, input_a, output_mem_config), t_nan, result, operation::DEFAULT_OUTPUT_MEMORY_CONFIG, output_tensor);
        return output_tensor.value();
    }
    result = where(queue_id, ttnn::ltz(queue_id, input_a, output_mem_config), t_nan, result);
    return result;
}
Tensor power_fp(
    uint8_t queue_id,
    const Tensor& input_a,
    float exponent,
    const MemoryConfig& output_mem_config, /* = operation::DEFAULT_OUTPUT_MEMORY_CONFIG */
    std::optional<Tensor> output_tensor) {
    return operation::decorate_as_composite(__func__, _power_fp)(queue_id, input_a, exponent, output_mem_config, output_tensor);
}

Tensor pow(uint8_t queue_id, const Tensor& input_a, float exponent, const MemoryConfig& output_mem_config, std::optional<Tensor> output_tensor) {
    return power_fp(queue_id, input_a, exponent, output_mem_config, output_tensor);
}
Tensor pow(const Tensor& input_a, float exponent, const MemoryConfig& output_mem_config, std::optional<Tensor> output_tensor) {
    uint8_t default_queue_id = 0;
    return power_fp(default_queue_id, input_a, exponent, output_mem_config, output_tensor);
}

Tensor pow(uint8_t queue_id, const Tensor& input_a, int exponent, const MemoryConfig& output_mem_config, std::optional<Tensor> output_tensor) {
    return ttnn::power(queue_id, input_a, exponent, output_mem_config, output_tensor);
}
Tensor pow(const Tensor& input_a, int exponent, const MemoryConfig& output_mem_config, std::optional<Tensor> output_tensor) {
    uint8_t default_queue_id = 0;
    return ttnn::power(default_queue_id, input_a, exponent, output_mem_config, output_tensor);
}

Tensor create_mask(const Tensor& input_a, const MemoryConfig& output_mem_config) {
    auto& padded_shape = input_a.get_legacy_shape();
    auto& unpadded_shape = padded_shape.without_padding();
    if (padded_shape == unpadded_shape)
        return input_a;
    float t_inf = -std::numeric_limits<float>::infinity();
    Tensor masked_input = tt::numpy::mask_padded_input<bfloat16>(padded_shape, unpadded_shape, DataType::BFLOAT16);
    masked_input = where(masked_input, input_a, t_inf, output_mem_config);
    return masked_input;
}
// Argmax returns the index of maximum element in the tensor
Tensor _argmax(const Tensor& input_t, int64_t _dim, bool all, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input_t}))};
    operation::launch_with_autoformat(
        [_dim, all, output_mem_config](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            const auto& input = input_tensors.at(0);
            auto& input_shape = input.get_legacy_shape();
            TT_FATAL(input_shape.rank() == 4, "supported for rank-4 tensors at this time");

            Tensor input_a = create_mask(input, output_mem_config);

            uint32_t dim = input_shape.get_normalized_index(_dim);
            int size = input_a.volume();

            if (!all) {
                if ((dim == (input_shape.rank() - 1)) || (dim == (input_shape.rank() - 2))) {
                    bool is_width = (dim == (input_shape.rank() - 1));
                    Tensor max_val = max(input_a, dim, output_mem_config);
                    Tensor max_tensor = zeros_like(input_a, output_mem_config);
                    Tensor tindex = tt::numpy::index_width<bfloat16>(
                        input_shape, DataType::BFLOAT16, Layout::TILE, input_a.device(), output_mem_config);
                    if (is_width) {
                        max_tensor = ttnn::add(max_tensor, max_val, std::nullopt, output_mem_config);
                    } else {
                        tindex = tt::numpy::index_height<bfloat16>(
                            input_shape, DataType::BFLOAT16, Layout::TILE, input_a.device(), output_mem_config);
                        max_tensor = ttnn::add(max_tensor, max_val, std::nullopt, output_mem_config);
                    }
                    tindex = tindex.to(input_a.device());
                    max_val.deallocate();
                    Tensor cmp_results = ttnn::eq(input_a, max_tensor, std::nullopt, output_mem_config);
                    max_tensor.deallocate();
                    Tensor max_indices = ttnn::multiply(cmp_results, tindex, std::nullopt, output_mem_config);
                    cmp_results.deallocate();
                    Tensor result = where(ttnn::eqz(max_indices), size, max_indices, output_mem_config);
                    max_indices.deallocate();
                    result = min(result, dim, output_mem_config);
                    Tensor res_index = zeros_like(result, output_mem_config);
                    result = where(ttnn::eq(result, size), res_index, result, output_mem_config);
                    std::vector<int64_t> permute_dims = {3, 0, 1, 2};
                    if (is_width) {
                        res_index = ttnn::add(res_index, result, std::nullopt, output_mem_config);
                    } else {
                        res_index = ttnn::add(res_index, result, std::nullopt, output_mem_config);
                        permute_dims[0] = 2;
                        permute_dims[3] = 3;
                    }
                    result.deallocate();
                    Tensor transpose_res = ttnn::permute(res_index, permute_dims, output_mem_config);
                    return {transpose_res};
                } else if ((dim == (input_shape.rank() - 3)) || (dim == (input_shape.rank() - 4))) {
                    bool is_channel = (dim == (input_shape.rank() - 3));
                    Tensor max_val = max(input_a, dim, output_mem_config);
                    int repeat = input.get_shape()[dim];
                    std::vector<Tensor> combined_tensors;
                    for (int cid = 0; cid < repeat; cid++) combined_tensors.emplace_back(max_val);
                    max_val.deallocate();
                    Tensor concat_out = concat(combined_tensors, dim, output_mem_config);
                    // Needed till `max` stops autoformatting output
                    concat_out = ttnn::reshape(concat_out, input_a.get_shape());
                    Tensor cmp_results = ttnn::eq(input_a, concat_out, std::nullopt, output_mem_config);
                    concat_out.deallocate();
                    Tensor tindex = tt::numpy::index_channel<bfloat16>(
                        input_shape, DataType::BFLOAT16, Layout::TILE, input_a.device(), output_mem_config);
                    if (!is_channel) {
                        tindex = tt::numpy::index_batch<bfloat16>(
                            input_shape, DataType::BFLOAT16, Layout::TILE, input_a.device(), output_mem_config);
                    }
                    tindex = tindex.to(input_a.device());
                    Tensor max_indices = ttnn::multiply(cmp_results, tindex, std::nullopt, output_mem_config);
                    cmp_results.deallocate();
                    Tensor midx = full_like(max_indices, size);
                    Tensor result = where(ttnn::eqz(max_indices), midx, max_indices, output_mem_config);
                    max_indices.deallocate();
                    result = min(result, dim, output_mem_config);
                    Tensor res_index = zeros_like(result, output_mem_config);
                    result = where(ttnn::eq(result, full_like(result, size)), res_index, result, output_mem_config);
                    res_index.deallocate();
                    if (is_channel) {
                        std::vector<int64_t> permute_dims = {1, 0, 2, 3};
                        Tensor transpose_res = ttnn::permute(result, permute_dims, output_mem_config);
                        return {transpose_res};
                    } else {
                        return {result};
                    }
                }
            }
            // TODO: Fix the index generation code. With the fix the code will work for argmax that return entire
            // maximum value index
            Tensor tindex = tt::numpy::index_all<bfloat16>(
                input_shape, DataType::BFLOAT16, Layout::TILE, input_a.device(), output_mem_config);
            Tensor max_val = global_max(input_a, output_mem_config);
            Tensor max_tensor = zeros_like(input_a, output_mem_config);
            max_tensor = ttnn::add(max_tensor, max_val, std::nullopt, output_mem_config);
            max_val.deallocate();
            Tensor cmp_results = ttnn::eq(input_a, max_tensor, std::nullopt, output_mem_config);
            max_tensor.deallocate();
            Tensor max_indices = ttnn::multiply(cmp_results, tindex, std::nullopt, output_mem_config);
            cmp_results.deallocate();
            Tensor result = where(ttnn::eqz(max_indices), size, max_indices, output_mem_config);
            max_indices.deallocate();
            result = global_min(result, output_mem_config);
            return {result};
        },
        {input_t},
        output_tensors);
    return output_tensors.at(0);
}

Tensor argmax(
    const Tensor& input_a,
    int64_t dim,
    bool all,
    const MemoryConfig& output_mem_config /* = operation::DEFAULT_OUTPUT_MEMORY_CONFIG */) {
    return operation::decorate_as_composite(__func__, _argmax)(input_a, dim, all, output_mem_config);
}

Tensor _argmin(const Tensor& input_a, int64_t _dim, bool all, const MemoryConfig& output_mem_config) {
    Tensor neg_input = ttnn::neg(input_a, output_mem_config);
    return (argmax(neg_input, _dim, all, output_mem_config));
}
Tensor argmin(
    const Tensor& input_a,
    int64_t dim,
    bool all,
    const MemoryConfig& output_mem_config /* = operation::DEFAULT_OUTPUT_MEMORY_CONFIG */) {
    return operation::decorate_as_composite(__func__, _argmin)(input_a, dim, all, output_mem_config);
}
}  // namespace tt_metal

}  // namespace tt
