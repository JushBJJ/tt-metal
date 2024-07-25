// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "ttnn/deprecated/tt_dnn/op_library/bcast/bcast_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/composite/composite_ops.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/complex/complex_ops.hpp"
#include "ttnn/operations/eltwise/complex_binary_backward/device/complex_binary_backward_op.hpp"
#include "ttnn/operations/eltwise/complex_binary/device/complex_binary_op.hpp"

namespace ttnn::operations::complex_unary {
using ComplexTensor = complex_binary::ComplexTensor;

Tensor _real(const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    return input[0];
}

Tensor _imag(const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    return input[1];
}

Tensor _angle(const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    return ttnn::neg( atan2(input[1],input[0],output_mem_config), output_mem_config );
}

Tensor _is_imag(const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    return ttnn::eqz( input[0], output_mem_config);
}

Tensor _is_real(const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    return ttnn::eqz( input[1], output_mem_config);
}

Tensor _abs(const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    return tt::tt_metal::hypot(input[0],input[1],output_mem_config);
}

ComplexTensor _conj(const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    return ComplexTensor({input[0], ttnn::neg(input[1],output_mem_config)});
}

ComplexTensor _reciprocal(const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    Tensor a_plus_b = ttnn::add(input[0],input[1],std::nullopt,output_mem_config);
    Tensor a_minus_b = ttnn::subtract(input[0],input[1],std::nullopt,output_mem_config);
    Tensor asqr_plus_bsqr = ttnn::add(ttnn::square(input[0],output_mem_config),ttnn::square(input[1],output_mem_config),
                                std::nullopt,output_mem_config);
    Tensor inv_dr = ttnn::reciprocal( asqr_plus_bsqr, output_mem_config );
    Tensor conj_im = ttnn::multiply( ttnn::neg(input[1],output_mem_config), inv_dr, std::nullopt, output_mem_config);
    Tensor conj_re = ttnn::multiply( input[0], inv_dr, std::nullopt, output_mem_config);
    return ComplexTensor({ conj_re, conj_im});
}

ComplexTensor _polar(const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    const Tensor& input_a = input.real();
    const Tensor& input_b = input.imag();
    Tensor c = ttnn::cos(input_b,output_mem_config);
    Tensor r = ttnn::multiply(input_a,c,std::nullopt,output_mem_config);
    c.deallocate();

    Tensor s = ttnn::sin(input_b,output_mem_config);
    Tensor i = ttnn::multiply(input_a,s,std::nullopt,output_mem_config);
    s.deallocate();

    return ComplexTensor({r,i});
}

}  // namespace ttnn::operations::complex_unary
