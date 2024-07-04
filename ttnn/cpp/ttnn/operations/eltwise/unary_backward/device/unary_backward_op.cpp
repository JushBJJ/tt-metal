// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/eltwise/unary_backward/device/unary_backward_op.hpp"

#include "third_party/magic_enum/magic_enum.hpp"
#include "tt_eager/tt_dnn/op_library/bcast/bcast_op.hpp"
#include "tt_eager/tt_dnn/op_library/unpad/unpad_op.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"

namespace ttnn::operations::unary_backward {

namespace utils {


std::vector<ttnn::Tensor> _mul_bw(
    const Tensor& grad, const Tensor& input, float scalar, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result = mul_unary(grad, scalar, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

std::vector<Tensor> _clamp_min_bw(
    const Tensor& grad, const Tensor& input, float min, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor minT = gte_unary(input, min, output_mem_config);
    Tensor result = ttnn::multiply(grad, minT, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}


std::function<std::vector<ttnn::Tensor>(const Tensor&, const Tensor&, const MemoryConfig&)> unary_get_function_type1(UnaryBackwardOpType OpType){
    switch (OpType) {
        default:
            TT_ASSERT(false && "Undefined op type");
            return 0;
    }
}

std::function<std::vector<ttnn::Tensor>(const Tensor&, const Tensor&, float, const MemoryConfig&)> unary_get_function_type1_w_float(UnaryBackwardOpType OpType){
    switch (OpType) {
        case UnaryBackwardOpType::MUL_BW:
            return _mul_bw;
        case UnaryBackwardOpType::CLAMP_MIN_BW:
            return _clamp_min_bw;
        default:
            TT_ASSERT(false && "Undefined op type");
            return 0;
    }
}

std::function<std::vector<ttnn::Tensor>(const Tensor&, const Tensor&, const Tensor&, const MemoryConfig&)> unary_get_function_overload_type1(UnaryBackwardOpType OpType){
    switch (OpType) {
        case UnaryBackwardOpType::MUL_BW:
            return ttnn::operations::binary_backward::utils::_mul_bw_inter;
        default:
            TT_ASSERT(false && "Undefined op type");
            return 0;
    }
}

std::function<std::vector<std::optional<ttnn::Tensor>>(uint8_t , const Tensor&, const Tensor&, const Tensor&, const MemoryConfig&, const std::vector<bool>&, std::optional<Tensor>, std::optional<Tensor>)> unary_get_function_overload_type2(UnaryBackwardOpType OpType){
    switch (OpType) {
        case UnaryBackwardOpType::MUL_BW:
            return ttnn::operations::binary_backward::utils::_mul_bw;
        default:
            TT_ASSERT(false && "Undefined op type");
            return 0;
    }
}

std::function<std::vector<std::optional<ttnn::Tensor>>(const Tensor&, const Tensor&, const Tensor&, const MemoryConfig&, const std::vector<bool>&, std::optional<Tensor>, std::optional<Tensor>)> unary_get_function_overload_type2_wo_qid(UnaryBackwardOpType OpType){
    switch (OpType) {
        case UnaryBackwardOpType::MUL_BW:
            return ttnn::operations::binary_backward::utils::_mul_bw_overload;
        default:
            TT_ASSERT(false && "Undefined op type");
            return 0;
    }
}

}

}  // namespace ttnn::operations::unary
