// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_softplus.h"
#include "llk_math_eltwise_unary_sfpu_3_param.h"
#include "llk_math_eltwise_unary_sfpu_init.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_softplus_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::softplus, APPROXIMATE>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_softplus(
    uint dst_index, uint param0, uint param1, uint param2, int vector_mode = (int)VectorMode::RC) {
    llk_math_eltwise_unary_sfpu_3_param<APPROXIMATE>(
        ckernel::sfpu::calculate_softplus<APPROXIMATE>,
        ckernel::sfpu::calculate_softplus<APPROXIMATE>,
        dst_index,
        vector_mode,
        param0,
        param1,
        param2);
}

}  // namespace ckernel
