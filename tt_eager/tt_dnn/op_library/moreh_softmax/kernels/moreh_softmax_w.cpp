// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#include "tt_eager/tt_dnn/kernels/compute/moreh_common.hpp"

namespace NAMESPACE {

void MAIN {
    constexpr auto cb_in0 = tt::CB::c_in0;
    constexpr auto cb_mask = tt::CB::c_in1;
    constexpr auto cb_bcast_scaler = tt::CB::c_in2;
    constexpr auto cb_out0 = tt::CB::c_out0;
    constexpr auto cb_exps = tt::CB::c_intermed0;
    constexpr auto cb_recipsumexps = tt::CB::c_intermed1;
    constexpr auto cb_max = tt::CB::c_intermed2;
    constexpr auto cb_x_m_max = tt::CB::c_intermed3;
    constexpr auto cb_tmp = tt::CB::c_intermed4;

    binary_op_init_common(cb_in0, cb_bcast_scaler);

    constexpr int dst0 = 0;
    constexpr int dst1 = 1;
    constexpr uint32_t onetile = 1;

    uint32_t N = get_compile_time_arg_val(0);
    uint32_t Wt = get_compile_time_arg_val(1);

    cb_wait_front(cb_mask, onetile);
    cb_wait_front(cb_bcast_scaler, onetile);

    for (uint32_t n = 0; n < N; ++n) {

        // If you want to check the value of intermed0, please comment it out.
        // copy_tile_to_cb(tt::CB::c_intermed0, cb_out0);

        // If you want to check the value of intermed0, please comment it out.
        // copy_tile_to_cb(tt::CB::c_intermed1, cb_out0);

        reduce_tile_to_cb<false>(REDUCE_OP, REDUCE_DIM, tt::CB::c_intermed0, tt::CB::c_intermed1, cb_out0, 1);
    }
}
}  // namespace NAMESPACE
