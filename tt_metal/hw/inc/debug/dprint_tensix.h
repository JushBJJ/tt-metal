// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dprint.h"
#include "compute_kernel_api.h"

// Print the contents of tile with index tile_id within the destination register
void dprint_tensix_dest_reg(int tile_id = 0) {
    dbg_halt();
    MATH({
        DPRINT << FIXED() << SETPRECISION(2);
        uint32_t rd_data[8+1]; // data + array_type
        for (int row = 0; row < 64; row++) {
            dbg_read_dest_acc_row(row + 64 * tile_id, rd_data);
            DPRINT << SETW(6) << TYPED_U32_ARRAY(TypedU32_ARRAY_Format_TensixRegister_FP16_B, rd_data, 8);
            if (row % 2 == 1) DPRINT << ENDL();
        }
    })
    dbg_unhalt();
}
