// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr0  = get_arg_val<uint32_t>(0);

    constexpr uint32_t out_cb_index = get_compile_time_arg_val(0);
    constexpr bool out_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr uint32_t Ht = get_compile_time_arg_val(2);
    constexpr uint32_t K = get_compile_time_arg_val(3);
    constexpr uint32_t Kt =  K % 32 == 0 ? K/32 : K/32 + 1;

    // can amortize the noc reads by doing them side by side for the two tensors
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(out_cb_index);
    const DataFormat data_format = get_dataformat(out_cb_index);

    const InterleavedAddrGenFast<out_is_dram> interleaved_accessor0 = {
        .bank_base_address = dst_addr0,
        .page_size = tile_bytes,
        .data_format = data_format
    };

    // // Get Kt rows of values and then Kt rows of indices from compute kernel
    for (uint32_t j = 0; j < Ht; ++j) {
        for (uint32_t i = 0; i < Kt; ++i) {
            //cb_wait_front(out_cb_index, onetile);
            // uint32_t l1_read_addr = get_read_ptr(out_cb_index);
            // noc_async_write_tile(j*Kt + i, interleaved_accessor0, l1_read_addr);
            // noc_async_write_barrier();
            //cb_pop_front(out_cb_index, onetile);
        }
    }
}
