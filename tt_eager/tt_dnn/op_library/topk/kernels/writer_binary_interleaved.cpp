// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#define DEBUG_PRINT 0

#if DEBUG_PRINT == 1
#include "debug/dprint.h"
#endif

void kernel_main() {
    uint32_t dst_addr0  = get_arg_val<uint32_t>(0);
    uint32_t dst_addr1  = get_arg_val<uint32_t>(1);

    constexpr uint32_t values_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t output_ind_cb_index = get_compile_time_arg_val(1);
    constexpr bool values_is_dram = get_compile_time_arg_val(2) == 1;
    constexpr bool output_ind_is_dram = get_compile_time_arg_val(3) == 1;
    constexpr uint32_t num_tiles = get_compile_time_arg_val(4);

    // can amortize the noc reads by doing them side by side for the two tensors
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes_values = get_tile_size(values_cb_index);
    const DataFormat data_format_values = get_dataformat(values_cb_index);

    const InterleavedAddrGenFast<values_is_dram> interleaved_accessor0 = {
        .bank_base_address = dst_addr0,
        .page_size = tile_bytes_values,
        .data_format = data_format_values
    };

    for (uint32_t i = 0; i < num_tiles; ++ i) {
        cb_wait_front(values_cb_index, onetile);
        uint32_t l1_read_addr = get_read_ptr(values_cb_index);
        noc_async_write_tile(i, interleaved_accessor0, l1_read_addr);
        noc_async_write_barrier();
        cb_pop_front(values_cb_index, onetile);
    }

    // single-tile ublocks
    const uint32_t tile_bytes_ind = get_tile_size(output_ind_cb_index);
    const DataFormat data_format_ind = get_dataformat(output_ind_cb_index);

    const InterleavedAddrGenFast<output_ind_is_dram> interleaved_accessor1 = {
        .bank_base_address = dst_addr1,
        .page_size = tile_bytes_ind,
        .data_format = data_format_ind
    };

    for (uint32_t i = 0; i < num_tiles; ++ i) {
        cb_wait_front(output_ind_cb_index, onetile);
        uint32_t l1_read_addr = get_read_ptr(output_ind_cb_index);
        noc_async_write_tile(i, interleaved_accessor1, l1_read_addr);
        noc_async_write_barrier();
        cb_pop_front(output_ind_cb_index, onetile);
    }

}
