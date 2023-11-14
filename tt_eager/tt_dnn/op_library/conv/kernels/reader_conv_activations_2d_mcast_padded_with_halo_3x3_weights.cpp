// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "debug_print.h"

auto s1 = SliceRange{ .h0 = 0, .h1 = 32, .hs = 1, .w0 = 0, .w1 = 1, .ws = 1 };

void kernel_main() {
    uint32_t i = 0;
    uint32_t conv_act_size_w = get_arg_val<uint32_t>(i); i+=1;
    uint32_t conv_act_size_h = get_arg_val<uint32_t>(i); i+=1;
    uint32_t conv_output_size_w = get_arg_val<uint32_t>(i); i+=1;
    uint32_t conv_output_size_h = get_arg_val<uint32_t>(i); i+=1;
    uint32_t weight_size_h = get_arg_val<uint32_t>(i); i+=1;
    uint32_t weight_size_w = get_arg_val<uint32_t>(i); i+=1;

    uint32_t act_block_h_datums = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_block_num_tiles = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_w_num_outer = get_arg_val<uint32_t>(i); i+=1;

    uint32_t first_partial_right_aligned_row_width = get_arg_val<uint32_t>(i); i+=1;
    uint32_t first_partial_image_num_rows          = get_arg_val<uint32_t>(i); i+=1;
    uint32_t num_full_images                       = get_arg_val<uint32_t>(i); i+=1;
    uint32_t last_partial_image_num_rows           = get_arg_val<uint32_t>(i); i+=1;
    uint32_t last_partial_left_aligned_row_width   = get_arg_val<uint32_t>(i); i+=1;

    uint32_t initial_skip                          = get_arg_val<uint32_t>(i); i+=1;
    uint32_t skip_after_partial_right_aligned_row  = get_arg_val<uint32_t>(i); i+=1;
    uint32_t skip_after_first_partial_image_row    = get_arg_val<uint32_t>(i); i+=1;
    uint32_t skip_after_full_image                 = get_arg_val<uint32_t>(i); i+=1;
    uint32_t skip_after_each_full_row              = get_arg_val<uint32_t>(i); i+=1;
    uint32_t skip_after_each_stick                 = get_arg_val<uint32_t>(i); i+=1;

    uint32_t window_outer                          = get_arg_val<uint32_t>(i); i+=1;
    uint32_t window_inner                          = get_arg_val<uint32_t>(i); i+=1;

    uint32_t noop = get_arg_val<uint32_t>(i); i+=1;
    if(noop) {
        return;
    }

    uint32_t act_mcast_dest_noc_start_x                  = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_mcast_dest_noc_start_y                  = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_mcast_dest_noc_end_x                    = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_mcast_dest_noc_end_y                    = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_mcast_num_dests                         = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_mcast_num_cores                         = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_mcast_sender_semaphore_addr             = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_mcast_receiver_semaphore_addr           = get_arg_val<uint32_t>(i); i+=1;

    uint32_t act_mcast_sender_size_bytes                 = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_mcast_sender_id                         = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_mcast_sender_noc_x                      = get_arg_val<uint32_t>(i); i+=1;
    volatile tt_l1_ptr uint32_t *act_mcast_sender_noc_y  = (volatile tt_l1_ptr uint32_t*)(get_arg_addr(i));


    constexpr bool act_in_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t stride_h = get_compile_time_arg_val(1);
    constexpr uint32_t stride_w = get_compile_time_arg_val(2);
    constexpr uint32_t conv_act_size_w_ = get_compile_time_arg_val(3);
    constexpr uint32_t conv_output_w_last_index = get_compile_time_arg_val(4) - 1;
    constexpr uint32_t conv_act_c_read_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t log_base_2_of_conv_act_size_c_bytes = get_compile_time_arg_val(6);

    constexpr uint32_t cb_id_act = 0;
    constexpr uint32_t cb_id_sharded_act = 3;
    constexpr uint32_t cb_id_sharded_act_mcast_receiver = 6;


    // Assumptions. Must be true. Validate on host.
    // assert(act_block_w_datums == C * weight_size_w)
    // assert(num_blocks_act_w == weight_size_h)
    // assert(act_block_w_datums % C == 0)
    // assert(act_block_w_datums % 32 == 0)
    // assert( % 32 == 0)
    // assert(act_block_h_ntiles == act_block_h_datums/32)
    // assert(act_block_w_ntiles == act_block_w_datums/32)
    // assert(act_block_num_tiles == (act_block_h_datums * act_block_w_datums)/1024)


    DPRINT << "--- initial skip: " << initial_skip << ENDL();
    DPRINT << "partial right aligned width: " << first_partial_right_aligned_row_width << ENDL();
    DPRINT << "--- skip after: " << skip_after_partial_right_aligned_row << ENDL();
    DPRINT << "first partial image rows: " << first_partial_image_num_rows << ENDL();
    DPRINT << "--- skip after: " << skip_after_first_partial_image_row << ENDL();
    DPRINT << "full images: " << num_full_images << ENDL();
    DPRINT << "--- skip after image: " << skip_after_full_image << ENDL();
    DPRINT << "--- skip after stick: " << skip_after_each_stick << ENDL();
    DPRINT << "--- skip after row: " << skip_after_each_full_row << ENDL();
    DPRINT << "last partial image rows: " << last_partial_image_num_rows << ENDL();
    DPRINT << "partial left aligned width: " << last_partial_left_aligned_row_width << ENDL();


    // DUMMY LOOP TO FILL READER INDICES
    constexpr uint32_t cb_reader_indices = tt::CB::c_in4;
    volatile tt_l1_ptr uint32_t* reader_indices_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_reader_indices));

    uint32_t weights_top_left_corner_idx = 0;
    uint32_t reader_idx = 0;

    // First partial right-aligned row
    for (uint32_t k = 0; k < first_partial_right_aligned_row_width; k++) {
        reader_indices_ptr[reader_idx++] = weights_top_left_corner_idx;
        weights_top_left_corner_idx += skip_after_each_stick;
    }
    weights_top_left_corner_idx += skip_after_partial_right_aligned_row; // Skip padded width

    // First partial image
    for (uint32_t j = 0; j < first_partial_image_num_rows; j++) {
        for (uint32_t k = 0; k < conv_output_size_w; k++) {
            reader_indices_ptr[reader_idx++] = weights_top_left_corner_idx;
            weights_top_left_corner_idx += skip_after_each_stick;
        }
        weights_top_left_corner_idx += skip_after_each_full_row;
    }
    weights_top_left_corner_idx += skip_after_first_partial_image_row; // Skip padded rows

    // Full images
    for (uint32_t i = 0; i < num_full_images; i++) {
        for (uint32_t j = 0; j < conv_output_size_h; j++) {
            for (uint32_t k = 0; k < conv_output_size_w; k++) {
                reader_indices_ptr[reader_idx++] = weights_top_left_corner_idx;
                weights_top_left_corner_idx += skip_after_each_stick;
            }
            weights_top_left_corner_idx += skip_after_each_full_row;
        }
        weights_top_left_corner_idx += skip_after_full_image; // Skip padded rows
    }

    // Last partial image
    for (uint32_t j = 0; j < last_partial_image_num_rows; j++) {
        for (uint32_t k = 0; k < conv_output_size_w; k++) {
            reader_indices_ptr[reader_idx++] = weights_top_left_corner_idx;
            weights_top_left_corner_idx += skip_after_each_stick;
        }
        weights_top_left_corner_idx += skip_after_each_full_row;
    }

    // Last partial left-alighted row
    for (uint32_t k = 0; k < last_partial_left_aligned_row_width; k++) {
        reader_indices_ptr[reader_idx++] = weights_top_left_corner_idx;
        weights_top_left_corner_idx += skip_after_each_stick;
    }

    //for (uint32_t k = 0; k < 160; k++) {
    //    DPRINT << reader_indices_ptr[k] << ENDL();
    //}


    // DUMMY LOOP TO FILL READER OFFSETS
    /* We can add another loop to read chunks of a stick as well.
     * - Duplicate reader_offset for same stick X times (window_inner must be 1)
     * - New loop between outer and inner that loops X times reading from same stick
     * - Read conv_act_c_read_bytes / X each time
     * - Update l1_write_addr_act by conv_act_c_read_bytes
     */
    constexpr uint32_t cb_reader_offsets = tt::CB::c_in5;
    volatile tt_l1_ptr uint32_t* reader_offsets_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_reader_offsets));
    uint32_t reader_offset = 0; // Constant offset for each pixel within filter window
    uint32_t reader_offset_idx = 0;
    for (uint32_t channel_stick_h = 0; channel_stick_h < weight_size_h; channel_stick_h++) {
        for (uint32_t channel_stick_w = 0; channel_stick_w < weight_size_w; channel_stick_w++) {
            reader_offsets_ptr[reader_offset_idx++] = reader_offset++;
        }
        // -1 to go back to previous reader_offset
        reader_offset += conv_act_size_w - 1; // Assuming (weight_size_w - 1) / 2 == pad_w
    }


    // Set ur local VALID value, to be mcasted to destinations flag address after the data has been mcasted
    volatile tt_l1_ptr uint32_t* act_mcast_receiver_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(act_mcast_receiver_semaphore_addr);
    noc_semaphore_set(act_mcast_receiver_semaphore_addr_ptr, VALID);
    // local address that will be atomically incremented by mcast receivers, to know when all receivers are ready
    // to receive the mcast
    volatile tt_l1_ptr uint32_t* act_mcast_sender_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(act_mcast_sender_semaphore_addr);

    uint64_t act_multicast_noc_addr = get_noc_multicast_addr(
        act_mcast_dest_noc_start_x,
        act_mcast_dest_noc_start_y,
        act_mcast_dest_noc_end_x,
        act_mcast_dest_noc_end_y,
        0
    );

    uint64_t act_mcast_receiver_semaphore_noc_addr = act_multicast_noc_addr | act_mcast_receiver_semaphore_addr;


    // TODO: need to make the read coalescing optimization cleaner
    // pass coalesce_window_inner_reads as a compile time arg and num_coalesced_reads so we can constexpr the if
    // currently works for the case of num_coalesced_reads == weight_size_w since these reads are contiguous on both src/dst side
    // we check if window_inner == weight_size_w to make sure coalescing is legal along full window_inner so the loop can be removed
    constexpr bool coalesce_window_inner_reads = true;
    constexpr uint32_t num_coalesced_reads = 3;
    const uint32_t coalesced_read_bytes = num_coalesced_reads * conv_act_c_read_bytes;
    // we want to have the check hoisted out because in act_block_h_datums loop it would be to expensive (unless we make it ifdef)
    uint32_t act_l1_offset = 0;
    uint32_t act_l1_read_addr = get_read_ptr(cb_id_sharded_act_mcast_receiver);

    for (uint32_t act_w_outer_i = 0; act_w_outer_i < act_w_num_outer; act_w_outer_i++) {
        if (act_w_outer_i == act_mcast_sender_id) {
            // MCAST SENDER: send entire sharded input to other cores in column
            // wait until all act mcast destinations have atomically incremented the act semaphore_addr (i.e. its value should be act_mcast_num_dests), then reset
            // the semaphore_addr value back to zero for the next block
            noc_semaphore_wait(act_mcast_sender_semaphore_addr_ptr, act_mcast_num_dests);
            noc_semaphore_set(act_mcast_sender_semaphore_addr_ptr, 0);

            // Now we have the block in the CB address, we can mcast to dests!
            uint32_t sharded_act_start_address = get_read_ptr(cb_id_sharded_act);
            uint64_t act_multicast_data_addr = act_multicast_noc_addr | act_l1_read_addr;
            // num_dests will source, since we are copying to a different local CB as well
            noc_async_write_multicast_loopback_src(sharded_act_start_address, act_multicast_data_addr, act_mcast_sender_size_bytes, act_mcast_num_cores + 1);

            // Note: no need for write barrier, since these two multicasts are done on the same noc id, same vc, same cmd_buf
            // Also, this only works because we are setting VCs statically (using NOC_CMD_STATIC_VC).

            // We should also multicast VALID flag to destinations for receiver semaphore
            noc_semaphore_set_multicast(act_mcast_receiver_semaphore_addr, act_mcast_receiver_semaphore_noc_addr, act_mcast_num_cores);

            noc_async_write_barrier();
        } else {
            // MCAST RECEIVER: receive entire sharded input from sender core
            // Set act semaphore value to INVALID
            noc_semaphore_set(act_mcast_receiver_semaphore_addr_ptr, INVALID);

            // Atomic increment source core counter
            uint64_t act_mcast_sender_semaphore_noc_addr = get_noc_addr(act_mcast_sender_noc_x, act_mcast_sender_noc_y[act_w_outer_i], act_mcast_sender_semaphore_addr);
            noc_semaphore_inc(act_mcast_sender_semaphore_noc_addr, 1);

            // wait on act semaphore value to become VALID (set by mcast sender after it multicasts data)
            noc_semaphore_wait(act_mcast_receiver_semaphore_addr_ptr, VALID);
        }

        for (uint32_t outer = 0; outer < window_outer; outer++) {
            // Reset reader_idx to finish act_block_h_datums
            reader_idx = 0;
            cb_reserve_back(cb_id_act, act_block_num_tiles);
            uint32_t l1_write_addr_act = get_write_ptr(cb_id_act);

            for (uint32_t bh = 0; bh < act_block_h_datums; bh++) {
                reader_offset_idx = 0;
                for (uint32_t inner = 0; inner < window_inner; inner++) {
                    // local read from reader_index + reader_offset;
                    act_l1_offset = (reader_indices_ptr[reader_idx] + reader_offsets_ptr[reader_offset_idx]) << log_base_2_of_conv_act_size_c_bytes;
                    noc_async_read_one_packet(get_noc_addr(act_l1_read_addr + act_l1_offset), l1_write_addr_act, coalesced_read_bytes);
                    l1_write_addr_act += coalesced_read_bytes;
                    reader_offset_idx += num_coalesced_reads;
                }
                reader_idx++;
            }
            noc_async_read_barrier();
            //if (act_w_outer_i == 0) {
            if (outer == 0) {
                // Print all 9 windows, full 5 tiles
                for (int window = 0; window < 9; window++) {
                    //DPRINT << "-------> Window: " << window << ENDL();
                for (int height = 0; height < 5; height++) {
                    //DPRINT << "------------> TILE: " << height << ENDL();
                    for (int i = 0; i < 9; i++) {
                        auto offset = (9 - (32 * i) % 9 + window) % 9;

                        auto s2_p = SliceRange{ .h0 = offset, .h1 = 32, .hs = 9, .w0 = 0, .w1 = 1, .ws = 1 };
                        //DPRINT << TileSlice(cb_id_act, height*9 + i, s2_p, true, false);
                    }
                }
                }
            }
            //}
            cb_push_back(cb_id_act, act_block_num_tiles);
        }
    }
}
