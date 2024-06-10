// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ckernel_include.h"
#include "ckernel_globals.h"
#include "ckernel.h"
#include "stream_interface.h"
#include "stream_io_map.h"
#include "hostdevcommon/common_runtime_address_map.h"
#include "llk_unpack_common_api.h"
#include "tools/profiler/kernel_profiler.hpp"

// #include "debug/dprint.h"

// MT: Temp extern declaration
extern uint32_t tiles_proc_delay;
extern uint32_t wait_tiles_cnt;
extern uint32_t apply_cnt;
extern uint32_t my_noc_x;
extern uint32_t my_noc_y;

using namespace ckernel;

// "llk_setup_operands" is the old function name that HLKC emits
inline void llk_setup_operands(bool apply_delay=false) {
    volatile tt_l1_ptr std::uint32_t* circular_buffer_config_addr = (volatile uint32_t*)(CIRCULAR_BUFFER_CONFIG_BASE);

    for (uint32_t cb_id = 0; cb_id < NUM_CIRCULAR_BUFFERS; cb_id++) {

        // NOTE: fifo_addr, fifo_size and fifo_limit in 16B words!
        uint32_t fifo_addr = circular_buffer_config_addr[0];
        uint32_t fifo_size = circular_buffer_config_addr[1];
        uint32_t fifo_num_pages = circular_buffer_config_addr[2]; // not used atm
        uint32_t fifo_page_size = circular_buffer_config_addr[3];

        cb_interface[cb_id].fifo_rd_ptr = fifo_addr;
        cb_interface[cb_id].fifo_size = fifo_size;
        cb_interface[cb_id].fifo_limit = fifo_addr + fifo_size;  // Check if there is overflow
        cb_interface[cb_id].tiles_acked = 0;
        cb_interface[cb_id].fifo_page_size = fifo_page_size;

        circular_buffer_config_addr += UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG; // move by 3 uint32's
    }
    // 1, 4, 7, 9
    // Identify noc coordinates
    uint32_t noc_id_reg = NOC_CMD_BUF_READ_REG(0, 0, NOC_NODE_ID);
    my_noc_x = noc_id_reg & NOC_NODE_ID_MASK;
    my_noc_y = (noc_id_reg >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;
    volatile uint32_t* dbg_noc_id = (volatile uint32_t*) 0x15200;
    *(dbg_noc_id+0) = ((my_noc_x&0xffff)) << 0 | ((my_noc_y&0xffff) << 16);
    if (apply_delay) {
        tiles_proc_delay = 6144*2; // Delay odd rows of cores
        // *(dbg_noc_id+1) = 0xdeda99;
        // switch (my_noc_y) {
        //     case 0:
        //     case 1: tiles_proc_delay = 0; break;
        //     case 2: tiles_proc_delay = 6144; break;
        //     case 3:
        //     case 4: tiles_proc_delay = 0; break;
        //     case 5: tiles_proc_delay = 6144; break;
        //     case 6:
        //     case 7:
        //     case 8: tiles_proc_delay = 0; break;
        //     case 9: tiles_proc_delay = 6144; break;
        //     case 10: tiles_proc_delay = 0; break;
        //     case 11: tiles_proc_delay = 6144; break;
        //     // case 1: tiles_proc_delay = 5120; break;
        //     // case 2: tiles_proc_delay = 0; break;
        //     // case 3:
        //     // case 4: tiles_proc_delay = 5120; break;
        //     // case 5: tiles_proc_delay = 0; break;
        //     // case 6:
        //     // case 7:
        //     // case 8: tiles_proc_delay = 0; break;
        //     // case 9: tiles_proc_delay = 5120; break;
        //     // case 10: tiles_proc_delay = 0; break;
        //     // case 11: tiles_proc_delay = 5120; break;
        // }
        // *(dbg_noc_id+2) = tiles_proc_delay;
    }
}

// Wait for N tiles available in the incoming stream
inline void llk_wait_tiles(int operand, std::int32_t num_tiles) {
    // TODO(MO): Manually uncomment until issue #6619 is resolved
    //DeviceZoneScopedSumN1("CB-COMPUTE-WAIT-FRONT");
    std::uint32_t input = operand;
    volatile tt_l1_ptr std::uint32_t * tiles_received_ptr = get_cb_tiles_received_ptr(operand);
    std::uint16_t num_tiles_u = (std::uint16_t)num_tiles;

    std::uint16_t tiles_received;

    uint16_t num_tiles_recv;
    uint64_t time_start = read_wall_clock();
    do {
        tiles_received = (std::uint16_t) reg_read((std::uint32_t)tiles_received_ptr);
        num_tiles_recv = tiles_received - cb_interface[input].tiles_acked;
    } while (num_tiles_recv < num_tiles_u);
    uint64_t time_end = read_wall_clock();
    uint64_t time_delta = time_end - time_start;
    wait_tiles_cnt++;
    if (time_delta > 0 && (wait_tiles_cnt==2)) {        // Apply delay only if second operand has arrived
        apply_cnt++;
        wait(tiles_proc_delay);

        // uint32_t noc_id_reg = NOC_CMD_BUF_READ_REG(0, 0, NOC_NODE_ID);
        // my_noc_x = noc_id_reg & NOC_NODE_ID_MASK;
        // my_noc_y = (noc_id_reg >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;
        // if (tiles_proc_delay != 0) {
        //      DPRINT << "wait " << my_noc_y << ", " << my_noc_x <<  ENDL();
        // }
    }
}

// Pop N tiles from the incoming stream
inline void llk_pop_tiles(
    const std::int32_t operand, const std::int32_t num_tiles, const std::int32_t block_c_dim = 0) {

    std::uint32_t input = operand;
    volatile tt_reg_ptr std::uint32_t* tiles_acked_ptr =
        (volatile std::uint32_t*)((((volatile std::uint32_t)get_cb_tiles_acked_ptr(operand)) >> 2) & 0x3ffff);
    std::uint32_t num_words = num_tiles * cb_interface[operand].fifo_page_size;

    cb_interface[input].tiles_acked += num_tiles;
    TT_SETDMAREG(0, cb_interface[input].tiles_acked, 0, LO_16(4));
    TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::UNPACK);
    TT_STOREREG(4, (std::uint32_t)&tiles_acked_ptr[0]);
    cb_interface[input].fifo_rd_ptr += num_words;

    if (cb_interface[input].fifo_rd_ptr >= cb_interface[input].fifo_limit) {
        cb_interface[input].fifo_rd_ptr -= cb_interface[input].fifo_size;
    }

    wait_tiles_cnt--;
}

inline void llk_wait_blocks(int operand, std::int32_t num_blocks) { llk_wait_tiles(operand, num_blocks); }


// FIXME-WH-UPLIFT
// FIXME: FP32 accumulation --> pop tiles in the operand? just change rd_ptr?
inline void llk_clear_tiles(std::uint32_t operand, std::uint32_t num_tiles) {
    // std::uint32_t input = operand_to_input_index(operand);
    // if (cb_interface[input].accumulation_buffer) {
    //     std::uint32_t num_words = num_tiles * cb_interface[input].fifo_page_size;

    //     cb_interface[input].fifo_rd_ptr += num_words;

    //     if (cb_interface[input].f.fifo_rd_ptr >= operands[input].fifo_limit) {
    //         cb_interface[input].f.fifo_rd_ptr -= operands[input].fifo_size;
    //     }

    //     cb_interface[input].f.fifo_rd_base_ptr = operands[input].fifo_rd_ptr; //inc base ptr

    //     cb_interface[input].curr_iter = 0;
    // }
}
