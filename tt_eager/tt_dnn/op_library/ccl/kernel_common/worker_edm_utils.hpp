// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/assert.h"
#include "tt_eager/tt_dnn/op_library/ccl/shared_with_host/hetergeneous_data_structs.hpp"

using tt::tt_metal::ccl::ShardType;
using tt::tt_metal::ccl::WorkerXY;

FORCE_INLINE void push_filler_pages_to_cb(const uint32_t& cb_id, uint32_t num_pages) {
    ASSERT(num_pages < cb_interface[cb_id].fifo_num_pages);
    cb_reserve_back(cb_id, num_pages);
    cb_push_back(cb_id, num_pages);
}
FORCE_INLINE void pop_filler_pages_from_cb(const uint32_t& cb_id, uint32_t num_pages) {
    ASSERT(num_pages < cb_interface[cb_id].fifo_num_pages);
    cb_wait_front(cb_id, num_pages);
    cb_pop_front(cb_id, num_pages);
}


FORCE_INLINE void fetch_chunk(
    const uint32_t& cb_id, const uint32_t& num_pages, const uint32_t& page_size, uint64_t remote_l1_read_addr) {
    cb_reserve_back(cb_id, num_pages);
    uint32_t l1_write_addr = get_write_ptr(cb_id);
    noc_async_read(remote_l1_read_addr, l1_write_addr, page_size * num_pages);
    noc_async_read_barrier();
    cb_push_back(cb_id, num_pages);
}
FORCE_INLINE void fetch_chunk_sharded(
    const uint32_t& cb_id, const uint32_t& num_pages, const uint32_t& page_size, uint64_t remote_l1_read_addr) {
    cb_reserve_back(cb_id, num_pages);
    uint32_t l1_write_addr = get_write_ptr(cb_id);
    noc_async_read(remote_l1_read_addr, l1_write_addr, num_pages * page_size);
    noc_async_read_barrier();
    cb_push_back(cb_id, num_pages);
}

FORCE_INLINE void send_chunk(
    const uint32_t& cb_id, const uint32_t& num_pages, const uint32_t& page_size, uint64_t remote_l1_write_addr) {
    cb_wait_front(cb_id, num_pages);
    uint32_t l1_read_addr = get_read_ptr(cb_id);
    noc_async_write(l1_read_addr, remote_l1_write_addr, page_size * num_pages);
    noc_async_write_barrier();
    cb_pop_front(cb_id, num_pages);
}
FORCE_INLINE void send_chunk_sharded(
    const uint32_t& cb_id, const uint32_t& num_pages, const uint32_t& page_size, uint64_t remote_l1_write_addr, uint64_t eth_l1_sender_semaphore_addr) {
    cb_wait_front(cb_id, num_pages);
    uint32_t l1_read_addr = get_read_ptr(cb_id);
    noc_async_write(l1_read_addr, remote_l1_write_addr, page_size * num_pages);
    noc_semaphore_inc(eth_l1_sender_semaphore_addr, 1);
    noc_async_write_barrier();
    cb_pop_front(cb_id, num_pages);
}
