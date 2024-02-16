// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

void eth_setup_handshake(std::uint32_t handshake_register_address, bool is_sender) {
    if (is_sender) {
        eth_send_bytes(handshake_register_address,handshake_register_address, 16);
        eth_wait_for_receiver_done();

        // eth_wait_for_bytes(16);
        // eth_receiver_done();
    } else {
        eth_wait_for_bytes(16);
        eth_receiver_done();

        // eth_send_bytes(handshake_register_address,handshake_register_address, 16);
        // eth_wait_for_receiver_done();
    }
}

void kernel_main() {
    std::uint32_t local_eth_l1_src_addr = get_arg_val<uint32_t>(0);
    std::uint32_t remote_eth_l1_dst_addr = get_arg_val<uint32_t>(1);
    std::uint32_t num_bytes_ = get_arg_val<uint32_t>(2);
    std::uint32_t num_loops_ = get_arg_val<uint32_t>(3);
    std::uint32_t num_sends_per_loop_ = get_arg_val<uint32_t>(4);

    constexpr uint32_t num_bytes_per_send = get_compile_time_arg_val(0);
    constexpr uint32_t num_bytes_per_send_word_size = get_compile_time_arg_val(1);

    constexpr std::uint32_t num_bytes = get_compile_time_arg_val(2);
    constexpr std::uint32_t num_loops = get_compile_time_arg_val(3);
    constexpr std::uint32_t num_sends_per_loop = get_compile_time_arg_val(4);


    eth_setup_handshake(remote_eth_l1_dst_addr, true);

    kernel_profiler::mark_time(10);
    uint32_t wrap_mask = num_sends_per_loop - 1;
    uint32_t j = 0;
    for (uint32_t i = 0; i < num_loops; i++) {
        kernel_profiler::mark_time(20);
        eth_send_bytes(
            local_eth_l1_src_addr /*+ (j * num_bytes)*/, remote_eth_l1_dst_addr /*+ (j * num_bytes)*/, num_bytes, num_bytes_per_send, num_bytes_per_send_word_size);
        kernel_profiler::mark_time(21);

        if (j == wrap_mask) {
            kernel_profiler::mark_time(22);
            eth_wait_for_receiver_done();
        }
        kernel_profiler::mark_time(23);
        j = (j + 1) & wrap_mask;
    }
    if (j != 0) {
        kernel_profiler::mark_time(24);
        eth_wait_for_receiver_done();
    }
    kernel_profiler::mark_time(11);
}
