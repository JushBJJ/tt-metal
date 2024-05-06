// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ccl_common.hpp"

namespace tt {
namespace tt_metal {
namespace ccl {

void generate_edm_kernels_for_ring_or_linear_topology(
    tt_metal::Program &program,
    Device const* device,
    std::vector<ccl::EriscDatamoverBuilder> const& clockwise_edm_builders,
    std::vector<ccl::EriscDatamoverBuilder> const& counter_clockwise_edm_builders,
    std::optional<uint32_t> receiver_device_id,
    std::optional<uint32_t> sender_device_id,
    // TODO: move to linear/ring topology specific config
    uint32_t num_links,
    uint32_t ring_size,
    uint32_t ring_index,
    bool is_linear) {

    auto sender_noc = detail::GetPreferredNOCForDRAMRead(tt::Cluster::instance().arch());
    auto receiver_noc = detail::GetPreferredNOCForDRAMWrite(tt::Cluster::instance().arch());
    uint32_t sender_socket_idx = 0;
    uint32_t receiver_socket_idx = 0;
    if (receiver_device_id == sender_device_id) {
        if (ring_index == 0) {
            receiver_socket_idx = 1;
        } else {
            sender_socket_idx = 1;
        }
    }
    for (uint32_t i = 0; i < num_links; ++i) {
        bool is_clockwise_direction_edm_enabled = !is_linear || ring_index != ring_size - 1;
        if (is_clockwise_direction_edm_enabled) {
            auto eth_sender_core = device->get_ethernet_sockets(receiver_device_id.value()).at(sender_socket_idx);
            log_trace(tt::LogOp, "EDM CLOCKWISE KERNEL RT ARGS: ");
            auto eth_sender_kernel = ccl::generate_edm_kernel(
                program,
                device,
                clockwise_edm_builders.at(i),
                eth_sender_core,
                sender_noc);
            // eth_sender_kernels.push_back(eth_sender_kernel);
            log_trace(tt::LogOp, "RingIndex: {}. Link {}. Clockwise EDM Core (x={},y={})", ring_index, i, eth_sender_core.x, eth_sender_core.y);
        }

        bool is_counter_clockwise_direction_edm_enabled = !is_linear || ring_index != 0;
        if (is_counter_clockwise_direction_edm_enabled) {
            log_trace(tt::LogOp, "EDM COUNTER CLOCKWISE KERNEL RT ARGS: ");
            auto eth_receiver_core = device->get_ethernet_sockets(sender_device_id.value()).at(receiver_socket_idx);
            auto eth_receiver_kernel = ccl::generate_edm_kernel(
                program,
                device,
                counter_clockwise_edm_builders.at(i),
                eth_receiver_core,
                receiver_noc);
            log_trace(tt::LogOp, "RingIndex: {}. Link {}. Counter-clockwise EDM Core (x={},y={})", ring_index, i, eth_receiver_core.x, eth_receiver_core.y);
        }

        if (receiver_device_id == sender_device_id) {
            receiver_socket_idx += 2;
            sender_socket_idx += 2;
        } else {
            receiver_socket_idx += 1;
            sender_socket_idx += 1;
        }
    }

}

KernelHandle generate_edm_kernel(
    tt_metal::Program &program,
    Device const* device,
    ccl::EriscDatamoverBuilder const& edm_builder,
    CoreCoord const& eth_core,
    NOC noc_id) {
    log_trace(tt::LogOp, "EDM CLOCKWISE KERNEL RT ARGS: ");
    edm_builder.dump_to_log();

    // auto eth_sender_core = device->get_ethernet_sockets(receiver_device_id.value()).at(sender_socket_idx);

    std::vector<uint32_t> const& edm_clockwise_kernel_rt_args = edm_builder.emit_runtime_args();
    // Ethernet Kernels
    std::vector<uint32_t> eth_sender_ct_args = edm_builder.emit_compile_time_args();

    auto eth_sender_kernel = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/ccl/edm/erisc_datamover.cpp",
        eth_core,
        tt_metal::EthernetConfig{.noc=noc_id, .compile_args=eth_sender_ct_args});


    tt_metal::SetRuntimeArgs(
        program,
        eth_sender_kernel,
        eth_core,
        edm_clockwise_kernel_rt_args);

    // eth_sender_kernels.push_back(eth_sender_kernel);
    // log_trace(tt::LogOp, "RingIndex: {}. Link {}. Clockwise EDM Core (x={},y={})", ring_index, i, eth_sender_core.x, eth_sender_core.y);

    std::stringstream ss;
    ss << "EDM ARGS:\n";
    for (auto const& s : edm_clockwise_kernel_rt_args) {
        ss << "\t" << s << "\n";
    }
    log_trace(tt::LogOp, "{}", ss.str());

    return eth_sender_kernel;
}


ccl::EriscDatamoverBuilder create_erisc_datamover_builder(std::size_t num_channels, uint32_t page_size, ccl::EriscDataMoverBufferSharingMode buffer_sharing_mode) {

    std::vector<uint32_t> edm_sem_addresses(num_channels, 0);
    std::vector<uint32_t> edm_buffer_addresses(num_channels, 0);

    uint32_t edm_sem_addr = ccl::EriscDatamoverConfig::get_semaphores_base_address(num_channels);
    uint32_t edm_buffer_addr = ccl::EriscDatamoverConfig::get_buffers_base_address(num_channels);
    const uint32_t buffer_size = ccl::EriscDatamoverConfig::compute_buffer_size(num_channels, page_size);
    for (std::size_t c = 0; c < num_channels; ++c) {
        edm_sem_addresses.push_back(edm_sem_addr);
        edm_sem_addr += ccl::EriscDatamoverConfig::semaphore_size;
        edm_buffer_addresses.push_back(edm_buffer_addr);
        edm_buffer_addr += buffer_size;
        TT_ASSERT((c == 0) || (edm_buffer_addresses.back() != edm_buffer_addresses.front()));
        TT_ASSERT((c == 0) || (edm_sem_addresses.back() != edm_sem_addresses.front()));
    }

    return ccl::EriscDatamoverBuilder(
        buffer_size, ccl::EriscDatamoverConfig::get_edm_handshake_address(), edm_sem_addresses, edm_buffer_addresses, buffer_sharing_mode);
}

} // namespace ccl
} // namespace tt_metal
} // namespace tt
