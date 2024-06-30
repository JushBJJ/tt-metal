// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "eth_l1_address_map.h"
#include "tt_eager/tt_dnn/op_library/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "tt_eager/tt_dnn/op_library/ccl/edm/erisc_async_datamover.hpp"

// Args Schema:
// 1) handshake addr
// 2) sender channels offset (indicates for the erisc channels, where the senders start
//    so sender and receivers don't clash when paired with sender/receiver on the other
//    end of the link.)
// 3) sender num channels (How many erisc channels to use. ALso how many buffers to instantiate)
//    Informs how many times to iterate through the next group of args
//    4) sender_buffer_address
//    5) sender_num_messages_to_send
//    6) sender_channel_size
//    7) sender_semaphores_base_address
//    8) worker_semaphore_address
//    9) sender_num_workers
//       Informs how many worker X/Y coords to accept in the next loop. Each X/Y pair is 2 uint16s
//       10) worker_coord(s)
// ...
// Repeat from step 2 for receiver side

// Intended only for (performance) test use cases
FORCE_INLINE void eth_setup_handshake2(std::uint32_t handshake_register_address, bool is_sender) {
    if (is_sender) {
        DPRINT << "eth_send_bytes\n";
        eth_send_bytes(handshake_register_address, handshake_register_address, 16);
        DPRINT << "eth_wait_for_receiver_done\n";
        eth_wait_for_receiver_done();
    } else {
        DPRINT << "eth_wait_for_bytes\n";
        eth_wait_for_bytes(16);
        DPRINT << "wait eth_receiver_done\n";
        eth_receiver_channel_done(0);
    }
}

using tt::tt_metal::ccl::WorkerXY;

template<uint8_t num_senders, uint8_t num_receivers>
struct sender_receiver_index_t {
    static constexpr bool ZERO_SENDERS = num_senders == 0;
    static constexpr bool ZERO_RECEIVERS = num_receivers == 0;
    static constexpr bool NUM_SENDERS_IS_POW_2 = !ZERO_SENDERS && (((num_senders - 1) & num_senders) == 0);
    static constexpr bool NUM_RECEIVERS_IS_POW_2 = !ZERO_RECEIVERS && (((num_receivers - 1) & num_receivers) == 0);
    static constexpr uint16_t SENDER_INCR_MASK = !ZERO_SENDERS ? num_senders - 1 : 0;
    static constexpr uint16_t RECEIVER_INCR_MASK = !ZERO_RECEIVERS ? num_receivers - 1 : 0;
    static constexpr uint16_t COMBINED_INCR_MASK = SENDER_INCR_MASK << 8 | RECEIVER_INCR_MASK;
    static constexpr uint16_t COMBINED_INCR = (1 << 8) | 1;
    union {
        struct {
            uint8_t sender;
            uint8_t receiver;
        };
        uint16_t combined;
    } index;
    union {
        struct {
            uint8_t sender;
            uint8_t receiver;
        };
        uint16_t combined;
    } real_index;
    union {
        struct {
            uint8_t sender;
            uint8_t receiver;
        };
        uint16_t combined;
    } start;

    sender_receiver_index_t(uint8_t send_start, uint8_t receive_start, uint8_t num_send, uint8_t num_receive) {
        start.sender = send_start;
        start.receiver = receive_start;
        index.sender = 0;
        index.receiver = 0;
        real_index.sender = send_start;
        real_index.receiver = receive_start;
    }

    FORCE_INLINE void increment() {
        if constexpr (NUM_SENDERS_IS_POW_2 and NUM_RECEIVERS_IS_POW_2) {
            index.combined = (index.combined + COMBINED_INCR) & COMBINED_INCR_MASK;
            real_index.combined = start.combined + index.combined;
        } else if constexpr (ZERO_RECEIVERS and NUM_SENDERS_IS_POW_2) {
            index.sender = (index.sender + 1) & SENDER_INCR_MASK;
            real_index.sender = start.sender + index.sender;
        } else if constexpr (ZERO_SENDERS and NUM_RECEIVERS_IS_POW_2) {
            index.receiver = (index.receiver + 1) & RECEIVER_INCR_MASK;
            real_index.receiver = start.receiver + index.receiver;
        } else {
            index.combined += COMBINED_INCR;
            index.sender = index.sender >= num_senders ? 0 : index.sender;
            index.receiver = index.receiver >= num_receivers ? 0 : index.receiver;
            real_index.combined = start.combined + index.combined;
        }
    }
};

void kernel_main() {
    DPRINT << "START ERISC KERNEL" << ENDL();
    // COMPILE TIME ARGS
    // If true, will enable this erisc's sender functionality
    constexpr bool enable_sender_side = get_compile_time_arg_val(0) != 0;

    // If true, will enable this erisc's receiver functionality
    constexpr bool enable_receiver_side = get_compile_time_arg_val(1) != 0;

    constexpr uint32_t num_senders = get_compile_time_arg_val(2);
    constexpr uint32_t num_receivers = get_compile_time_arg_val(3);

    constexpr tt::tt_metal::ccl::EriscDataMoverBufferSharingMode edm_buffer_sharing_mode =
        static_cast<tt::tt_metal::ccl::EriscDataMoverBufferSharingMode>(get_compile_time_arg_val(4));

    constexpr tt::tt_metal::ccl::EriscDataMoverTerminationMode terminate_on_worker_signal =
        static_cast<tt::tt_metal::ccl::EriscDataMoverTerminationMode>(get_compile_time_arg_val(5));

    constexpr auto EDM_CONFIG = erisc::datamover::EriscDatamoverConfig<edm_buffer_sharing_mode, terminate_on_worker_signal>();
    using EDM_CONFIG_T = decltype(EDM_CONFIG);
    using ChannelBufferT = erisc::datamover::ChannelBuffer<EDM_CONFIG_T>;
    std::array<ChannelBufferT, eth_l1_mem::address_map::MAX_NUM_CONCURRENT_TRANSACTIONS> buffer_channels;

    //
    std::array<uint32_t, eth_l1_mem::address_map::MAX_NUM_CONCURRENT_TRANSACTIONS> printed_receiver_done;

    // SENDER ARGS
    uint32_t args_offset = 0;
    uint32_t handshake_addr = get_arg_val<uint32_t>(args_offset++);

    uint8_t const sender_channels_start = get_arg_val<uint32_t>(args_offset++);
    uint32_t const sender_num_channels = num_senders;//get_arg_val<uint32_t>(args_offset++);
    uint8_t num_senders_with_no_work = 0;
    for (uint32_t channel = 0; channel < sender_num_channels; channel++) {
        uint32_t const sender_buffer_address = get_arg_val<uint32_t>(args_offset++);
        uint32_t const sender_num_messages_to_send = get_arg_val<uint32_t>(args_offset++);
        // Each channel buffer is at buffer_base + (channel_id * sender_channel_size)
        // Each channel currently constrained to the same buffer size
        uint32_t const sender_channel_size = get_arg_val<uint32_t>(args_offset++);
        // The erisc's local l1 copy of the semaphore workers remotely increment
        uint32_t const sender_semaphores_base_address = get_arg_val<uint32_t>(args_offset++);
        // worker's semaphore L1 address
        const uint32_t worker_semaphore_address = get_arg_val<uint32_t>(args_offset++);
        const uint32_t sender_num_workers = get_arg_val<uint32_t>(args_offset++);
        const uint32_t workers_xy_list_addr = get_arg_addr(args_offset);
        args_offset += sender_num_workers;
        new (&buffer_channels[sender_channels_start + channel]) ChannelBufferT(
            sender_channels_start + channel,
            sender_buffer_address,
            sender_channel_size,
            worker_semaphore_address,
            sender_num_workers,
            sender_num_messages_to_send,
            (volatile tt_l1_ptr uint32_t *const)sender_semaphores_base_address,
            (const WorkerXY *)workers_xy_list_addr,
            true);
        if constexpr (terminate_on_worker_signal == EriscDataMoverTerminationMode::MESSAGE_COUNT_REACHED) {
            if (sender_num_messages_to_send == 0) {
                num_senders_with_no_work++;
            }
        }
    }
    DPRINT << "Setup reciever args" << ENDL();
    // Receiver args
    uint8_t const receiver_channels_start = get_arg_val<uint32_t>(args_offset++);
    uint32_t const receiver_num_channels = num_receivers;//get_arg_val<uint32_t>(args_offset++);
    uint8_t num_receivers_with_no_work = 0;
    for (uint32_t channel = 0; channel < receiver_num_channels; channel++) {
        uint32_t const receiver_buffers_base_address = get_arg_val<uint32_t>(args_offset++);
        uint32_t const receiver_num_messages_to_send = get_arg_val<uint32_t>(args_offset++);
        // Each channel buffer is at buffer_base + (channel_id * sender_channel_size)
        // Each channel currently constrained to the same buffer size
        uint32_t const receiver_channel_size = get_arg_val<uint32_t>(args_offset++);
        uint32_t const receiver_semaphores_base_address = get_arg_val<uint32_t>(args_offset++);
        uint32_t const worker_semaphore_address = get_arg_val<uint32_t>(args_offset++);
        uint32_t const receiver_num_workers = get_arg_val<uint32_t>(args_offset++);
        const uint32_t workers_xy_list_addr = get_arg_addr(args_offset);
        args_offset += receiver_num_workers;
        new (&buffer_channels[receiver_channels_start + channel]) ChannelBufferT(
            receiver_channels_start + channel,
            receiver_buffers_base_address,
            receiver_channel_size,
            worker_semaphore_address,
            receiver_num_workers,
            receiver_num_messages_to_send,
            (volatile tt_l1_ptr uint32_t *const)receiver_semaphores_base_address,
            (const WorkerXY *)workers_xy_list_addr,
            false);

        if constexpr (terminate_on_worker_signal == EriscDataMoverTerminationMode::MESSAGE_COUNT_REACHED) {
            if (receiver_num_messages_to_send == 0) {
                num_receivers_with_no_work++;
            }
        }
    }

    // Handshake with other erisc to make sure it's safe to start sending/receiving
    // Chose an arbitrary ordering mechanism to guarantee one of the erisc's will always be "sender" and the other
    // will always be "receiver" (only for handshake purposes)
    bool act_as_sender_in_handshake =
        (sender_channels_start < receiver_channels_start || receiver_num_channels == 0) && sender_num_channels > 0;
    DPRINT << "Start handhsake " << handshake_addr << " " << (uint32_t)act_as_sender_in_handshake << ENDL();
    erisc::datamover::eth_setup_handshake(handshake_addr, act_as_sender_in_handshake);
    DPRINT << "Done handshake" << ENDL();
    uint32_t eth_transaction_ack_word_addr = handshake_addr + 16;
    uint32_t eth_transaction_complete_addr = handshake_addr + 32;

    constexpr uint32_t SWITCH_INTERVAL = 4000000;
    uint32_t did_nothing_count = 0;

    uint32_t num_senders_complete = !enable_sender_side ? sender_num_channels : num_senders_with_no_work;
    uint32_t num_receivers_complete = !enable_receiver_side ? receiver_num_channels : num_receivers_with_no_work;
    bool senders_in_progress = num_senders_complete != sender_num_channels;
    bool receivers_in_progress = num_receivers_complete != receiver_num_channels;

    auto send_recv_index = sender_receiver_index_t<num_senders,num_receivers>(sender_channels_start, receiver_channels_start, sender_num_channels, receiver_num_channels);
    DPRINT << "IPR: " << (uint32_t)(senders_in_progress) << " " << (uint32_t)(receivers_in_progress) << ENDL();
    DPRINT << "RECIEVER: " << (uint32_t)(enable_receiver_side) << ENDL();
    while (senders_in_progress || receivers_in_progress) {
        // DPRINT << "In while loop" << ENDL();
        bool did_something_sender = false;
        bool did_something_receiver = false;

        uint32_t num_receivers_complete_old = num_receivers_complete;
        uint32_t num_senders_complete_old = num_senders_complete;
        //////////////////////////////////////
        // SENDER
        if constexpr (enable_sender_side) {
            ChannelBufferT &current_sender = buffer_channels[send_recv_index.real_index.sender];
            switch (current_sender.get_state()) {
                case ChannelBufferT::STATE::WAITING_FOR_WORKER:
                did_something_sender =
                    erisc::datamover::sender_noc_receive_payload_ack_check_sequence(current_sender, num_senders_complete);
                senders_in_progress = senders_in_progress && num_senders_complete != sender_num_channels;
                break;

                case ChannelBufferT::STATE::READY_FOR_ETH_TRANSFER:
                did_something_sender = erisc::datamover::sender_eth_send_data_sequence(current_sender);
                    break;

                case ChannelBufferT::STATE::SIGNALING_WORKER:
                did_something_sender = erisc::datamover::sender_notify_workers_if_buffer_available_sequence(
                                    current_sender, num_senders_complete);
                senders_in_progress = senders_in_progress && num_senders_complete != sender_num_channels;
                break;

                case ChannelBufferT::STATE::WAITING_FOR_ETH:
                did_something_sender =
                    erisc::datamover::sender_eth_check_receiver_ack_sequence(current_sender, num_senders_complete);
                senders_in_progress = senders_in_progress && num_senders_complete != sender_num_channels;
                break;

                default:
                break;
            };
        }

        //////////////////////////////////////
        // RECEIVER
        if constexpr (enable_receiver_side) {
            // DPRINT << "Reciever Mode" << ENDL();
            ChannelBufferT &current_receiver = buffer_channels[send_recv_index.real_index.receiver];

            switch (current_receiver.get_state()) {
                case ChannelBufferT::STATE::WAITING_FOR_ETH:
                DPRINT << "Accept payload" << ENDL();
                did_something_receiver = erisc::datamover::receiver_eth_accept_payload_sequence(current_receiver, num_receivers_complete, eth_transaction_ack_word_addr);
                receivers_in_progress = receivers_in_progress && num_receivers_complete != receiver_num_channels;
                DPRINT << "Done ACCEPT" << ENDL();
                break;

                case ChannelBufferT::STATE::SIGNALING_WORKER:
                // DPRINT << "Notify Worker" << ENDL();
                did_something_receiver =
                    erisc::datamover::receiver_eth_notify_workers_payload_available_sequence(current_receiver);
                // DPRINT << "Done notify worker" << ENDL();
                break;

                case ChannelBufferT::STATE::WAITING_FOR_WORKER:
                // DPRINT << "Wait for Worker" << ENDL();
                did_something_receiver = erisc::datamover::receiver_noc_read_worker_completion_check_sequence(
                                    current_receiver, num_receivers_complete, eth_transaction_complete_addr);
                receivers_in_progress = receivers_in_progress && num_receivers_complete != receiver_num_channels;
                // DPRINT << "Done Wait for Worker" << ENDL();
                break;

                default:
                break;
            };
        }
        send_recv_index.increment();
        //////////////////////////////////////

        // Enabling this block as is (with all the "did_something"s, seems to cause a loss of about
        // 0.5 GBps in throughput)
        if (did_something_sender || did_something_receiver) {
            did_nothing_count = 0;
        } else {
            if (did_nothing_count++ > SWITCH_INTERVAL) {
                did_nothing_count = 0;
                run_routing();
            }
        }
    }

    for (uint32_t s = 0; s < num_senders + num_receivers; s++ ) {
        auto const& channel = buffer_channels[s];
        // We need to explicitly check for channel send done because we may
        // advance sender channel state as soon as we receive an ack. Since we
        // may be the last active channel, and advance to done state just from ack
        // from the receiver ("I got a payload"), then we need to wait for done
        // at the very end here. Otherise if we invoke another erisc op back-to-back,
        // we may mess up transaction state because it's possible for receiver of this
        // op to send the completion done after that one has already started.
        uint32_t wait_count = 0;
        uint32_t wait_max = 50000;
        while(!channel.eth_is_receiver_channel_send_done()) {
            wait_count++;
            if (wait_count > wait_max) {

                DEBUG_STATUS("STK");
                run_routing();
                wait_count = 0;
            }
        }
    }
    DPRINT << "Done ERISC KERNEL" << ENDL();
    DEBUG_STATUS("DONE");
}
