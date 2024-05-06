#pragma once


#include "eth_l1_address_map.h"
#include "tensor/tensor_impl.hpp"
#include "tt_eager/tt_dnn/op_library/ccl/shared_with_host/hetergeneous_data_structs.hpp"



namespace tt {
namespace tt_metal {
namespace ccl {

enum Topology {
    Ring = 0,
    Linear = 1,
    Meash = 2
};


struct EriscDatamoverConfig {
    static constexpr std::size_t total_l1_buffer_space = eth_l1_mem::address_map::MAX_L1_LOADING_SIZE - eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;
    static constexpr std::size_t usable_l1_base_address = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;

    static constexpr std::size_t semaphore_size = 4;
    static constexpr std::size_t handshake_location_size = 16; // ethernet word size
    static constexpr std::size_t eth_word_size_bytes = 16;

    static uint32_t get_edm_handshake_address() {
        return usable_l1_base_address;
    }
    static uint32_t get_semaphores_base_address(std::size_t num_edm_channels) {
        return usable_l1_base_address + handshake_location_size;
    }
    static uint32_t get_buffers_base_address(std::size_t num_edm_channels) {
        uint32_t base_address = round_up(get_semaphores_base_address(num_edm_channels) + num_edm_channels * semaphore_size, eth_word_size_bytes);
        TT_ASSERT(base_address % eth_word_size_bytes == 0);
        return base_address;
    }
    static uint32_t compute_buffer_size(std::size_t num_edm_channels, uint32_t page_size = eth_word_size_bytes) {
        page_size = std::max<uint32_t>(page_size, eth_word_size_bytes);
        uint32_t buffer_size = round_down((total_l1_buffer_space - get_buffers_base_address(num_edm_channels)) / (num_edm_channels), page_size);
        TT_ASSERT(buffer_size > 0 && buffer_size % page_size == 0);
        return buffer_size;
    }
};



struct CCLOpConfig {
   public:
    CCLOpConfig(const Tensor& input_tensor, const Tensor &output_tensor) :
        input_sharded(input_tensor.is_sharded()),
        output_sharded(output_tensor.is_sharded()),
        page_size(input_tensor.buffer()->page_size()),
        input_shard_size_bytes(
            input_tensor.is_sharded() ?
                static_cast<std::optional<uint32_t>>((input_tensor.buffer()->page_size() * input_tensor.buffer()->shard_spec().tensor2d_shape[0] * input_tensor.buffer()->shard_spec().tensor2d_shape[1]) / input_tensor.shard_spec()->num_cores()) :
                std::nullopt),
        output_shard_size_bytes(
            output_tensor.is_sharded() ?
                static_cast<std::optional<uint32_t>>((output_tensor.buffer()->page_size() * output_tensor.buffer()->shard_spec().tensor2d_shape[0] * output_tensor.buffer()->shard_spec().tensor2d_shape[1]) / input_tensor.shard_spec()->num_cores()) :
                std::nullopt),
        shard_grid_size(input_tensor.shard_spec()->num_cores())
    {
        TT_ASSERT(!this->is_input_sharded() || input_shard_size_bytes.has_value());
        TT_ASSERT(!this->is_output_sharded() || output_shard_size_bytes.has_value());
    }

    uint32_t get_input_shard_size_bytes() const {
        TT_ASSERT(input_shard_size_bytes.has_value());
        return input_shard_size_bytes.value();
    }
    uint32_t get_output_shard_size_bytes() const {
        TT_ASSERT(output_shard_size_bytes.has_value());
        return output_shard_size_bytes.value();
    }
    uint32_t get_page_size() const {
        return page_size;
    }
    bool is_input_sharded() const {
        return input_sharded;
    }
    bool is_output_sharded() const {
        return output_sharded;
    }
    bool get_shard_grid_size() const {
        return shard_grid_size;
    }

   private:
    std::optional<uint32_t> input_shard_size_bytes; // TODO: split off into CCL op input config ()
    std::optional<uint32_t> output_shard_size_bytes; // TODO: split off into CCL op input config ()
    uint32_t page_size;
    uint32_t shard_grid_size;
    bool input_sharded;
    bool output_sharded;
};

class EriscDatamoverBuilder {
   public:
    struct ChannelBufferInterface {
        uint32_t eth_buffer_l1_address;
        uint32_t eth_semaphore_l1_address;
    };

    EriscDatamoverBuilder(uint32_t eth_buffer_size, uint32_t handshake_addr, std::vector<uint32_t> const& local_semaphore_addresses, std::vector<uint32_t> const& local_buffer_addresses, ccl::EriscDataMoverBufferSharingMode buffer_sharing_mode) :
        local_semaphore_addresses(local_semaphore_addresses),
        local_buffer_addresses(local_buffer_addresses),
        eth_buffer_size_bytes(eth_buffer_size),
        handshake_addr(handshake_addr),
        num_channel_buffers(local_buffer_addresses.size()),
        buffer_sharing_mode(buffer_sharing_mode),
        enable_sender(false),
        enable_receiver(false),
        num_senders(0),
        num_receivers(0)
        {
            TT_ASSERT(local_buffer_addresses.size() == local_semaphore_addresses.size());
            active_channels.reserve(num_channel_buffers);
            TT_ASSERT(eth_buffer_size_bytes < 163000);
            log_trace(tt::LogOp, "EriscDatamoverBuilder:");
            for (auto const& addr : local_semaphore_addresses) {
                log_trace(tt::LogOp, "\tsemaphore_address: {}", addr);
            }
            for (auto const& addr : local_buffer_addresses) {
                log_trace(tt::LogOp, "\tbuffer_address: {}", addr);
            }
        }

    // EriscDatamoverBuilder(AllGatherConfig const& all_gather_config, std::vector<uint32_t> const& local_semaphore_addresses, std::vector<uint32_t> const& local_buffer_addresses, ccl::EriscDataMoverBufferSharingMode buffer_sharing_mode) :
    //     local_semaphore_addresses(local_semaphore_addresses),
    //     local_buffer_addresses(local_buffer_addresses),
    //     eth_buffer_size_bytes(all_gather_config.get_eth_buffer_size()),
    //     handshake_addr(all_gather_config.get_erisc_handshake_address()),
    //     num_channel_buffers(all_gather_config.get_num_eth_buffers_per_edm()),
    //     buffer_sharing_mode(buffer_sharing_mode),
    //     enable_sender(false),
    //     enable_receiver(false),
    //     num_senders(0),
    //     num_receivers(0)
    //     {
    //         active_channels.reserve(num_channel_buffers);
    //         TT_ASSERT(eth_buffer_size_bytes < 163000);
    //         log_trace(tt::LogOp, "EriscDatamoverBuilder:");
    //         for (auto const& addr : local_semaphore_addresses) {
    //             log_trace(tt::LogOp, "\tsemaphore_address: {}", addr);
    //         }
    //         for (auto const& addr : local_buffer_addresses) {
    //             log_trace(tt::LogOp, "\tbuffer_address: {}", addr);
    //         }
    //     }

    [[nodiscard]]
    ChannelBufferInterface add_sender_channel(uint32_t worker_semaphore_address, uint32_t num_eth_messages_to_forward, std::vector<ccl::WorkerXY> const& worker_coords) {
        this->enable_sender = true;
        this->num_senders++;
        auto channel = active_channels.size();
        active_channels.emplace_back(true, worker_semaphore_address, num_eth_messages_to_forward, channel, worker_coords);
        log_trace(tt::LogOp, "Adding sender channel:");
        log_trace(tt::LogOp, "\tworker_semaphore_address: {}", active_channels.back().worker_semaphore_address);
        log_trace(tt::LogOp, "\tnum_eth_messages_to_forward: {}", active_channels.back().num_eth_messages_to_forward);
        log_trace(tt::LogOp, "\tchannel: {}", active_channels.back().channel);
        log_trace(tt::LogOp, "\tis_sender: {}", active_channels.back().is_sender ? 1 : 0);
        log_trace(tt::LogOp, "\tbuffer_address: {}", local_buffer_addresses.at(channel));
        log_trace(tt::LogOp, "\tsemaphore_address: {}", local_semaphore_addresses.at(channel));

        return ChannelBufferInterface{local_buffer_addresses.at(channel), local_semaphore_addresses.at(channel)};
    }
    [[nodiscard]]
    ChannelBufferInterface add_receiver_channel(uint32_t worker_semaphore_address, uint32_t num_eth_messages_to_forward, std::vector<ccl::WorkerXY> const& worker_coords) {
        this->enable_receiver = true;
        this->num_receivers++;
        auto channel = active_channels.size();
        active_channels.emplace_back(false, worker_semaphore_address, num_eth_messages_to_forward, channel, worker_coords);
        log_trace(tt::LogOp, "Adding receiver channel:");
        log_trace(tt::LogOp, "\tworker_semaphore_address: {}", active_channels.back().worker_semaphore_address);
        log_trace(tt::LogOp, "\tnum_eth_messages_to_forward: {}", active_channels.back().num_eth_messages_to_forward);
        log_trace(tt::LogOp, "\tchannel: {}", active_channels.back().channel);
        log_trace(tt::LogOp, "\tis_sender: {}", active_channels.back().is_sender ? 1 : 0);
        return ChannelBufferInterface{local_buffer_addresses.at(channel), local_semaphore_addresses.at(channel)};
    }

    [[nodiscard]]
    std::vector<uint32_t> emit_compile_time_args() const {
        return std::vector<uint32_t>{
            static_cast<uint32_t>(this->enable_sender ? 1 : 0),
            static_cast<uint32_t>(this->enable_receiver ? 1 : 0),
            this->num_senders,
            this->num_receivers,
            this->buffer_sharing_mode};
    }

    [[nodiscard]]
    std::vector<uint32_t> emit_runtime_args() const {
        std::vector<uint32_t> args;
        uint32_t size = 3 + active_channels.size() * 6;
        for (auto const& channel : active_channels) {
            size += channel.worker_coords.size();
        }
        args.reserve(size);

        // Handshake address
        args.push_back(handshake_addr);

        bool senders_below_receivers = active_channels.size() == 0 || this->active_channels.front().is_sender;

        // Sender channel args
        uint32_t sender_channels_offset = senders_below_receivers ? 0 : this->num_receivers;
        args.push_back(sender_channels_offset);
        for (auto const& channel : this->active_channels) {
            if (!channel.is_sender) {
                continue;
            }
            push_back_channel_args(args, channel);
        }

        // Receiver channel args
        uint32_t receiver_channels_offset = senders_below_receivers ? this->num_senders : 0;
        args.push_back(receiver_channels_offset);
        for (auto const& channel : this->active_channels) {
            if (channel.is_sender) {
                continue;
            }
            push_back_channel_args(args, channel);
        }

        return args;
    }

    void dump_to_log() const {
        auto const& rt_args = this->emit_runtime_args();
        log_trace(tt::LogOp, "EDM RT Args:");
        for (auto const& arg : rt_args) {
            log_trace(tt::LogOp, "\t{}", arg);
        }
    };

   private:
    struct ChannelBufferSpec {
        ChannelBufferSpec(
            bool is_sender,
            uint32_t worker_semaphore_address,
            uint32_t num_eth_messages_to_forward,
            uint32_t channel,
            std::vector<ccl::WorkerXY> const& worker_coords
        ) :
            worker_coords(worker_coords),
            worker_semaphore_address(worker_semaphore_address),
            num_eth_messages_to_forward(num_eth_messages_to_forward),
            channel(channel),
            is_sender(is_sender) {}

        std::vector<ccl::WorkerXY> const worker_coords;
        uint32_t worker_semaphore_address;
        uint32_t num_eth_messages_to_forward;
        uint32_t channel;
        bool is_sender;
    };

    void push_back_channel_args (std::vector<uint32_t> &args, ChannelBufferSpec const& channel) const {
        args.push_back(this->local_buffer_addresses.at(channel.channel));
        args.push_back(channel.num_eth_messages_to_forward);
        args.push_back(this->eth_buffer_size_bytes);
        args.push_back(this->local_semaphore_addresses.at(channel.channel));
        args.push_back(channel.worker_semaphore_address);
        args.push_back(channel.worker_coords.size());
        for (auto const& worker_coord : channel.worker_coords) {
            args.push_back(worker_coord.to_uint32());
        }
    }

    std::vector<ChannelBufferSpec> active_channels;
    std::vector<uint32_t> const local_semaphore_addresses;
    std::vector<uint32_t> const local_buffer_addresses;
    uint32_t eth_buffer_size_bytes;
    uint32_t handshake_addr;
    uint32_t const num_channel_buffers;
    ccl::EriscDataMoverBufferSharingMode const buffer_sharing_mode;
    uint32_t num_senders;
    uint32_t num_receivers;

    bool enable_sender;
    bool enable_receiver;
};

}; // namespace ccl
}; // namespace tt_metal
}; // namespace tt
