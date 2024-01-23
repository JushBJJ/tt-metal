// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/dispatch/command_queue.hpp"

#include "debug_tools.hpp"
#include "noc/noc_parameters.h"
#include "tt_metal/detail/program.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/buffers/semaphore.hpp"
#include "tt_metal/impl/debug/dprint_server.hpp"
#include "tt_metal/impl/dispatch/dispatch_core_manager.hpp"
#include "tt_metal/third_party/umd/device/tt_xy_pair.h"
#include "dev_msgs.h"
#include <algorithm> // for copy() and assign()
#include <iterator> // for back_inserter

namespace tt::tt_metal {

#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"

EnqueueRestartCommand::EnqueueRestartCommand(
    uint32_t command_queue_channel,
    Device* device,
    SystemMemoryManager& manager
): command_queue_channel(command_queue_channel), manager(manager) {
    this->device = device;
}

const DeviceCommand EnqueueRestartCommand::assemble_device_command(uint32_t) {
    DeviceCommand cmd;
    cmd.set_restart();
    cmd.set_issue_queue_size(this->manager.get_issue_queue_size(this->command_queue_channel));
    cmd.set_completion_queue_size(this->manager.get_completion_queue_size(this->command_queue_channel));
    cmd.set_finish();
    return cmd;
}

void EnqueueRestartCommand::process() {
    uint32_t write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_channel);
    const DeviceCommand cmd = this->assemble_device_command(0);
    uint32_t cmd_size = DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND;
    this->manager.issue_queue_reserve_back(cmd_size, this->command_queue_channel);
    this->manager.cq_write(cmd.get_desc().data(), DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND, write_ptr);
    this->manager.issue_queue_push_back(cmd_size, false, this->command_queue_channel);
}

// EnqueueReadBufferCommandSection
EnqueueReadBufferCommand::EnqueueReadBufferCommand(
    uint32_t command_queue_id,
    Device* device,
    Buffer& buffer,
    void* dst,
    SystemMemoryManager& manager,
    uint32_t src_page_index,
    std::optional<uint32_t> pages_to_read) :
    command_queue_id(command_queue_id), dst(dst), manager(manager), buffer(buffer), src_page_index(src_page_index), pages_to_read(pages_to_read.has_value() ? pages_to_read.value() : buffer.num_pages()) {
    this->device = device;
}

const DeviceCommand EnqueueReadShardedBufferCommand::create_buffer_transfer_instruction(uint32_t dst_address, uint32_t padded_page_size, uint32_t num_pages) {
    DeviceCommand command;

    TT_ASSERT(is_sharded(this->buffer.buffer_layout()));
    uint32_t buffer_address = this->buffer.address();
    uint32_t dst_page_index = 0;

    uint32_t num_cores = this->buffer.num_cores();
    uint32_t shard_size = this->buffer.shard_spec().size();
    //TODO: for now all shards are same size of pages
    vector<uint32_t> num_pages_in_shards(num_cores, shard_size);
    vector<uint32_t> core_id_x;
    core_id_x.reserve(num_cores);
    vector<uint32_t> core_id_y;
    core_id_y.reserve(num_cores);
    auto all_cores = this->buffer.all_cores();
    for (const auto & core: all_cores) {
        CoreCoord physical_core = this->device->worker_core_from_logical_core(core);
        core_id_x.push_back(physical_core.x);
        core_id_y.push_back(physical_core.y);
    }
    command.add_buffer_transfer_sharded_instruction(
        buffer_address,
        dst_address,
        num_pages,
        padded_page_size,
        (uint32_t)this->buffer.buffer_type(),
        uint32_t(BufferType::SYSTEM_MEMORY),
        this->src_page_index,
        dst_page_index,
        num_pages_in_shards,
        core_id_x,
        core_id_y
    );

    command.set_sharded_buffer_num_cores(num_cores);
    return command;
}

const DeviceCommand EnqueueReadInterleavedBufferCommand::create_buffer_transfer_instruction(uint32_t dst_address, uint32_t padded_page_size, uint32_t num_pages) {
    DeviceCommand command;
    TT_ASSERT(not is_sharded(this->buffer.buffer_layout()));

    uint32_t buffer_address = this->buffer.address();
    uint32_t dst_page_index = 0;

    command.add_buffer_transfer_interleaved_instruction(
        buffer_address,
        dst_address,
        num_pages,
        padded_page_size,
        (uint32_t)this->buffer.buffer_type(),
        uint32_t(BufferType::SYSTEM_MEMORY),
        this->src_page_index,
        dst_page_index);

    command.set_sharded_buffer_num_cores(1);
    return command;
}

const DeviceCommand EnqueueReadBufferCommand::assemble_device_command(uint32_t dst_address) {
    uint32_t padded_page_size = align(this->buffer.page_size(), 32);
    uint32_t num_pages = this->pages_to_read;
    DeviceCommand command = this->create_buffer_transfer_instruction(dst_address, padded_page_size, num_pages);

    // Targeting fast dispatch on remote device means commands have to be tunneled through ethernet
    bool cmd_consumer_on_ethernet = not device->is_mmio_capable();
    uint32_t consumer_cb_num_pages = (get_consumer_data_buffer_size(cmd_consumer_on_ethernet) / padded_page_size);

    if (consumer_cb_num_pages >= 4) {
        consumer_cb_num_pages = (consumer_cb_num_pages / 4) * 4;
        command.set_producer_consumer_transfer_num_pages(consumer_cb_num_pages / 4);
    } else {
        command.set_producer_consumer_transfer_num_pages(1);
    }

    uint32_t consumer_cb_size = consumer_cb_num_pages * padded_page_size;
    TT_ASSERT(padded_page_size <= consumer_cb_size, "Page is too large to fit in consumer buffer");

    uint32_t producer_cb_num_pages = consumer_cb_num_pages * 2;
    uint32_t producer_cb_size = producer_cb_num_pages * padded_page_size;

    command.set_stall();
    command.set_page_size(padded_page_size);
    command.set_producer_cb_size(producer_cb_size);
    command.set_consumer_cb_size(consumer_cb_size);
    command.set_producer_cb_num_pages(producer_cb_num_pages);
    command.set_consumer_cb_num_pages(consumer_cb_num_pages);
    command.set_num_pages(num_pages);

    return command;
}

void EnqueueReadBufferCommand::process() {
    uint32_t write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
    this->read_buffer_addr = this->manager.get_completion_queue_read_ptr(this->command_queue_id);

    const DeviceCommand cmd = this->assemble_device_command(this->read_buffer_addr);

    this->manager.issue_queue_reserve_back(DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND, this->command_queue_id);
    this->manager.cq_write(cmd.get_desc().data(), DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND, write_ptr);
    this->manager.issue_queue_push_back(DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND, LAZY_COMMAND_QUEUE_MODE, this->command_queue_id);
}

// EnqueueWriteBufferCommand section
EnqueueWriteBufferCommand::EnqueueWriteBufferCommand(
    uint32_t command_queue_id,
    Device* device,
    Buffer& buffer,
    const void* src,
    SystemMemoryManager& manager,
    uint32_t dst_page_index,
    std::optional<uint32_t> pages_to_write) :
    command_queue_id(command_queue_id), manager(manager), src(src), buffer(buffer), dst_page_index(dst_page_index), pages_to_write(pages_to_write.has_value() ? pages_to_write.value() : buffer.num_pages()) {
    TT_ASSERT(
        buffer.buffer_type() == BufferType::DRAM or buffer.buffer_type() == BufferType::L1,
        "Trying to write to an invalid buffer");
    this->device = device;
}


const DeviceCommand EnqueueWriteInterleavedBufferCommand::create_buffer_transfer_instruction(uint32_t src_address, uint32_t padded_page_size, uint32_t num_pages) {
    DeviceCommand command;

    TT_ASSERT(not is_sharded(this->buffer.buffer_layout()));

    uint32_t buffer_address = this->buffer.address();
    uint32_t src_page_index = 0;
    command.add_buffer_transfer_interleaved_instruction(
        src_address,
        buffer_address,
        num_pages,
        padded_page_size,
        (uint32_t) BufferType::SYSTEM_MEMORY,
        (uint32_t) this->buffer.buffer_type(),
        src_page_index,
        this->dst_page_index
    );
    return command;

}

const DeviceCommand EnqueueWriteShardedBufferCommand::create_buffer_transfer_instruction(uint32_t src_address, uint32_t padded_page_size, uint32_t num_pages) {
    DeviceCommand command;

    TT_ASSERT(is_sharded(this->buffer.buffer_layout()));
    uint32_t buffer_address = this->buffer.address();
    uint32_t src_page_index = 0;

    uint32_t num_cores = this->buffer.num_cores();
    uint32_t shard_size = this->buffer.shard_spec().size();
    //TODO: for now all shards are same size of pages
    vector<uint32_t> num_pages_in_shards(num_cores, shard_size);
    vector<uint32_t> core_id_x;
    core_id_x.reserve(num_cores);
    vector<uint32_t> core_id_y;
    core_id_y.reserve(num_cores);
    auto all_cores = this->buffer.all_cores();
    for (const auto & core: all_cores) {
        CoreCoord physical_core = this->device->worker_core_from_logical_core(core);
        core_id_x.push_back(physical_core.x);
        core_id_y.push_back(physical_core.y);
    }
    command.add_buffer_transfer_sharded_instruction(
        src_address,
        buffer_address,
        num_pages,
        padded_page_size,
        (uint32_t) BufferType::SYSTEM_MEMORY,
        (uint32_t) this->buffer.buffer_type(),
        src_page_index,
        this->dst_page_index,
        num_pages_in_shards,
        core_id_x,
        core_id_y
    );

    command.set_sharded_buffer_num_cores(num_cores);

    return command;
}



const DeviceCommand EnqueueWriteBufferCommand::assemble_device_command(uint32_t src_address) {
    uint32_t num_pages = this->pages_to_write;
    uint32_t padded_page_size = this->buffer.page_size();
    if (this->buffer.page_size() != this->buffer.size()) { // should buffer.size() be num_pages * page_size
        padded_page_size = align(this->buffer.page_size(), 32);
    }

    // Targeting fast dispatch on remote device means commands have to be tunneled through ethernet
    bool cmd_consumer_on_ethernet = not device->is_mmio_capable();
    uint32_t consumer_cb_num_pages = (get_consumer_data_buffer_size(cmd_consumer_on_ethernet) / padded_page_size);
    DeviceCommand command = this->create_buffer_transfer_instruction(src_address, padded_page_size, num_pages);

    if (consumer_cb_num_pages >= 4) {
        consumer_cb_num_pages = (consumer_cb_num_pages / 4) * 4;
        command.set_producer_consumer_transfer_num_pages(consumer_cb_num_pages / 4);
    } else {
        command.set_producer_consumer_transfer_num_pages(1);
    }

    uint32_t consumer_cb_size = consumer_cb_num_pages * padded_page_size;
    TT_ASSERT(padded_page_size <= consumer_cb_size, "Page is too large to fit in consumer buffer");
    uint32_t producer_cb_num_pages = consumer_cb_num_pages * 2;
    uint32_t producer_cb_size = producer_cb_num_pages * padded_page_size;

    command.set_page_size(padded_page_size);
    command.set_producer_cb_size(producer_cb_size);
    command.set_consumer_cb_size(consumer_cb_size);
    command.set_producer_cb_num_pages(producer_cb_num_pages);
    command.set_consumer_cb_num_pages(consumer_cb_num_pages);
    command.set_num_pages(num_pages);


    command.set_data_size(padded_page_size * num_pages);
    return command;
}

void EnqueueWriteBufferCommand::process() {
    uint32_t write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
    uint32_t system_memory_temporary_storage_address = write_ptr + DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND;

    const DeviceCommand cmd = this->assemble_device_command(system_memory_temporary_storage_address);
    uint32_t data_size_in_bytes = cmd.get_data_size();

    uint32_t cmd_size = DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND + data_size_in_bytes;
    this->manager.issue_queue_reserve_back(cmd_size, this->command_queue_id);

    this->manager.cq_write(cmd.get_desc().data(), DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND, write_ptr);
    uint32_t unpadded_src_offset = this->dst_page_index * this->buffer.page_size();

    if (this->buffer.page_size() % 32 != 0 and this->buffer.page_size() != this->buffer.size()) {
        // If page size is not 32B-aligned, we cannot do a contiguous write
        uint32_t src_address_offset = unpadded_src_offset;
        uint32_t padded_page_size = align(this->buffer.page_size(), 32);
        for (uint32_t sysmem_address_offset = 0; sysmem_address_offset < data_size_in_bytes; sysmem_address_offset += padded_page_size) {
            this->manager.cq_write((char*)this->src + src_address_offset, this->buffer.page_size(), system_memory_temporary_storage_address + sysmem_address_offset);
            src_address_offset += this->buffer.page_size();
        }
    } else {
        this->manager.cq_write((char*)this->src + unpadded_src_offset, data_size_in_bytes, system_memory_temporary_storage_address);
    }

    this->manager.issue_queue_push_back(cmd_size, LAZY_COMMAND_QUEUE_MODE, this->command_queue_id);

    auto cmd_desc = cmd.get_desc();
}

EnqueueProgramCommand::EnqueueProgramCommand(
    uint32_t command_queue_id,
    Device* device,
    Buffer& buffer,
    tt::tt_metal::detail::ProgramMap& program_to_dev_map,
    SystemMemoryManager& manager,
    const Program& program,
    bool stall,
    std::optional<std::reference_wrapper<Trace>> trace
    ) :
    command_queue_id(command_queue_id), buffer(buffer), program_to_dev_map(program_to_dev_map), manager(manager), program(program), stall(stall) {
    this->device = device;
    this->trace = trace;
}

const DeviceCommand EnqueueProgramCommand::assemble_device_command(uint32_t host_data_src) {
    DeviceCommand command;
    command.set_num_workers(this->program_to_dev_map.num_workers);

    auto populate_program_data_transfer_instructions =
        [&command](const vector<uint32_t>& num_transfers_per_page, const vector<tt::tt_metal::detail::transfer_info>& transfers_in_pages) {
            uint32_t i = 0;
            for (uint32_t j = 0; j < num_transfers_per_page.size(); j++) {
                uint32_t num_transfers_in_page = num_transfers_per_page[j];
                command.write_program_entry(num_transfers_in_page);
                for (uint32_t k = 0; k < num_transfers_in_page; k++) {
                    const auto [num_bytes, dst, dst_noc, num_receivers, last_multicast_in_group, linked] = transfers_in_pages[i];
                    command.add_write_page_partial_instruction(num_bytes, dst, dst_noc, num_receivers, last_multicast_in_group, linked);
                    i++;
                }
            }
        };

    command.set_is_program();

    // Not used, since we specified that this is a program command, and the consumer just looks at the write program
    // info
    constexpr static uint32_t dummy_dst_addr = 0;
    constexpr static uint32_t dummy_buffer_type = 0;
    uint32_t num_runtime_arg_pages = this->program_to_dev_map.num_transfers_in_runtime_arg_pages.size();
    uint32_t num_cb_config_pages = this->program_to_dev_map.num_transfers_in_cb_config_pages.size();
    uint32_t num_program_binary_pages = this->program_to_dev_map.num_transfers_in_program_pages.size();
    uint32_t num_go_signal_pages = this->program_to_dev_map.num_transfers_in_go_signal_pages.size();
    uint32_t num_host_data_pages = num_runtime_arg_pages + num_cb_config_pages;
    uint32_t num_cached_pages = num_program_binary_pages + num_go_signal_pages;
    uint32_t total_num_pages = num_host_data_pages + num_cached_pages;

    command.set_page_size(DeviceCommand::PROGRAM_PAGE_SIZE);
    command.set_num_pages(DeviceCommand::TransferType::RUNTIME_ARGS, num_runtime_arg_pages);
    command.set_num_pages(DeviceCommand::TransferType::CB_CONFIGS, num_cb_config_pages);
    command.set_num_pages(DeviceCommand::TransferType::PROGRAM_PAGES, num_program_binary_pages);
    command.set_num_pages(DeviceCommand::TransferType::GO_SIGNALS, num_go_signal_pages);
    command.set_num_pages(total_num_pages);

    command.set_data_size(
        DeviceCommand::PROGRAM_PAGE_SIZE *
        num_host_data_pages);

    const uint32_t page_index_offset = 0;
    if (num_host_data_pages) {
        command.add_buffer_transfer_interleaved_instruction(
            host_data_src,
            dummy_dst_addr,
            num_host_data_pages,
            DeviceCommand::PROGRAM_PAGE_SIZE,
            uint32_t(BufferType::SYSTEM_MEMORY),
            dummy_buffer_type, page_index_offset, page_index_offset);

        if (num_runtime_arg_pages) {
            populate_program_data_transfer_instructions(
                this->program_to_dev_map.num_transfers_in_runtime_arg_pages, this->program_to_dev_map.runtime_arg_page_transfers);
        }

        if (num_cb_config_pages) {
            populate_program_data_transfer_instructions(
                this->program_to_dev_map.num_transfers_in_cb_config_pages, this->program_to_dev_map.cb_config_page_transfers);
        }
    }

    if (num_cached_pages) {
        command.add_buffer_transfer_interleaved_instruction(
            this->buffer.address(),
            dummy_dst_addr,
            num_cached_pages,
            DeviceCommand::PROGRAM_PAGE_SIZE,
            uint32_t(this->buffer.buffer_type()),
            dummy_buffer_type, page_index_offset, page_index_offset);

        if (num_program_binary_pages) {
            populate_program_data_transfer_instructions(
                this->program_to_dev_map.num_transfers_in_program_pages, this->program_to_dev_map.program_page_transfers);
        }

        if (num_go_signal_pages) {
            populate_program_data_transfer_instructions(
                this->program_to_dev_map.num_transfers_in_go_signal_pages, this->program_to_dev_map.go_signal_page_transfers);
        }
    }

    // TODO (abhullar): deduce whether the producer is on ethernet core rather than hardcoding assuming tensix worker
    const uint32_t producer_cb_num_pages = (get_producer_data_buffer_size(/*use_eth_l1=*/false) / DeviceCommand::PROGRAM_PAGE_SIZE);
    const uint32_t producer_cb_size = producer_cb_num_pages * DeviceCommand::PROGRAM_PAGE_SIZE;

    // Targeting fast dispatch on remote device means commands have to be tunneled through ethernet
    bool cmd_consumer_on_ethernet = not device->is_mmio_capable();
    const uint32_t consumer_cb_num_pages = (get_consumer_data_buffer_size(cmd_consumer_on_ethernet) / DeviceCommand::PROGRAM_PAGE_SIZE);
    const uint32_t consumer_cb_size = consumer_cb_num_pages * DeviceCommand::PROGRAM_PAGE_SIZE;

    command.set_producer_cb_size(producer_cb_size);
    command.set_consumer_cb_size(consumer_cb_size);
    command.set_producer_cb_num_pages(producer_cb_num_pages);
    command.set_consumer_cb_num_pages(consumer_cb_num_pages);

    // Should only ever be set if we are
    // enqueueing a program immediately
    // after writing it to a buffer
    if (this->stall) {
        command.set_stall();
    }

    // This needs to be quite small, since programs are small
    command.set_producer_consumer_transfer_num_pages(4);

    return command;
}

void EnqueueProgramCommand::process() {
    uint32_t write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
    uint32_t system_memory_temporary_storage_address = write_ptr + DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND;

    const DeviceCommand cmd = this->assemble_device_command(system_memory_temporary_storage_address);

    uint32_t data_size_in_bytes = cmd.get_data_size();
    const uint32_t cmd_size = DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND + data_size_in_bytes;
    this->manager.issue_queue_reserve_back(cmd_size, this->command_queue_id);
    this->manager.cq_write(cmd.get_desc().data(), DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND, write_ptr);

    bool tracing = this->trace.has_value();
    vector<uint32_t> trace_host_data;
    uint32_t start_addr = system_memory_temporary_storage_address;
    constexpr static uint32_t padding_alignment = 16;
    for (size_t kernel_id = 0; kernel_id < this->program.num_kernels(); kernel_id++) {
        Kernel* kernel = detail::GetKernel(program, kernel_id);
        for (const auto& c: kernel->cores_with_runtime_args()) {
            const auto & core_runtime_args = kernel->runtime_args(c);
            this->manager.cq_write(core_runtime_args.data(), core_runtime_args.size() * sizeof(uint32_t), system_memory_temporary_storage_address);
            system_memory_temporary_storage_address = align(system_memory_temporary_storage_address + core_runtime_args.size() * sizeof(uint32_t), padding_alignment);

            if (tracing) {
                trace_host_data.insert(trace_host_data.end(), core_runtime_args.begin(), core_runtime_args.end());
                trace_host_data.resize(align(trace_host_data.size(), padding_alignment / sizeof(uint32_t)));
            }
        }
    }

    system_memory_temporary_storage_address = start_addr + align(system_memory_temporary_storage_address - start_addr, DeviceCommand::PROGRAM_PAGE_SIZE);

    array<uint32_t, 4> cb_data;
    for (const shared_ptr<CircularBuffer>& cb : program.circular_buffers()) {
        for (const auto buffer_index : cb->buffer_indices()) {
            cb_data = {cb->address() >> 4, cb->size() >> 4, cb->num_pages(buffer_index), cb->size() / cb->num_pages(buffer_index) >> 4};
            this->manager.cq_write(cb_data.data(), padding_alignment, system_memory_temporary_storage_address);
            system_memory_temporary_storage_address += padding_alignment;
            if (tracing) {
                // No need to resize since cb_data size is guaranteed to be 16 bytes
                trace_host_data.insert(trace_host_data.end(), cb_data.begin(), cb_data.end());
            }
        }
    }

    this->manager.issue_queue_push_back(cmd_size, LAZY_COMMAND_QUEUE_MODE, this->command_queue_id);
    if (tracing) {
        Trace::TraceNode trace_node = {.command = cmd, .data = trace_host_data, .command_type = this->type(), .num_data_bytes = cmd.get_data_size()};
        Trace& trace_ = trace.value();
        trace_.record(trace_node);
    }
}

FinishCommand::FinishCommand(uint32_t command_queue_id, Device* device, SystemMemoryManager& manager) : command_queue_id(command_queue_id), manager(manager) { this->device = device; }

const DeviceCommand FinishCommand::assemble_device_command(uint32_t) {
    DeviceCommand command;
    command.set_finish();
    return command;
}

void FinishCommand::process() {
    uint32_t write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
    const DeviceCommand cmd = this->assemble_device_command(0);
    uint32_t cmd_size = DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND;
    this->manager.issue_queue_reserve_back(cmd_size, this->command_queue_id);
    this->manager.cq_write(cmd.get_desc().data(), DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND, write_ptr);
    this->manager.issue_queue_push_back(cmd_size, false, this->command_queue_id);
}

// EnqueueWrapCommand section
EnqueueWrapCommand::EnqueueWrapCommand(uint32_t command_queue_id, Device* device, SystemMemoryManager& manager, DeviceCommand::WrapRegion wrap_region) : command_queue_id(command_queue_id), manager(manager), wrap_region(wrap_region) {
    this->device = device;
}

const DeviceCommand EnqueueWrapCommand::assemble_device_command(uint32_t) {
    DeviceCommand command;
    command.set_wrap(this->wrap_region);
    return command;
}

void EnqueueWrapCommand::process() {
    uint32_t write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
    uint32_t space_left_in_bytes = this->manager.get_issue_queue_limit(this->command_queue_id) - write_ptr;
    // There may not be enough space in the issue queue to submit another command
    // In that case we write as big of a vector as we can with the wrap index (0) set to wrap type
    // To ensure that the issue queue write pointer does wrap, we need the wrap packet to be the full size of the issue queue
    uint32_t wrap_packet_size_bytes = std::min(space_left_in_bytes, DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND);

    const DeviceCommand cmd = this->assemble_device_command(0);
    this->manager.issue_queue_reserve_back(wrap_packet_size_bytes, this->command_queue_id);
    this->manager.cq_write(cmd.get_desc().data(), wrap_packet_size_bytes, write_ptr);
    if (this->wrap_region == DeviceCommand::WrapRegion::COMPLETION) {
        // Wrap the read pointers for completion queue because device will start writing data at head of completion queue and there are no more reads to be done at current completion queue write pointer
        // If we don't wrap the read then the subsequent read buffer command may attempt to read past the total command queue size
        // because the read buffer command will see updated write pointer to compute num pages to read but the local read pointer is pointing to tail of completion queue
        this->manager.wrap_completion_queue_rd_ptr(this->command_queue_id);
        this->manager.issue_queue_push_back(wrap_packet_size_bytes, LAZY_COMMAND_QUEUE_MODE, this->command_queue_id);
    } else {
        this->manager.wrap_issue_queue_wr_ptr(this->command_queue_id);
    }
}

// CommandQueue section
CommandQueue::CommandQueue(Device* device, uint32_t id): manager(*device->manager) {
    this->device = device;
    this->id = id;

    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device->id());
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device->id());
    this->size_B = tt::Cluster::instance().get_host_channel_size(mmio_device_id, channel) / device->num_hw_cqs();

    tt_cxy_pair issue_q_reader_location = dispatch_core_manager::get(device->num_hw_cqs()).issue_queue_reader_core(device->id(), channel, this->id);
    tt_cxy_pair completion_q_writer_location = dispatch_core_manager::get(device->num_hw_cqs()).completion_queue_writer_core(device->id(), channel, this->id);

    this->issue_queue_reader_core = CoreCoord(issue_q_reader_location.x, issue_q_reader_location.y);
    this->completion_queue_writer_core = CoreCoord(completion_q_writer_location.x, completion_q_writer_location.y);
}

CommandQueue::~CommandQueue() {}

void CommandQueue::enqueue_command(Command& command, bool blocking) {
    // For the time-being, doing the actual work of enqueing in
    // the main thread.
    // TODO(agrebenisan): Perform the following in a worker thread
    command.process();

    if (blocking) {
        this->finish();
    }
}


//TODO: Currently converting page ordering from interleaved to sharded and then doing contiguous read/write
// Look into modifying command to do read/write of a page at a time to avoid doing copy
void convert_interleaved_to_sharded_on_host(const void * host,
                                        const Buffer & buffer,
                                        bool read=false) {

    const uint32_t num_pages = buffer.num_pages();
    const uint32_t page_size = buffer.page_size();

    const uint32_t size_in_bytes = num_pages * page_size;

    void * temp = malloc(size_in_bytes);
    memcpy(temp, host, size_in_bytes);

    const void * dst = host;
    std::set<uint32_t> pages_seen;
    for (uint32_t host_page_id = 0; host_page_id < num_pages; host_page_id++) {
        auto dev_page_id = buffer.get_mapped_page_id(host_page_id);

        TT_ASSERT(dev_page_id < num_pages and dev_page_id >= 0);
        if (read) {
            memcpy((char* )dst + dev_page_id*page_size,
                (char *)temp + host_page_id*page_size,
                page_size
                );
        }
        else {
            memcpy((char* )dst + host_page_id*page_size,
                (char *)temp + dev_page_id*page_size,
                page_size
                );
        }
    }
    free(temp);
}

// Read buffer command is enqueued in the issue region and device writes requested buffer data into the completion region
void CommandQueue::enqueue_read_buffer(Buffer& buffer, void* dst, bool blocking) {
    ZoneScopedN("CommandQueue_read_buffer");
    TT_FATAL(blocking, "EnqueueReadBuffer only has support for blocking mode currently");

    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(this->device->id());
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->device->id());
    uint32_t read_buffer_command_size = DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND;

    uint32_t padded_page_size = align(buffer.page_size(), 32);
    uint32_t total_pages_to_read = buffer.num_pages();
    uint32_t unpadded_dst_offset = 0;
    uint32_t src_page_index = 0;
    while (total_pages_to_read > 0) {
        if ((this->manager.get_issue_queue_write_ptr(this->id)) + read_buffer_command_size >= this->manager.get_issue_queue_limit(this->id)) {
            this->wrap(DeviceCommand::WrapRegion::ISSUE, blocking);
        }

        const uint32_t command_completion_limit = this->manager.get_completion_queue_limit(this->id);
        uint32_t num_pages_available = (command_completion_limit - get_cq_completion_wr_ptr<false>(this->device->id(), this->id, this->size_B)) / padded_page_size;
        uint32_t pages_to_read = std::min(total_pages_to_read, num_pages_available);
        if (pages_to_read == 0) {
            // Wrap the completion region because a single page won't fit in available space
            // Wrap needs to be blocking because host needs updated write pointer to compute how many pages can be read
            this->wrap(DeviceCommand::WrapRegion::COMPLETION, true);
            num_pages_available = (command_completion_limit - get_cq_completion_wr_ptr<false>(this->device->id(), this->id, this->size_B)) / padded_page_size;
            pages_to_read = std::min(total_pages_to_read, num_pages_available);
        }

        tt::log_debug(tt::LogDispatch, "EnqueueReadBuffer for channel {}", this->id);
        uint32_t command_read_buffer_addr;
        if (is_sharded(buffer.buffer_layout())) {
            auto command = EnqueueReadShardedBufferCommand(this->id, this->device, buffer, dst, this->manager, src_page_index, pages_to_read);
            this->enqueue_command(command, blocking);
            command_read_buffer_addr = command.read_buffer_addr;
        }
        else {
            auto command = EnqueueReadInterleavedBufferCommand(this->id, this->device, buffer, dst, this->manager, src_page_index, pages_to_read);
            this->enqueue_command(command, blocking);
            command_read_buffer_addr = command.read_buffer_addr;
        }
        this->manager.completion_queue_wait_front(this->id); // wait for device to write data

        uint32_t bytes_read = pages_to_read * padded_page_size;
        if ((buffer.page_size() % 32) != 0) {
            // If page size is not 32B-aligned, we cannot do a contiguous copy
            uint32_t dst_address_offset = unpadded_dst_offset;
            for (uint32_t sysmem_address_offset = 0; sysmem_address_offset < bytes_read; sysmem_address_offset += padded_page_size) {
                tt::Cluster::instance().read_sysmem((char*)dst + dst_address_offset, buffer.page_size(), command_read_buffer_addr + sysmem_address_offset, mmio_device_id, channel);
                dst_address_offset += buffer.page_size();
            }
        } else {
            tt::Cluster::instance().read_sysmem((char*)dst + unpadded_dst_offset, bytes_read, command_read_buffer_addr, mmio_device_id, channel);
        }

        this->manager.completion_queue_pop_front(bytes_read, this->id);
        total_pages_to_read -= pages_to_read;
        src_page_index += pages_to_read;
        unpadded_dst_offset += pages_to_read * buffer.page_size();
    }

    if (buffer.buffer_layout() == TensorMemoryLayout::WIDTH_SHARDED or
        buffer.buffer_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
        convert_interleaved_to_sharded_on_host(dst,
                                        buffer,
                                        true);
    }
}

void CommandQueue::enqueue_write_buffer(Buffer& buffer, const void* src, bool blocking) {
    ZoneScopedN("CommandQueue_write_buffer");

    // TODO(agrebenisan): Fix these asserts after implementing multi-core CQ
    // TODO (abhullar): Use eth mem l1 size when issue queue interface kernel is on ethernet core
    TT_ASSERT(
        buffer.page_size() < MEM_L1_SIZE - get_data_section_l1_address(false),
        "Buffer pages must fit within the command queue data section");

    if (buffer.buffer_layout() == TensorMemoryLayout::WIDTH_SHARDED or
        buffer.buffer_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
        convert_interleaved_to_sharded_on_host(src,
                                    buffer
                                    );
    }

    uint32_t padded_page_size = align(buffer.page_size(), 32);
    uint32_t total_pages_to_write = buffer.num_pages();
    const uint32_t command_issue_limit = this->manager.get_issue_queue_limit(this->id);
    uint32_t dst_page_index = 0;
    while (total_pages_to_write > 0) {
        int32_t num_pages_available = (int32_t(command_issue_limit - this->manager.get_issue_queue_write_ptr(this->id)) - int32_t(DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND)) / int32_t(padded_page_size);
        // If not even a single device command fits, we hit this edgecase
        num_pages_available = std::max(num_pages_available, 0);

        uint32_t pages_to_write = std::min(total_pages_to_write, (uint32_t)num_pages_available);
        if (pages_to_write == 0) {
            // No space for command and data
            this->wrap(DeviceCommand::WrapRegion::ISSUE, blocking);
            num_pages_available = (int32_t(command_issue_limit - this->manager.get_issue_queue_write_ptr(this->id)) - int32_t(DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND)) / int32_t(padded_page_size);
            pages_to_write = std::min(total_pages_to_write, (uint32_t)num_pages_available);
        }

        tt::log_debug(tt::LogDispatch, "EnqueueWriteBuffer for channel {}", this->id);
        if (is_sharded(buffer.buffer_layout())) {
            auto command = EnqueueWriteShardedBufferCommand(this->id, this->device, buffer, src, this->manager, dst_page_index, pages_to_write);
            this->enqueue_command(command, blocking);
        }
        else {
            auto command = EnqueueWriteInterleavedBufferCommand(this->id, this->device, buffer, src, this->manager, dst_page_index, pages_to_write);
            this->enqueue_command(command, blocking);
        }

        total_pages_to_write -= pages_to_write;
        dst_page_index += pages_to_write;
    }
}

void CommandQueue::enqueue_program(Program& program, std::optional<std::reference_wrapper<Trace>> trace, bool blocking) {
    ZoneScopedN("CommandQueue_enqueue_program");

    // Need to relay the program into DRAM if this is the first time
    // we are seeing it
    const uint64_t program_id = program.get_id();

    // Whether or not we should stall the producer from prefetching binary data. If the
    // data is cached, then we don't need to stall, otherwise we need to wait for the
    // data to land in DRAM first
    bool stall = false;
    // No shared cache so far, can come at a later time
    if (not this->program_ids.count(program_id)) {
        this->program_ids.insert(program_id);
        stall = true;
        const void* program_data = program.get_program_map(this->device).program_pages.data();
        this->enqueue_write_buffer(program.get_program_buffer(device), program_data, false);
    }

    tt::log_debug(tt::LogDispatch, "EnqueueProgram for channel {}", this->id);
    tt::tt_metal::detail::ProgramMap& program_map = program.get_program_map(this->device);
    uint32_t host_data_num_pages = program_map.runtime_arg_page_transfers.size() + program_map.cb_config_page_transfers.size();
    uint32_t host_data_and_device_command_size =
        DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND + (host_data_num_pages * DeviceCommand::PROGRAM_PAGE_SIZE);
    if ((this->manager.get_issue_queue_write_ptr(this->id)) + host_data_and_device_command_size >=
        this->manager.get_issue_queue_size(this->id)) {
        TT_FATAL(
            host_data_and_device_command_size <= this->manager.get_issue_queue_size(this->id) - CQ_START, "EnqueueProgram command size too large");
        this->wrap(DeviceCommand::WrapRegion::ISSUE, blocking);
    }

    EnqueueProgramCommand command(
        this->id,
        this->device,
        program.get_program_buffer(this->device),
        program_map,
        this->manager,
        program,
        stall,
        trace);

    this->enqueue_command(command, blocking);
}

void CommandQueue::wait_finish() {
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(this->device->id());
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->device->id());

    // Poll to check that we're done.
    uint32_t finish_addr_offset = this->id * this->size_B;
    uint32_t finish;
    do {
        tt::Cluster::instance().read_sysmem(&finish, 4, HOST_CQ_FINISH_PTR + finish_addr_offset, mmio_device_id, channel);

        // There's also a case where the device can be hung due to an unanswered DPRINT WAIT and
        // a full print buffer. Poll the print server for this case and throw if it happens.
        if (DPrintServerHangDetected()) {
            TT_THROW("Command Queue could not finish: device hang due to unanswered DPRINT WAIT.");
        }
    } while (finish != 1);
    // Reset this value to 0 before moving on
    finish = 0;
    tt::Cluster::instance().write_sysmem(&finish, 4, HOST_CQ_FINISH_PTR + finish_addr_offset, mmio_device_id, channel);
}

void CommandQueue::finish() {
    ZoneScopedN("CommandQueue_finish");
    if ((this->manager.get_issue_queue_write_ptr(this->id)) + DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND >=
        this->manager.get_issue_queue_limit(this->id)) {
        this->wrap(DeviceCommand::WrapRegion::ISSUE, false);
    }
    tt::log_debug(tt::LogDispatch, "Finish for command queue {}", this->id);

    FinishCommand command(this->id, this->device, this->manager);
    this->enqueue_command(command, false);
    this->wait_finish();
}

void CommandQueue::wrap(DeviceCommand::WrapRegion wrap_region, bool blocking) {
    ZoneScopedN("CommandQueue_wrap");
    tt::log_debug(tt::LogDispatch, "EnqueueWrap for channel {}", this->id);
    EnqueueWrapCommand command(this->id, this->device, this->manager, wrap_region);
    this->enqueue_command(command, blocking);
}

void CommandQueue::restart() {
    ZoneScopedN("CommandQueue_restart");
    tt::log_debug(tt::LogDispatch, "EnqueueRestart for channel {}", this->id);
    EnqueueRestartCommand command(this->id, this->device, this->manager);
    this->enqueue_command(command, false);
    this->wait_finish();

    // Reset the manager
    this->manager.reset(this->id);
}

Trace::Trace(CommandQueue& command_queue): command_queue(command_queue) {
    this->trace_complete = false;
    this->num_data_bytes = 0;
}

void Trace::record(const TraceNode& trace_node) {
    TT_ASSERT(not this->trace_complete, "Cannot record any more for a completed trace");
    this->num_data_bytes += trace_node.num_data_bytes;
    this->history.push_back(trace_node);
}

void Trace::create_replay() {
    // Reconstruct the hugepage from the command cache
    SystemMemoryManager& manager = this->command_queue.manager;
    const uint32_t command_queue_id = this->command_queue.id;
    const bool lazy_push = true;
    for (auto& [device_command, data, command_type, num_data_bytes]: this->history) {
        uint32_t issue_write_ptr = manager.get_issue_queue_write_ptr(command_queue_id);
        device_command.update_buffer_transfer_src(0, issue_write_ptr + DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND);
        manager.cq_write(device_command.get_desc().data(), DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND, issue_write_ptr);
        manager.issue_queue_push_back(DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND, lazy_push, command_queue_id);

        uint32_t host_data_size = align(data.size() * sizeof(uint32_t), 16);
        manager.cq_write(data.data(), host_data_size, manager.get_issue_queue_write_ptr(command_queue_id));
        vector<uint32_t> read_back(host_data_size / sizeof(uint32_t), 0);
        tt::Cluster::instance().read_sysmem(read_back.data(), host_data_size, issue_write_ptr + DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND, 0, 0);
        manager.issue_queue_push_back(host_data_size, lazy_push, command_queue_id);
    }
}

void EnqueueReadBuffer(CommandQueue& cq, Buffer& buffer, vector<uint32_t>& dst, bool blocking) {
    // TODO(agrebenisan): Move to deprecated
    ZoneScoped;
    tt_metal::detail::DispatchStateCheck(true);
    TT_FATAL(blocking, "Non-blocking EnqueueReadBuffer not yet supported");

    // Only resizing here to keep with the original implementation. Notice how in the void*
    // version of this API, I assume the user mallocs themselves
    dst.resize(buffer.page_size() * buffer.num_pages() / sizeof(uint32_t));
    cq.enqueue_read_buffer(buffer, dst.data(), blocking);
}

void EnqueueWriteBuffer(CommandQueue& cq, Buffer& buffer, vector<uint32_t>& src, bool blocking) {
    // TODO(agrebenisan): Move to deprecated
    ZoneScoped;
    tt_metal::detail::DispatchStateCheck(true);
    cq.enqueue_write_buffer(buffer, src.data(), blocking);
}

void EnqueueReadBuffer(CommandQueue& cq, Buffer& buffer, void* dst, bool blocking) {
    ZoneScoped;
    tt_metal::detail::DispatchStateCheck(true);
    cq.enqueue_read_buffer(buffer, dst, blocking);
}

void EnqueueWriteBuffer(CommandQueue& cq, Buffer& buffer, const void* src, bool blocking) {
    ZoneScoped;
    tt_metal::detail::DispatchStateCheck(true);
    cq.enqueue_write_buffer(buffer, src, blocking);
}

void EnqueueProgram(CommandQueue& cq, Program& program, bool blocking, std::optional<std::reference_wrapper<Trace>> trace) {
    ZoneScoped;
    TT_ASSERT(cq.id == 0, "EnqueueProgram only supported on first command queue on device for time being.");
    detail::DispatchStateCheck(true);

    detail::CompileProgram(cq.device, program);

    program.allocate_circular_buffers();
    detail::ValidateCircularBufferRegion(program, cq.device);

    cq.enqueue_program(program, trace, blocking);
}

void Finish(CommandQueue& cq) {
    ZoneScoped;
    tt_metal::detail::DispatchStateCheck(true);
    cq.finish();
}

void ClearProgramCache(CommandQueue& cq) {
    // detail::DispatchStateCheck(true);
    // cq.program_to_buffer(cq.device->id()).clear();
    // cq.program_to_dev_map(cq.device->id()).clear();
}

Trace BeginTrace(CommandQueue& command_queue) {
    // Resets the command queue state
    command_queue.restart();

    return Trace(command_queue);
}

void EndTrace(Trace& trace) {
    TT_ASSERT(not trace.trace_complete, "Already completed this trace");
    SystemMemoryManager& manager = trace.command_queue.manager;
    const uint32_t command_queue_id = trace.command_queue.id;
    TT_FATAL(trace.num_data_bytes + trace.history.size() * DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND <= manager.get_issue_queue_limit(command_queue_id), "Trace does not fit in issue queue");
    trace.trace_complete = true;
    manager.set_issue_queue_size(command_queue_id, trace.num_data_bytes + DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND * trace.history.size());
    trace.command_queue.restart();
    trace.create_replay();
    manager.reset(trace.command_queue.id);
}

void EnqueueTrace(Trace& trace, bool blocking) {
    // Run the trace
    CommandQueue& command_queue = trace.command_queue;
    uint32_t trace_size = trace.history.size() * DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND + trace.num_data_bytes;
    command_queue.manager.issue_queue_reserve_back(trace_size, command_queue.id);
    command_queue.manager.issue_queue_push_back(trace_size, false, command_queue.id);

    // This will block because the wr toggles will be different between the host and the device
    if (blocking) {
        command_queue.manager.issue_queue_reserve_back(trace_size, command_queue.id);
    }
}

namespace detail {

void EnqueueRestart(CommandQueue& cq) {
    ZoneScoped;
    detail::DispatchStateCheck(true);
    cq.restart();
}

}

}  // namespace tt::tt_metal
