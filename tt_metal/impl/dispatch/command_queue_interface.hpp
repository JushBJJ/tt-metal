// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <mutex>

#include "tt_metal/common/base.hpp"
#include "tt_metal/common/math.hpp"
#include "tt_metal/impl/dispatch/cq_commands.hpp"
#include "tt_metal/impl/dispatch/dispatch_address_map.hpp"
#include "tt_metal/impl/dispatch/dispatch_core_manager.hpp"
#include "tt_metal/llrt/llrt.hpp"

using namespace tt::tt_metal;

// todo consider moving these to dispatch_addr_map
static constexpr uint32_t MAX_HUGEPAGE_SIZE = 1 << 30; // 1GB;
static constexpr uint32_t MAX_DEV_CHANNEL_SIZE = 1 << 28; // 256 MB;
static constexpr uint32_t DEVICES_PER_UMD_CHANNEL = MAX_HUGEPAGE_SIZE / MAX_DEV_CHANNEL_SIZE; // 256 MB;


static constexpr uint32_t MEMCPY_ALIGNMENT = sizeof(__m128i);

struct dispatch_constants {
   public:
    dispatch_constants &operator=(const dispatch_constants &) = delete;
    dispatch_constants &operator=(dispatch_constants &&other) noexcept = delete;
    dispatch_constants(const dispatch_constants &) = delete;
    dispatch_constants(dispatch_constants &&other) noexcept = delete;

    static const dispatch_constants &get(const CoreType &core_type) {
        static dispatch_constants inst = dispatch_constants(core_type);
        return inst;
    }

    typedef uint16_t prefetch_q_entry_type;
    static constexpr uint32_t PREFETCH_Q_LOG_MINSIZE = 4;
    static constexpr uint32_t PREFETCH_Q_BASE = DISPATCH_L1_UNRESERVED_BASE;

    static constexpr uint32_t LOG_TRANSFER_PAGE_SIZE = 12;
    static constexpr uint32_t TRANSFER_PAGE_SIZE = 1 << LOG_TRANSFER_PAGE_SIZE;
    static constexpr uint32_t ISSUE_Q_ALIGNMENT = 32; // TODO: Should this be PCIe alignment?

    static constexpr uint32_t DISPATCH_BUFFER_LOG_PAGE_SIZE = 12;
    static constexpr uint32_t DISPATCH_BUFFER_SIZE_BLOCKS = 4;
    static constexpr uint32_t DISPATCH_BUFFER_BASE =
        ((DISPATCH_L1_UNRESERVED_BASE - 1) | ((1 << DISPATCH_BUFFER_LOG_PAGE_SIZE) - 1)) + 1;

    static constexpr uint32_t PREFETCH_D_BUFFER_LOG_PAGE_SIZE = 12;
    static constexpr uint32_t PREFETCH_D_BUFFER_BLOCKS = 4;

    static constexpr uint32_t EVENT_PADDED_SIZE = 16;
    // When page size of buffer to write/read exceeds MAX_PREFETCH_COMMAND_SIZE, the PCIe aligned page size is broken
    // down into equal sized partial pages BASE_PARTIAL_PAGE_SIZE denotes the initial partial page size to use, it is
    // incremented by PCIe alignment until page size can be evenly split
    static constexpr uint32_t BASE_PARTIAL_PAGE_SIZE = 4096;

    uint32_t prefetch_q_entries() const { return prefetch_q_entries_; }

    uint32_t prefetch_q_size() const { return prefetch_q_size_; }

    uint32_t max_prefetch_command_size() const { return max_prefetch_command_size_; }

    uint32_t cmddat_q_base() const { return cmddat_q_base_; }

    uint32_t cmddat_q_size() const { return cmddat_q_size_; }

    uint32_t scratch_db_base() const { return scratch_db_base_; }

    uint32_t scratch_db_size() const { return scratch_db_size_; }

    uint32_t dispatch_buffer_block_size_pages() const { return dispatch_buffer_block_size_pages_; }

    uint32_t dispatch_buffer_pages() const { return dispatch_buffer_pages_; }

    uint32_t prefetch_d_buffer_size() const { return prefetch_d_buffer_size_; }

    uint32_t prefetch_d_buffer_pages() const { return prefetch_d_buffer_pages_; }

    uint32_t mux_buffer_size(uint8_t num_hw_cqs = 1) const { return prefetch_d_buffer_size_ / num_hw_cqs; }

    uint32_t mux_buffer_pages(uint8_t num_hw_cqs = 1) const { return prefetch_d_buffer_pages_ / num_hw_cqs; }

   private:
    dispatch_constants(const CoreType &core_type) {
        TT_ASSERT(core_type == CoreType::WORKER or core_type == CoreType::ETH);
        // make this 2^N as required by the packetized stages
        uint32_t dispatch_buffer_block_size;
        if (core_type == CoreType::WORKER) {
            prefetch_q_entries_ = 1534;
            max_prefetch_command_size_ = 128 * 1024;
            cmddat_q_size_ = 256 * 1024;
            scratch_db_size_ = 128 * 1024;
            dispatch_buffer_block_size = 512 * 1024;
            prefetch_d_buffer_size_ = 256 * 1024;
        } else {
            prefetch_q_entries_ = 128;
            max_prefetch_command_size_ = 32 * 1024;
            cmddat_q_size_ = 64 * 1024;
            scratch_db_size_ = 20 * 1024;
            dispatch_buffer_block_size = 128 * 1024;
            prefetch_d_buffer_size_ = 128 * 1024;
        }
        TT_ASSERT(cmddat_q_size_ >= 2 * max_prefetch_command_size_);
        TT_ASSERT(scratch_db_size_ % 2 == 0);
        TT_ASSERT((dispatch_buffer_block_size & (dispatch_buffer_block_size - 1)) == 0);

        prefetch_q_size_ = prefetch_q_entries_ * sizeof(prefetch_q_entry_type);
        cmddat_q_base_ = PREFETCH_Q_BASE + ((prefetch_q_size_ + PCIE_ALIGNMENT - 1) / PCIE_ALIGNMENT * PCIE_ALIGNMENT);
        scratch_db_base_ = cmddat_q_base_ + ((cmddat_q_size_ + PCIE_ALIGNMENT - 1) / PCIE_ALIGNMENT * PCIE_ALIGNMENT);
        const uint32_t l1_size = core_type == CoreType::WORKER ? MEM_L1_SIZE : MEM_ETH_SIZE;
        TT_ASSERT(scratch_db_base_ + scratch_db_size_ < l1_size);
        dispatch_buffer_block_size_pages_ =
            dispatch_buffer_block_size / (1 << DISPATCH_BUFFER_LOG_PAGE_SIZE) / DISPATCH_BUFFER_SIZE_BLOCKS;
        dispatch_buffer_pages_ = dispatch_buffer_block_size_pages_ * DISPATCH_BUFFER_SIZE_BLOCKS;
        uint32_t dispatch_cb_end = DISPATCH_BUFFER_BASE + (1 << DISPATCH_BUFFER_LOG_PAGE_SIZE) * dispatch_buffer_pages_;
        TT_ASSERT(dispatch_cb_end < l1_size);
        prefetch_d_buffer_pages_ = prefetch_d_buffer_size_ >> PREFETCH_D_BUFFER_LOG_PAGE_SIZE;
    }

    uint32_t prefetch_q_entries_;
    uint32_t prefetch_q_size_;
    uint32_t max_prefetch_command_size_;
    uint32_t cmddat_q_base_;
    uint32_t cmddat_q_size_;
    uint32_t scratch_db_base_;
    uint32_t scratch_db_size_;
    uint32_t dispatch_buffer_block_size_pages_;
    uint32_t dispatch_buffer_pages_;
    uint32_t prefetch_d_buffer_size_;
    uint32_t prefetch_d_buffer_pages_;
};

// Define a queue type, for when they're interchangeable.
#define CQ_COMPLETION_QUEUE true
#define CQ_ISSUE_QUEUE false
typedef bool cq_queue_t;

/// @brief Get offset of the command queue relative to its channel
/// @param cq_id uint8_t ID the command queue
/// @param cq_size uint32_t size of the command queue
/// @return uint32_t relative offset
inline uint32_t get_relative_cq_offset(uint8_t cq_id, uint32_t cq_size) { return cq_id * cq_size; }

inline uint16_t get_umd_channel(uint16_t channel) {
    return channel & 0x3;
}

/// @brief Get absolute offset of the command queue
/// @param channel uint16_t channel ID (hugepage)
/// @param cq_id uint8_t ID the command queue
/// @param cq_size uint32_t size of the command queue
/// @return uint32_t absolute offset
inline uint32_t get_absolute_cq_offset(uint16_t channel, uint8_t cq_id, uint32_t cq_size) {
    return (MAX_HUGEPAGE_SIZE * get_umd_channel(channel)) + ((channel >> 2) * MAX_DEV_CHANNEL_SIZE) + get_relative_cq_offset(cq_id, cq_size);
}

template <bool addr_16B>
inline uint32_t get_cq_issue_rd_ptr(chip_id_t chip_id, uint8_t cq_id, uint32_t cq_size) {
    uint32_t recv;
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(chip_id);
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(chip_id);
    uint32_t channel_offset = (channel >> 2) * MAX_DEV_CHANNEL_SIZE;
    tt::Cluster::instance().read_sysmem(
        &recv,
        sizeof(uint32_t),
        HOST_CQ_ISSUE_READ_PTR + channel_offset + get_relative_cq_offset(cq_id, cq_size),
        mmio_device_id,
        channel);
    if (not addr_16B) {
        return recv << 4;
    }
    return recv;
}

template <bool addr_16B>
inline uint32_t get_cq_issue_wr_ptr(chip_id_t chip_id, uint8_t cq_id, uint32_t cq_size) {
    uint32_t recv;
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(chip_id);
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(chip_id);
    tt::Cluster::instance().read_sysmem(&recv, sizeof(uint32_t), HOST_CQ_ISSUE_WRITE_PTR + get_relative_cq_offset(cq_id, cq_size), mmio_device_id, channel);
    if (not addr_16B) {
        return recv << 4;
    }
    return recv;
}

template <bool addr_16B>
inline uint32_t get_cq_completion_wr_ptr(chip_id_t chip_id, uint8_t cq_id, uint32_t cq_size) {
    uint32_t recv;
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(chip_id);
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(chip_id);
    uint32_t channel_offset = (channel >> 2) * MAX_DEV_CHANNEL_SIZE;
    tt::Cluster::instance().read_sysmem(
        &recv,
        sizeof(uint32_t),
        HOST_CQ_COMPLETION_WRITE_PTR + channel_offset + get_relative_cq_offset(cq_id, cq_size),
        mmio_device_id,
        channel);
    if (not addr_16B) {
        return recv << 4;
    }
    return recv;
}

template <bool addr_16B>
inline uint32_t get_cq_completion_rd_ptr(chip_id_t chip_id, uint8_t cq_id, uint32_t cq_size) {
    uint32_t recv;
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(chip_id);
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(chip_id);
    tt::Cluster::instance().read_sysmem(&recv, sizeof(uint32_t), HOST_CQ_COMPLETION_READ_PTR + get_relative_cq_offset(cq_id, cq_size), mmio_device_id, channel);
    if (not addr_16B) {
        return recv << 4;
    }
    return recv;
}

// Ideally would work by cachelines, but the min size is less than that
// TODO: Revisit this w/ regard to possibly eliminating min sizes and orphan writes at the end
// TODO: ditto alignment isues
template <bool debug_sync = false>
static inline void memcpy_to_device(void *__restrict dst, const void *__restrict src, size_t n) {
    TT_ASSERT((uintptr_t)dst % MEMCPY_ALIGNMENT == 0);
    TT_ASSERT(n % sizeof(uint32_t) == 0);

    static constexpr uint32_t inner_loop = 8;
    static constexpr uint32_t inner_blk_size = inner_loop * sizeof(__m256i);

    uint8_t *src8 = (uint8_t *)src;
    uint8_t *dst8 = (uint8_t *)dst;

    if (size_t num_lines = n / inner_blk_size) {
        for (size_t i = 0; i < num_lines; ++i) {
            for (size_t j = 0; j < inner_loop; ++j) {
                __m256i blk = _mm256_loadu_si256((const __m256i *)src8);
                _mm256_stream_si256((__m256i *)dst8, blk);
                src8 += sizeof(__m256i);
                dst8 += sizeof(__m256i);
            }
            n -= inner_blk_size;
        }
    }

    if (n > 0) {
        if (size_t num_lines = n / sizeof(__m256i)) {
            for (size_t i = 0; i < num_lines; ++i) {
                __m256i blk = _mm256_loadu_si256((const __m256i *)src8);
                _mm256_stream_si256((__m256i *)dst8, blk);
                src8 += sizeof(__m256i);
                dst8 += sizeof(__m256i);
            }
            n -= num_lines * sizeof(__m256i);
        }
        if (size_t num_lines = n / sizeof(__m128i)) {
            for (size_t i = 0; i < num_lines; ++i) {
                __m128i blk = _mm_loadu_si128((const __m128i *)src8);
                _mm_stream_si128((__m128i *)dst8, blk);
                src8 += sizeof(__m128i);
                dst8 += sizeof(__m128i);
            }
            n -= n / sizeof(__m128i) * sizeof(__m128i);
        }
        if (n > 0) {
            for (size_t i = 0; i < n / sizeof(int32_t); ++i) {
                _mm_stream_si32((int32_t *)dst8, *(int32_t *)src8);
                src8 += sizeof(int32_t);
                dst8 += sizeof(int32_t);
            }
        }
    }
    if constexpr (debug_sync) {
        tt_driver_atomics::sfence();
    }
}

struct SystemMemoryCQInterface {
    // CQ is split into issue and completion regions
    // Host writes commands and data for H2D transfers in the issue region, device reads from the issue region
    // Device signals completion and writes data for D2H transfers in the completion region, host reads from the
    // completion region Equation for issue fifo size is | issue_fifo_wr_ptr + command size B - issue_fifo_rd_ptr |
    // Space available would just be issue_fifo_limit - issue_fifo_size
    SystemMemoryCQInterface(uint16_t channel, uint8_t cq_id, uint32_t cq_size) :
        command_completion_region_size(
            (((cq_size - CQ_START) / dispatch_constants::TRANSFER_PAGE_SIZE) / 4) *
            dispatch_constants::TRANSFER_PAGE_SIZE),
        command_issue_region_size((cq_size - CQ_START) - this->command_completion_region_size),
        issue_fifo_size(command_issue_region_size >> 4),
        issue_fifo_limit(
            ((CQ_START + this->command_issue_region_size) + get_absolute_cq_offset(channel, cq_id, cq_size)) >> 4),
        completion_fifo_size(command_completion_region_size >> 4),
        completion_fifo_limit(issue_fifo_limit + completion_fifo_size),
        offset(get_absolute_cq_offset(channel, cq_id, cq_size)),
        id(cq_id) {
        TT_ASSERT(
            this->command_completion_region_size % PCIE_ALIGNMENT == 0 and
                this->command_issue_region_size % PCIE_ALIGNMENT == 0,
            "Issue queue and completion queue need to be {}B aligned!",
            PCIE_ALIGNMENT);
        TT_ASSERT(this->issue_fifo_limit != 0, "Cannot have a 0 fifo limit");
        // Currently read / write pointers on host and device assumes contiguous ranges for each channel
        // Device needs absolute offset of a hugepage to access the region of sysmem that holds a particular command
        // queue
        //  but on host, we access a region of sysmem using addresses relative to a particular channel
        this->issue_fifo_wr_ptr = (CQ_START + this->offset) >> 4;  // In 16B words
        this->issue_fifo_wr_toggle = 0;

        this->completion_fifo_rd_ptr = this->issue_fifo_limit;
        this->completion_fifo_rd_toggle = 0;
    }

    // Percentage of the command queue that is dedicated for issuing commands. Issue queue size is rounded to be 32B
    // aligned and remaining space is dedicated for completion queue Smaller issue queues can lead to more stalls for
    // applications that send more work to device than readback data.
    static constexpr float default_issue_queue_split = 0.75;
    const uint32_t command_completion_region_size;
    const uint32_t command_issue_region_size;
    const uint8_t id;

    uint32_t issue_fifo_size;
    uint32_t issue_fifo_limit;  // Last possible FIFO address
    const uint32_t offset;
    uint32_t issue_fifo_wr_ptr;
    bool issue_fifo_wr_toggle;

    uint32_t completion_fifo_size;
    uint32_t completion_fifo_limit;  // Last possible FIFO address
    uint32_t completion_fifo_rd_ptr;
    bool completion_fifo_rd_toggle;
};

class SystemMemoryManager {
   private:
    chip_id_t device_id;
    uint8_t num_hw_cqs;
    const uint32_t m_dma_buf_size;
    const std::function<void(uint32_t, uint32_t, const uint8_t *, uint32_t)> fast_write_callable;
    vector<uint32_t> completion_byte_addrs;
    char *cq_sysmem_start;
    vector<SystemMemoryCQInterface> cq_interfaces;
    uint32_t cq_size;
    uint32_t channel_offset;
    vector<int> cq_to_event;
    vector<int> cq_to_last_completed_event;
    vector<std::mutex> cq_to_event_locks;
    vector<tt_cxy_pair> prefetcher_cores;
    vector<tt::Writer> prefetch_q_writers;
    vector<uint32_t> prefetch_q_dev_ptrs;
    vector<uint32_t> prefetch_q_dev_fences;

    bool bypass_enable;
    vector<uint32_t> bypass_buffer;
    uint32_t bypass_buffer_write_offset;

   public:
    SystemMemoryManager(chip_id_t device_id, uint8_t num_hw_cqs) :
        device_id(device_id),
        num_hw_cqs(num_hw_cqs),
        m_dma_buf_size(tt::Cluster::instance().get_m_dma_buf_size(device_id)),
        fast_write_callable(tt::Cluster::instance().get_fast_pcie_static_tlb_write_callable(device_id)),
        bypass_enable(false),
        bypass_buffer_write_offset(0) {
        this->completion_byte_addrs.resize(num_hw_cqs);
        this->prefetcher_cores.resize(num_hw_cqs);
        this->prefetch_q_writers.reserve(num_hw_cqs);
        this->prefetch_q_dev_ptrs.resize(num_hw_cqs);
        this->prefetch_q_dev_fences.resize(num_hw_cqs);

        // Split hugepage into however many pieces as there are CQs
        chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id);
        uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device_id);
        char *hugepage_start = (char *)tt::Cluster::instance().host_dma_address(0, mmio_device_id, channel);
        hugepage_start += (channel >> 2) * MAX_DEV_CHANNEL_SIZE;
        this->cq_sysmem_start = hugepage_start;

        // TODO(abhullar): Remove env var and expose sizing at the API level
        char *cq_size_override_env = std::getenv("TT_METAL_CQ_SIZE_OVERRIDE");
        if (cq_size_override_env != nullptr) {
            uint32_t cq_size_override = std::stoi(string(cq_size_override_env));
            this->cq_size = cq_size_override;
        } else {
            this->cq_size = tt::Cluster::instance().get_host_channel_size(mmio_device_id, channel) / num_hw_cqs;
            if (tt::Cluster::instance().is_galaxy_cluster()) {
                //We put 4 galaxy devices per huge page since number of hugepages available is less than number of devices.
                this->cq_size = this->cq_size / DEVICES_PER_UMD_CHANNEL;
            }
        }
        this->channel_offset = MAX_HUGEPAGE_SIZE * get_umd_channel(channel) + (channel >> 2) * MAX_DEV_CHANNEL_SIZE;

        CoreType core_type = dispatch_core_manager::get(num_hw_cqs).get_dispatch_core_type(device_id);
        for (uint8_t cq_id = 0; cq_id < num_hw_cqs; cq_id++) {
            tt_cxy_pair prefetcher_core =
                dispatch_core_manager::get(num_hw_cqs).prefetcher_core(device_id, channel, cq_id);
            tt_cxy_pair prefetcher_physical_core =
                tt_cxy_pair(prefetcher_core.chip, tt::get_physical_core_coordinate(prefetcher_core, core_type));
            this->prefetcher_cores[cq_id] = prefetcher_physical_core;
            this->prefetch_q_writers.emplace_back(tt::Cluster::instance().get_static_tlb_writer(prefetcher_physical_core));

            tt_cxy_pair completion_queue_writer_core =
                dispatch_core_manager::get(num_hw_cqs).completion_queue_writer_core(device_id, channel, cq_id);
            const std::tuple<uint32_t, uint32_t> completion_interface_tlb_data =
                tt::Cluster::instance()
                    .get_tlb_data(tt_cxy_pair(
                        completion_queue_writer_core.chip,
                        tt::get_physical_core_coordinate(completion_queue_writer_core, core_type)))
                    .value();
            auto [completion_tlb_offset, completion_tlb_size] = completion_interface_tlb_data;
            this->completion_byte_addrs[cq_id] = completion_tlb_offset + CQ_COMPLETION_READ_PTR % completion_tlb_size;

            this->cq_interfaces.push_back(SystemMemoryCQInterface(channel, cq_id, this->cq_size));
            // Prefetch queue acts as the sync mechanism to ensure that issue queue has space to write, so issue queue
            // must be as large as the max amount of space the prefetch queue can specify Plus 1 to handle wrapping Plus
            // 1 to allow us to start writing to issue queue before we reserve space in the prefetch queue
            TT_FATAL(
                dispatch_constants::get(core_type).max_prefetch_command_size() *
                    (dispatch_constants::get(core_type).prefetch_q_entries() + 2) <=
                this->get_issue_queue_size(cq_id));
            this->cq_to_event.push_back(0);
            this->cq_to_last_completed_event.push_back(0);
            this->prefetch_q_dev_ptrs[cq_id] = dispatch_constants::PREFETCH_Q_BASE;
            this->prefetch_q_dev_fences[cq_id] =
                dispatch_constants::PREFETCH_Q_BASE + dispatch_constants::get(core_type).prefetch_q_entries() *
                                                          sizeof(dispatch_constants::prefetch_q_entry_type);
        }
        vector<std::mutex> temp_mutexes(num_hw_cqs);
        cq_to_event_locks.swap(temp_mutexes);
    }

    // Returns the number of pages taken up by this command (includes data).
    uint32_t dump_dispatch_cmd(CQDispatchCmd *cmd, uint32_t cmd_addr, std::ofstream &cq_file) {
        uint32_t stride = sizeof(CQDispatchCmd); // Default stride is just the command
        CQDispatchCmdId cmd_id = cmd->base.cmd_id;

        if (cmd_id < CQ_DISPATCH_CMD_MAX_COUNT) {
            cq_file << fmt::format("{:#010x}: {}", cmd_addr, cmd_id);
            switch (cmd_id) {
                case CQ_DISPATCH_CMD_WRITE_LINEAR:
                case CQ_DISPATCH_CMD_WRITE_LINEAR_H:
                    cq_file << fmt::format(
                        " (num_mcast_dests={}, noc_xy_addr={:#010x}, addr={:#010x}, length={:#010x})",
                        cmd->write_linear.num_mcast_dests,
                        cmd->write_linear.noc_xy_addr,
                        cmd->write_linear.addr,
                        cmd->write_linear.length);
                    stride += cmd->write_linear.length;
                    break;
                case CQ_DISPATCH_CMD_WRITE_LINEAR_H_HOST:
                    if (cmd->write_linear_host.is_event) {
                        uint32_t *event_ptr = (uint32_t *)(cmd + 1);
                        cq_file << fmt::format(" (completed_event_id={})", *event_ptr);
                    } else {
                        cq_file << fmt::format(" (length={:#010x})", cmd->write_linear_host.length);
                    }
                    stride += cmd->write_linear_host.length;
                    break;
                case CQ_DISPATCH_CMD_WRITE_PAGED:
                    cq_file << fmt::format(
                        " (is_dram={}, start_page={}, base_addr={:#010x}, page_size={:#010x}, pages={})",
                        cmd->write_paged.is_dram,
                        cmd->write_paged.start_page,
                        cmd->write_paged.base_addr,
                        cmd->write_paged.page_size,
                        cmd->write_paged.pages);
                    stride += cmd->write_paged.pages * cmd->write_paged.page_size;
                    break;
                case CQ_DISPATCH_CMD_WRITE_PACKED:
                    cq_file << fmt::format(
                        " (flags={:#02x}, count={}, addr={:#010x}, size={:04x})",
                        cmd->write_packed.flags,
                        cmd->write_packed.count,
                        cmd->write_packed.addr,
                        cmd->write_packed.size);
                    // TODO: How does the page count for for packed writes?
                    break;
                case CQ_DISPATCH_CMD_WRITE_PACKED_LARGE:
                    cq_file << fmt::format(
                        " (count={}, alignment={})", cmd->write_packed_large.count, cmd->write_packed_large.alignment);
                    break;
                case CQ_DISPATCH_CMD_WAIT:
                    cq_file << fmt::format(
                        " (barrier={}, notify_prefetch={}, clear_count=(), wait={}, addr={:#010x}, "
                        "count = {})",
                        cmd->wait.barrier,
                        cmd->wait.notify_prefetch,
                        cmd->wait.clear_count,
                        cmd->wait.wait,
                        cmd->wait.addr,
                        cmd->wait.count);
                    break;
                case CQ_DISPATCH_CMD_DEBUG:
                    cq_file << fmt::format(
                        " (pad={}, key={}, checksum={:#010x}, size={}, stride={})",
                        cmd->debug.pad,
                        cmd->debug.key,
                        cmd->debug.checksum,
                        cmd->debug.size,
                        cmd->debug.stride);
                    break;
                case CQ_DISPATCH_CMD_DELAY: cq_file << fmt::format(" (delay={})", cmd->delay.delay); break;
                // These commands don't have any additional data to dump.
                case CQ_DISPATCH_CMD_ILLEGAL: break;
                case CQ_DISPATCH_CMD_GO: break;
                case CQ_DISPATCH_CMD_SINK: break;
                case CQ_DISPATCH_CMD_EXEC_BUF_END: break;
                case CQ_DISPATCH_CMD_REMOTE_WRITE: break;
                case CQ_DISPATCH_CMD_TERMINATE: break;
                default: TT_FATAL("Unrecognized dispatch command: {}", cmd_id); break;
            }
        }
        return stride;
    }

    // Returns the number of bytes to the next prefetch command.
    uint32_t dump_prefetch_cmd(CQPrefetchCmd *cmd, uint32_t cmd_addr, std::ofstream &iq_file) {
        uint32_t stride = dispatch_constants::ISSUE_Q_ALIGNMENT; // Default stride matches alignment.
        CQPrefetchCmdId cmd_id = cmd->base.cmd_id;

        if (cmd_id < CQ_PREFETCH_CMD_MAX_COUNT) {
            iq_file << fmt::format("{:#010x}: {}", cmd_addr, cmd_id);
            switch (cmd_id) {
                case CQ_PREFETCH_CMD_RELAY_LINEAR:
                    iq_file << fmt::format(
                        " (noc_xy_addr={:#010x}, addr={:#010x}, length={:#010x})",
                        cmd->relay_linear.noc_xy_addr,
                        cmd->relay_linear.addr,
                        cmd->relay_linear.length);
                    break;
                case CQ_PREFETCH_CMD_RELAY_PAGED:
                    iq_file << fmt::format(
                        " (packed_page_flags={:#02x}, length_adjust={:#x}, base_addr={:#010x}, page_size={:#010x}, "
                        "pages={:#010x})",
                        cmd->relay_paged.packed_page_flags,
                        cmd->relay_paged.length_adjust,
                        cmd->relay_paged.base_addr,
                        cmd->relay_paged.page_size,
                        cmd->relay_paged.pages);
                    break;
                case CQ_PREFETCH_CMD_RELAY_PAGED_PACKED:
                    iq_file << fmt::format(
                        " (count={}, total_length={:#010x}, stride={:#010x})",
                        cmd->relay_paged_packed.count,
                        cmd->relay_paged_packed.total_length,
                        cmd->relay_paged_packed.stride);
                    stride = cmd->relay_paged_packed.stride;
                    break;
                case CQ_PREFETCH_CMD_RELAY_INLINE:
                case CQ_PREFETCH_CMD_RELAY_INLINE_NOFLUSH:
                case CQ_PREFETCH_CMD_EXEC_BUF_END:
                    iq_file << fmt::format(
                        " (length={:#010x}, stride={:#010x})",
                        cmd->relay_inline.length,
                        cmd->relay_inline.stride);
                    stride = cmd->relay_inline.stride;
                    break;
                case CQ_PREFETCH_CMD_EXEC_BUF:
                    iq_file << fmt::format(
                        " (base_addr={:#010x}, log_page_size={}, pages={})",
                        cmd->exec_buf.base_addr,
                        cmd->exec_buf.log_page_size,
                        cmd->exec_buf.pages);
                    break;
                case CQ_PREFETCH_CMD_DEBUG:
                    iq_file << fmt::format(
                        " (pad={}, key={}, checksum={:#010x}, size={}, stride={})",
                        cmd->debug.pad,
                        cmd->debug.key,
                        cmd->debug.checksum,
                        cmd->debug.size,
                        cmd->debug.stride);
                    stride = cmd->debug.stride;
                    break;
                case CQ_PREFETCH_CMD_WAIT_FOR_EVENT:
                    iq_file << fmt::format(
                        " (sync_event={:#08x}, sync_event_addr={:#08x})",
                        cmd->event_wait.sync_event,
                        cmd->event_wait.sync_event_addr);
                    stride = CQ_PREFETCH_CMD_BARE_MIN_SIZE + sizeof(CQPrefetchHToPrefetchDHeader);
                    break;
                // These commands don't have any additional data to dump.
                case CQ_PREFETCH_CMD_ILLEGAL: break;
                case CQ_PREFETCH_CMD_STALL: break;
                case CQ_PREFETCH_CMD_TERMINATE: break;
                default: break;
            }
        }
        return stride;
    }

    void print_progress_bar(float progress, bool init = false) {
        if (progress > 1.0)
            progress = 1.0;
        static int prev_bar_position = -1;
        if (init)
            prev_bar_position = -1;
        int progress_bar_width = 80;
        int bar_position = static_cast<int>(progress * progress_bar_width);
        if (bar_position > prev_bar_position) {
            std::cout << "[";
            std::cout << string(bar_position, '=') << string(progress_bar_width - bar_position, ' ');
            std::cout << "]" << int(progress * 100.0) << " %\r" << std::flush;
            prev_bar_position = bar_position;
        }
    }

    void dump_completion_queue_entries(std::ofstream &cq_file, SystemMemoryCQInterface &cq_interface) {
        chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(this->device_id);
        uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->device_id);
        uint32_t completion_write_ptr = get_cq_completion_wr_ptr<true>(this->device_id, cq_interface.id, this->cq_size)
                                        << 4;
        uint32_t completion_read_ptr = get_cq_completion_rd_ptr<true>(this->device_id, cq_interface.id, this->cq_size)
                                       << 4;
        uint32_t completion_q_bytes = cq_interface.completion_fifo_size << 4;
        TT_ASSERT(completion_q_bytes % dispatch_constants::TRANSFER_PAGE_SIZE == 0);
        uint32_t base_addr = (cq_interface.issue_fifo_limit << 4);

        // Read out in pages, this is fine since all completion Q entries st
        vector<uint8_t> read_data;
        read_data.resize(dispatch_constants::TRANSFER_PAGE_SIZE);
        tt::log_info("Reading Device {} CQ {}, Completion Queue...", this->device_id, cq_interface.id);
        cq_file << fmt::format(
                       "Device {}, CQ {}, Completion Queue: write_ptr={:#010x}, read_ptr={:#010x}\n",
                       this->device_id,
                       cq_interface.id,
                       completion_write_ptr,
                       completion_read_ptr);
        uint32_t last_span_start;
        bool last_span_invalid = false;
        print_progress_bar(0.0, true);
        for (uint32_t page_offset = 0; page_offset < completion_q_bytes;) {  // page_offset increment at end of loop
            uint32_t page_addr = base_addr + page_offset;
            tt::Cluster::instance().read_sysmem(read_data.data(), read_data.size(), page_addr, mmio_device_id, channel);

            // Check if this page starts with a valid command id
            CQDispatchCmd *cmd = (CQDispatchCmd *)read_data.data();
            if (cmd->base.cmd_id < CQ_DISPATCH_CMD_MAX_COUNT && cmd->base.cmd_id < CQ_DISPATCH_CMD_ILLEGAL) {
                if (last_span_invalid) {
                    if (page_addr == last_span_start + dispatch_constants::TRANSFER_PAGE_SIZE) {
                        cq_file << fmt::format("{:#010x}: No valid dispatch commands detected.", last_span_start);
                    } else {
                        cq_file << fmt::format(
                            "{:#010x}-{:#010x}: No valid dispatch commands detected.",
                            last_span_start,
                            page_addr - dispatch_constants::TRANSFER_PAGE_SIZE);
                    }
                    last_span_invalid = false;
                    if (last_span_start <= (completion_write_ptr) &&
                        page_addr - dispatch_constants::TRANSFER_PAGE_SIZE >= (completion_write_ptr)) {
                        cq_file << fmt::format(" << write_ptr (0x{:08x})", completion_write_ptr);
                    }
                    if (last_span_start <= (completion_read_ptr) &&
                        page_addr - dispatch_constants::TRANSFER_PAGE_SIZE >= (completion_read_ptr)) {
                        cq_file << fmt::format(" << read_ptr (0x{:08x})", completion_read_ptr);
                    }
                    cq_file << std::endl;
                }
                uint32_t stride = dump_dispatch_cmd(cmd, page_addr, cq_file);
                // Completion Q is page-aligned
                uint32_t cmd_pages = (stride + dispatch_constants::TRANSFER_PAGE_SIZE - 1) / dispatch_constants::TRANSFER_PAGE_SIZE;
                page_offset += cmd_pages * dispatch_constants::TRANSFER_PAGE_SIZE;
                if (page_addr == completion_write_ptr)
                    cq_file << fmt::format(" << write_ptr (0x{:08x})", completion_write_ptr);
                if (page_addr == completion_read_ptr)
                    cq_file << fmt::format(" << read_ptr (0x{:08x})", completion_read_ptr);
                cq_file << std::endl;

                // Show which pages have data if present.
                if (cmd_pages > 2) {
                    cq_file << fmt::format(
                        "{:#010x}-{:#010x}: Data pages\n",
                        page_addr + dispatch_constants::TRANSFER_PAGE_SIZE,
                        page_addr + (cmd_pages - 1) * dispatch_constants::TRANSFER_PAGE_SIZE);
                } else if (cmd_pages == 2) {
                    cq_file << fmt::format("{:#010x}: Data page\n", page_addr + dispatch_constants::TRANSFER_PAGE_SIZE);
                }
            } else {
                // If no valid command, just move on and try the next page
                // cq_file << fmt::format("{:#010x}: No valid dispatch command", page_addr) << std::endl;
                if (!last_span_invalid)
                    last_span_start = page_addr;
                last_span_invalid = true;
                page_offset += dispatch_constants::TRANSFER_PAGE_SIZE;
            }

            print_progress_bar((float)page_offset / completion_q_bytes + 0.005);
        }
        if (last_span_invalid) {
            cq_file << fmt::format(
                "{:#010x}-{:#010x}: No valid dispatch commands detected.",
                last_span_start,
                base_addr + completion_q_bytes);
        }
        std::cout << std::endl;
    }

    void dump_issue_queue_entries(std::ofstream &iq_file, SystemMemoryCQInterface &cq_interface) {
        chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(this->device_id);
        uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->device_id);
        // TODO: Issue Q read ptr is not prefetcly updated 0 try to read it out from chip on dump?
        uint32_t issue_read_ptr = get_cq_issue_rd_ptr<true>(this->device_id, cq_interface.id, this->cq_size) << 4;
        uint32_t issue_write_ptr = get_cq_issue_wr_ptr<true>(this->device_id, cq_interface.id, this->cq_size) << 4;
        uint32_t issue_q_bytes = cq_interface.issue_fifo_size << 4;
        uint32_t issue_q_base_addr = cq_interface.offset + CQ_START;

        // Read out in 4K pages, could do ISSUE_Q_ALIGNMENT chunks to match the entries but this is ~2x faster.
        vector<uint8_t> read_data;
        read_data.resize(dispatch_constants::TRANSFER_PAGE_SIZE);
        tt::log_info("Reading Device {} CQ {}, Issue Queue...", this->device_id, cq_interface.id);
        iq_file << fmt::format(
            "Device {}, CQ {}, Issue Queue: write_ptr={:#010x}, read_ptr={:#010x} (read_ptr may not be up to date)\n",
            this->device_id,
            cq_interface.id,
            issue_write_ptr,
            issue_read_ptr);
        uint32_t last_span_start;
        bool last_span_invalid = false;
        print_progress_bar(0.0, true);
        uint32_t first_page_addr = issue_q_base_addr - (issue_q_base_addr % dispatch_constants::TRANSFER_PAGE_SIZE);
        uint32_t end_of_curr_page =
            first_page_addr + dispatch_constants::TRANSFER_PAGE_SIZE - 1;  // To track offset of latest page read out
        tt::Cluster::instance().read_sysmem(
            read_data.data(), read_data.size(), first_page_addr, mmio_device_id, channel);
        for (uint32_t offset = 0; offset < issue_q_bytes;) { // offset increments at end of loop
            uint32_t curr_addr = issue_q_base_addr + offset;
            uint32_t page_offset = curr_addr % dispatch_constants::TRANSFER_PAGE_SIZE;

            // Check if we need to read a new page
            if (curr_addr > end_of_curr_page) {
                uint32_t page_base = curr_addr - (curr_addr % dispatch_constants::TRANSFER_PAGE_SIZE);
                tt::Cluster::instance().read_sysmem(
                    read_data.data(), read_data.size(), page_base, mmio_device_id, channel);
                end_of_curr_page = page_base + dispatch_constants::TRANSFER_PAGE_SIZE - 1;
            }

            // Check for a valid command id
            CQPrefetchCmd *cmd = (CQPrefetchCmd *)(read_data.data() + page_offset);
            if (cmd->base.cmd_id < CQ_PREFETCH_CMD_MAX_COUNT && cmd->base.cmd_id != CQ_PREFETCH_CMD_ILLEGAL) {
                if (last_span_invalid) {
                    if (curr_addr == last_span_start + dispatch_constants::ISSUE_Q_ALIGNMENT) {
                        iq_file << fmt::format("{:#010x}: No valid prefetch commands detected.", last_span_start);
                    } else {
                        iq_file << fmt::format(
                            "{:#010x}-{:#010x}: No valid prefetch commands detected.",
                            last_span_start,
                            curr_addr - dispatch_constants::ISSUE_Q_ALIGNMENT);
                    }
                    last_span_invalid = false;
                    if (last_span_start <= (issue_write_ptr) &&
                        curr_addr - dispatch_constants::ISSUE_Q_ALIGNMENT >= (issue_write_ptr)) {
                        iq_file << fmt::format(" << write_ptr (0x{:08x})", issue_write_ptr);
                    }
                    if (last_span_start <= (issue_read_ptr) &&
                        curr_addr - dispatch_constants::ISSUE_Q_ALIGNMENT >= (issue_read_ptr)) {
                        iq_file << fmt::format(" << read_ptr (0x{:08x})", issue_read_ptr);
                    }
                    iq_file << std::endl;
                }

                uint32_t cmd_stride = dump_prefetch_cmd(cmd, curr_addr, iq_file);

                // Check for a bad stride (happen to have a valid cmd_id, overwritten values, etc.)
                if (cmd_stride + offset >= issue_q_bytes || cmd_stride == 0 ||
                    cmd_stride % dispatch_constants::ISSUE_Q_ALIGNMENT != 0) {
                    cmd_stride = dispatch_constants::ISSUE_Q_ALIGNMENT;
                    iq_file << " (bad stride)";
                }

                if (curr_addr == issue_write_ptr)
                    iq_file << fmt::format(" << write_ptr (0x{:08x})", issue_write_ptr);
                if (curr_addr == issue_read_ptr)
                    iq_file << fmt::format(" << read_ptr (0x{:08x})", issue_read_ptr);
                iq_file << std::endl;

                // If it's a RELAY_INLINE command, then the data inside is dispatch commands, show them.
                if ((cmd->base.cmd_id == CQ_PREFETCH_CMD_RELAY_INLINE ||
                     cmd->base.cmd_id == CQ_PREFETCH_CMD_RELAY_INLINE_NOFLUSH) &&
                    cmd_stride > dispatch_constants::ISSUE_Q_ALIGNMENT) {
                    uint32_t dispatch_offset = offset + sizeof(CQPrefetchCmd);
                    uint32_t dispatch_curr_addr = issue_q_base_addr + dispatch_offset;
                    while (dispatch_offset < offset + cmd_stride) {
                        // Check if we need to read a new page
                        if (dispatch_curr_addr > end_of_curr_page) {
                            uint32_t page_base =
                                dispatch_curr_addr - (dispatch_curr_addr % dispatch_constants::TRANSFER_PAGE_SIZE);
                            tt::Cluster::instance().read_sysmem(
                                read_data.data(), read_data.size(), page_base, mmio_device_id, channel);
                            end_of_curr_page = page_base + dispatch_constants::TRANSFER_PAGE_SIZE - 1;
                        }

                        // Read the dispatch command
                        uint32_t dispatch_page_offset = dispatch_curr_addr % dispatch_constants::TRANSFER_PAGE_SIZE;
                        CQDispatchCmd *dispatch_cmd = (CQDispatchCmd *)(read_data.data() + dispatch_page_offset);
                        if (dispatch_cmd->base.cmd_id < CQ_DISPATCH_CMD_MAX_COUNT) {
                            iq_file << "  ";
                            uint32_t dispatch_cmd_stride =
                                dump_dispatch_cmd(dispatch_cmd, issue_q_base_addr + dispatch_offset, iq_file);
                            dispatch_offset += dispatch_cmd_stride;
                            iq_file << std::endl;
                        } else {
                            dispatch_offset += sizeof(CQDispatchCmd);
                        }
                    }
                    offset += cmd_stride;
                } else {
                    offset += cmd_stride;
                }
            } else {
                // If not a valid command, just move on and try the next.
                if (!last_span_invalid)
                    last_span_start = curr_addr;
                last_span_invalid = true;
                offset += dispatch_constants::ISSUE_Q_ALIGNMENT;
            }
            print_progress_bar((float)offset / issue_q_bytes + 0.005);
        }
        if (last_span_invalid) {
            iq_file << fmt::format(
                "{:#010x}-{:#010x}: No valid prefetch commands detected.",
                last_span_start,
                issue_q_base_addr + issue_q_bytes);
        }
        std::cout << std::endl;
    }

    void dump_command_queue_raw_data(
        std::ofstream &out_file, SystemMemoryCQInterface &cq_interface, cq_queue_t queue_type) {
        chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(this->device_id);
        uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->device_id);

        // The following variables depend on completion Q vs issue Q
        uint32_t write_ptr, read_ptr, bytes_to_read, base_addr;
        string queue_type_name;
        if (queue_type == CQ_COMPLETION_QUEUE) {
            write_ptr = get_cq_completion_wr_ptr<true>(this->device_id, cq_interface.id, this->cq_size) << 4;
            read_ptr = get_cq_completion_rd_ptr<true>(this->device_id, cq_interface.id, this->cq_size) << 4;
            bytes_to_read = cq_interface.completion_fifo_size << 4; // Page-aligned, Issue Q is not.
            TT_ASSERT(bytes_to_read % dispatch_constants::TRANSFER_PAGE_SIZE == 0);
            base_addr = cq_interface.issue_fifo_limit << 4;
            queue_type_name = "Completion";
        } else { // Issue Q
            write_ptr = get_cq_issue_wr_ptr<true>(this->device_id, cq_interface.id, this->cq_size) << 4;
            read_ptr = get_cq_issue_rd_ptr<true>(this->device_id, cq_interface.id, this->cq_size) << 4;
            bytes_to_read = cq_interface.issue_fifo_size << 4;
            base_addr = cq_interface.offset + CQ_START;
            queue_type_name = "Issue";
        }

        // Read out in pages
        vector<uint8_t> read_data;
        read_data.resize(dispatch_constants::TRANSFER_PAGE_SIZE);
        out_file << fmt::format(
                        "Device {}, CQ {}, {} Queue Raw Data:\n", this->device_id, cq_interface.id, queue_type_name)
                 << std::hex;
        tt::log_info(
            "Reading Device {} CQ {}, {} Queue Raw Data...", this->device_id, cq_interface.id, queue_type_name);
        print_progress_bar(0.0, true);
        for (uint32_t page_offset = 0; page_offset < bytes_to_read;
             page_offset += dispatch_constants::TRANSFER_PAGE_SIZE) {
            uint32_t page_addr = base_addr + page_offset;
            print_progress_bar((float)page_offset / bytes_to_read + 0.005);

            // Print in 16B per line
            tt::Cluster::instance().read_sysmem(read_data.data(), read_data.size(), page_addr, mmio_device_id, channel);
            TT_ASSERT(read_data.size() % 16 == 0);
            for (uint32_t line_offset = 0; line_offset < read_data.size(); line_offset += 16) {
                uint32_t line_addr = page_addr + line_offset;

                // Issue Q may not be divisible by page size, so break early if we go past the end.
                if (queue_type == CQ_ISSUE_QUEUE) {
                    if (line_addr + 16 >= base_addr + bytes_to_read) {
                        break;
                    }
                }

                out_file << "0x" << std::setfill('0') << std::setw(8) << line_addr << ": ";
                for (uint32_t idx = 0; idx < 16; idx++) {
                    uint8_t val = read_data[line_offset + idx];
                    out_file << " " << std::setfill('0') << std::setw(2) << +read_data[line_offset + idx];
                }
                if (line_addr == write_ptr)
                    out_file << fmt::format(" << write_ptr (0x{:08x})", write_ptr);
                if (line_addr == read_ptr)
                    out_file << fmt::format(" << read_ptr (0x{:08x})", read_ptr);
                out_file << std::endl;
            }
        }
        std::cout << std::endl;
    }

    void dump_cqs(std::ofstream &cq_file, std::ofstream &iq_file, bool dump_raw_data = false) {
        for (SystemMemoryCQInterface &cq_interface : this->cq_interfaces) {
            chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(this->device_id);
            // Dump completion queue + issue queue
            dump_completion_queue_entries(cq_file, cq_interface);
            dump_issue_queue_entries(iq_file, cq_interface);

            // This is really slow, so don't do it by default. It's sometimes helpful to read the raw bytes though.
            if (dump_raw_data) {
                dump_command_queue_raw_data(cq_file, cq_interface, CQ_COMPLETION_QUEUE);
                dump_command_queue_raw_data(iq_file, cq_interface, CQ_ISSUE_QUEUE);
            }
        }
    }

    uint32_t get_next_event(const uint8_t cq_id) {
        cq_to_event_locks[cq_id].lock();
        uint32_t next_event = ++this->cq_to_event[cq_id]; // Event ids start at 1
        cq_to_event_locks[cq_id].unlock();
        return next_event;
    }

    void reset_event_id(const uint8_t cq_id) {
        cq_to_event_locks[cq_id].lock();
        this->cq_to_event[cq_id] = 0;
        cq_to_event_locks[cq_id].unlock();
    }

    void increment_event_id(const uint8_t cq_id, const uint32_t val) {
        cq_to_event_locks[cq_id].lock();
        this->cq_to_event[cq_id] += val;
        cq_to_event_locks[cq_id].unlock();
    }

    void set_last_completed_event(const uint8_t cq_id, const uint32_t event_id) {
        TT_ASSERT(
            event_id >= this->cq_to_last_completed_event[cq_id],
            "Event ID is expected to increase. Wrapping not supported for sync. Completed event {} but last recorded "
            "completed event is {}",
            event_id,
            this->cq_to_last_completed_event[cq_id]);
        cq_to_event_locks[cq_id].lock();
        this->cq_to_last_completed_event[cq_id] = event_id;
        cq_to_event_locks[cq_id].unlock();
    }

    uint32_t get_last_completed_event(const uint8_t cq_id) {
        cq_to_event_locks[cq_id].lock();
        uint32_t last_completed_event = this->cq_to_last_completed_event[cq_id];
        cq_to_event_locks[cq_id].unlock();
        return last_completed_event;
    }

    void reset(const uint8_t cq_id) {
        SystemMemoryCQInterface &cq_interface = this->cq_interfaces[cq_id];
        cq_interface.issue_fifo_wr_ptr = (CQ_START + cq_interface.offset) >> 4;  // In 16B words
        cq_interface.issue_fifo_wr_toggle = 0;
        cq_interface.completion_fifo_rd_ptr = cq_interface.issue_fifo_limit;
        cq_interface.completion_fifo_rd_toggle = 0;
    }

    void set_issue_queue_size(const uint8_t cq_id, const uint32_t issue_queue_size) {
        SystemMemoryCQInterface &cq_interface = this->cq_interfaces[cq_id];
        cq_interface.issue_fifo_size = (issue_queue_size >> 4);
        cq_interface.issue_fifo_limit = (CQ_START + cq_interface.offset + issue_queue_size) >> 4;
    }

    void set_bypass_mode(const bool enable, const bool clear) {
        this->bypass_enable = enable;
        if (clear) {
            this->bypass_buffer.clear();
            this->bypass_buffer_write_offset = 0;
        }
    }

    bool get_bypass_mode() { return this->bypass_enable; }

    std::vector<uint32_t> get_bypass_data() { return std::move(this->bypass_buffer); }

    uint32_t get_issue_queue_size(const uint8_t cq_id) const { return this->cq_interfaces[cq_id].issue_fifo_size << 4; }

    uint32_t get_issue_queue_limit(const uint8_t cq_id) const {
        return this->cq_interfaces[cq_id].issue_fifo_limit << 4;
    }

    uint32_t get_completion_queue_size(const uint8_t cq_id) const {
        return this->cq_interfaces[cq_id].completion_fifo_size << 4;
    }

    uint32_t get_completion_queue_limit(const uint8_t cq_id) const {
        return this->cq_interfaces[cq_id].completion_fifo_limit << 4;
    }

    uint32_t get_issue_queue_write_ptr(const uint8_t cq_id) const {
        if (this->bypass_enable) {
            return this->bypass_buffer_write_offset;
        } else {
            return this->cq_interfaces[cq_id].issue_fifo_wr_ptr << 4;
        }
    }

    uint32_t get_completion_queue_read_ptr(const uint8_t cq_id) const {
        return this->cq_interfaces[cq_id].completion_fifo_rd_ptr << 4;
    }

    uint32_t get_completion_queue_read_toggle(const uint8_t cq_id) const {
        return this->cq_interfaces[cq_id].completion_fifo_rd_toggle;
    }

    uint32_t get_cq_size() const { return this->cq_size; }

    void *issue_queue_reserve(uint32_t cmd_size_B, const uint8_t cq_id) {
        if (this->bypass_enable) {
            uint32_t curr_size = this->bypass_buffer.size();
            uint32_t new_size = curr_size + (cmd_size_B / sizeof(uint32_t));
            this->bypass_buffer.resize(new_size);
            return (void *)((char *)this->bypass_buffer.data() + this->bypass_buffer_write_offset);
        }

        uint32_t issue_q_write_ptr = this->get_issue_queue_write_ptr(cq_id);

        const uint32_t command_issue_limit = this->get_issue_queue_limit(cq_id);
        if (issue_q_write_ptr + align(cmd_size_B, PCIE_ALIGNMENT) > command_issue_limit) {
            this->wrap_issue_queue_wr_ptr(cq_id);
            issue_q_write_ptr = this->get_issue_queue_write_ptr(cq_id);
        }

        // Currently read / write pointers on host and device assumes contiguous ranges for each channel
        // Device needs absolute offset of a hugepage to access the region of sysmem that holds a particular command
        // queue
        //  but on host, we access a region of sysmem using addresses relative to a particular channel
        //  this->cq_sysmem_start gives start of hugepage for a given channel
        //  since all rd/wr pointers include channel offset from address 0 to match device side pointers
        //  so channel offset needs to be subtracted to get address relative to channel
        // TODO: Reconsider offset sysmem offset calculations based on
        // https://github.com/tenstorrent/tt-metal/issues/4757
        void *issue_q_region = this->cq_sysmem_start + (issue_q_write_ptr - this->channel_offset);

        return issue_q_region;
    }

    void cq_write(const void *data, uint32_t size_in_bytes, uint32_t write_ptr) {
        // Currently read / write pointers on host and device assumes contiguous ranges for each channel
        // Device needs absolute offset of a hugepage to access the region of sysmem that holds a particular command
        // queue
        //  but on host, we access a region of sysmem using addresses relative to a particular channel
        //  this->cq_sysmem_start gives start of hugepage for a given channel
        //  since all rd/wr pointers include channel offset from address 0 to match device side pointers
        //  so channel offset needs to be subtracted to get address relative to channel
        // TODO: Reconsider offset sysmem offset calculations based on
        // https://github.com/tenstorrent/tt-metal/issues/4757
        void *user_scratchspace = this->cq_sysmem_start + (write_ptr - this->channel_offset);

        if (this->bypass_enable) {
            std::copy(
                (uint8_t *)data, (uint8_t *)data + size_in_bytes, (uint8_t *)this->bypass_buffer.data() + write_ptr);
        } else {
            memcpy_to_device(user_scratchspace, data, size_in_bytes);
        }
    }

    // TODO: RENAME issue_queue_stride ?
    void issue_queue_push_back(uint32_t push_size_B, const uint8_t cq_id) {
        if (this->bypass_enable) {
            this->bypass_buffer_write_offset += push_size_B;
            return;
        }

        // All data needs to be 32B aligned
        uint32_t push_size_16B = align(push_size_B, dispatch_constants::ISSUE_Q_ALIGNMENT) >> 4;

        SystemMemoryCQInterface &cq_interface = this->cq_interfaces[cq_id];

        if (cq_interface.issue_fifo_wr_ptr + push_size_16B >= cq_interface.issue_fifo_limit) {
            cq_interface.issue_fifo_wr_ptr = (CQ_START + cq_interface.offset) >> 4;     // In 16B words
            cq_interface.issue_fifo_wr_toggle = not cq_interface.issue_fifo_wr_toggle;  // Flip the toggle
        } else {
            cq_interface.issue_fifo_wr_ptr += push_size_16B;
        }

        // Also store this data in hugepages, so if a hang happens we can see what was written by host.
        chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(this->device_id);
        uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->device_id);
        tt::Cluster::instance().write_sysmem(
            &cq_interface.issue_fifo_wr_ptr,
            sizeof(uint32_t),
            HOST_CQ_ISSUE_WRITE_PTR + get_relative_cq_offset(cq_id, this->cq_size),
            mmio_device_id,
            channel
        );
    }

    void completion_queue_wait_front(const uint8_t cq_id, volatile bool &exit_condition) const {
        uint32_t write_ptr_and_toggle;
        uint32_t write_ptr;
        uint32_t write_toggle;
        const SystemMemoryCQInterface &cq_interface = this->cq_interfaces[cq_id];

        do {
            write_ptr_and_toggle = get_cq_completion_wr_ptr<true>(this->device_id, cq_id, this->cq_size);
            write_ptr = write_ptr_and_toggle & 0x7fffffff;
            write_toggle = write_ptr_and_toggle >> 31;
        } while (cq_interface.completion_fifo_rd_ptr == write_ptr and
                 cq_interface.completion_fifo_rd_toggle == write_toggle and not exit_condition);
    }

    void send_completion_queue_read_ptr(const uint8_t cq_id) const {
        const SystemMemoryCQInterface &cq_interface = this->cq_interfaces[cq_id];

        uint32_t read_ptr_and_toggle =
            cq_interface.completion_fifo_rd_ptr | (cq_interface.completion_fifo_rd_toggle << 31);
        this->fast_write_callable(
            this->completion_byte_addrs[cq_id], 4, (uint8_t *)&read_ptr_and_toggle, this->m_dma_buf_size);

        // Also store this data in hugepages in case we hang and can't get it from the device.
        chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(this->device_id);
        uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->device_id);
        tt::Cluster::instance().write_sysmem(
            &read_ptr_and_toggle,
            sizeof(uint32_t),
            HOST_CQ_COMPLETION_READ_PTR + get_relative_cq_offset(cq_id, this->cq_size),
            mmio_device_id,
            channel
        );
    }

    void wrap_issue_queue_wr_ptr(const uint8_t cq_id) {
        if (this->bypass_enable)
            return;
        SystemMemoryCQInterface &cq_interface = this->cq_interfaces[cq_id];
        cq_interface.issue_fifo_wr_ptr = (CQ_START + cq_interface.offset) >> 4;
        cq_interface.issue_fifo_wr_toggle = not cq_interface.issue_fifo_wr_toggle;
    }

    void wrap_completion_queue_rd_ptr(const uint8_t cq_id) {
        SystemMemoryCQInterface &cq_interface = this->cq_interfaces[cq_id];
        cq_interface.completion_fifo_rd_ptr = cq_interface.issue_fifo_limit;
        cq_interface.completion_fifo_rd_toggle = not cq_interface.completion_fifo_rd_toggle;
    }

    void completion_queue_pop_front(uint32_t num_pages_read, const uint8_t cq_id) {
        uint32_t data_read_B = num_pages_read * dispatch_constants::TRANSFER_PAGE_SIZE;
        uint32_t data_read_16B = data_read_B >> 4;

        SystemMemoryCQInterface &cq_interface = this->cq_interfaces[cq_id];
        cq_interface.completion_fifo_rd_ptr += data_read_16B;
        if (cq_interface.completion_fifo_rd_ptr >= cq_interface.completion_fifo_limit) {
            cq_interface.completion_fifo_rd_ptr = cq_interface.issue_fifo_limit;
            cq_interface.completion_fifo_rd_toggle = not cq_interface.completion_fifo_rd_toggle;
        }

        // Notify dispatch core
        this->send_completion_queue_read_ptr(cq_id);
    }

    void fetch_queue_reserve_back(const uint8_t cq_id) {
        if (this->bypass_enable)
            return;

        // Helper to wait for fetch queue space, if needed
        auto wait_for_fetch_q_space = [&]() {
            // The condition for doing a read from device, no space in fetch queue
            bool read_from_device = (this->prefetch_q_dev_ptrs[cq_id] == this->prefetch_q_dev_fences[cq_id]);
            std::vector<uint32_t> prefetch_q_rd_ptrs;
            prefetch_q_rd_ptrs.resize(2); // Fence, pcie addr

            // Loop until space frees up
            while (this->prefetch_q_dev_ptrs[cq_id] == this->prefetch_q_dev_fences[cq_id]) {
                tt::Cluster::instance().read_core(
                    prefetch_q_rd_ptrs,
                    prefetch_q_rd_ptrs.size() * sizeof(uint32_t),
                    this->prefetcher_cores[cq_id],
                    CQ_PREFETCH_Q_RD_PTR);
                this->prefetch_q_dev_fences[cq_id] = prefetch_q_rd_ptrs[0];
            }

            // If we did a read, prefetch_q_rd_ptrs[1] holds how far the device has read. Save it in hugepages in case
            // we have an error and need to dump debug data.
            if (read_from_device) {
                chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(this->device_id);
                uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->device_id);
                prefetch_q_rd_ptrs[1] >>= 4; // CQ pointers are to 16B chunks
                tt::Cluster::instance().write_sysmem(
                    &prefetch_q_rd_ptrs[1],
                    sizeof(uint32_t),
                    HOST_CQ_ISSUE_READ_PTR + get_relative_cq_offset(cq_id, this->cq_size),
                    mmio_device_id,
                    channel);
            }
        };

        wait_for_fetch_q_space();

        // Wrap FetchQ if possible
        CoreType core_type = dispatch_core_manager::get(num_hw_cqs).get_dispatch_core_type(device_id);
        uint32_t prefetch_q_base = DISPATCH_L1_UNRESERVED_BASE;
        uint32_t prefetch_q_limit = prefetch_q_base + dispatch_constants::get(core_type).prefetch_q_entries() *
                                                          sizeof(dispatch_constants::prefetch_q_entry_type);
        if (this->prefetch_q_dev_ptrs[cq_id] == prefetch_q_limit) {
            this->prefetch_q_dev_ptrs[cq_id] = prefetch_q_base;
            wait_for_fetch_q_space();
        }
    }

    void fetch_queue_write(uint32_t command_size_B, const uint8_t cq_id, bool stall_prefetcher = false) {
        CoreType dispatch_core_type =
            dispatch_core_manager::get(this->num_hw_cqs).get_dispatch_core_type(this->device_id);
        uint32_t max_command_size_B = dispatch_constants::get(dispatch_core_type).max_prefetch_command_size();
        TT_ASSERT(
            command_size_B <= max_command_size_B,
            "Generated prefetcher command of size {} B exceeds max command size {} B",
            command_size_B,
            max_command_size_B);
        TT_ASSERT(
            (command_size_B >> dispatch_constants::PREFETCH_Q_LOG_MINSIZE) < 0xFFFF,
            "FetchQ command too large to represent");
        if (this->bypass_enable)
            return;
        tt_driver_atomics::sfence();
        dispatch_constants::prefetch_q_entry_type command_size_16B = command_size_B >> dispatch_constants::PREFETCH_Q_LOG_MINSIZE;

        // stall_prefetcher is used for enqueuing traces, as replaying a trace will hijack the cmd_data_q
        // so prefetcher fetches multiple cmds that include the trace cmd, they will be corrupted by trace pulling data
        // from DRAM stall flag prevents pulling prefetch q entries that occur after the stall entry Stall flag for
        // prefetcher is MSB of FetchQ entry.
        if (stall_prefetcher) {
            command_size_16B |= (1 << ((sizeof(dispatch_constants::prefetch_q_entry_type) * 8) - 1));
        }
        this->prefetch_q_writers[cq_id].write(this->prefetch_q_dev_ptrs[cq_id], command_size_16B);
        this->prefetch_q_dev_ptrs[cq_id] += sizeof(dispatch_constants::prefetch_q_entry_type);
    }
};
