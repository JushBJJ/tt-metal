// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>

#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/llrt/rtoptions.hpp"
#include "tt_metal/impl/dispatch/kernels/cq_cmds.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"
#include "common.h"

constexpr uint32_t DEFAULT_ITERATIONS = 10000;
constexpr uint32_t DEFAULT_WARMUP_ITERATIONS = 100;
constexpr uint32_t DEFAULT_DISPATCH_BUFFER_LOG_PAGE_SIZE = 12;
constexpr uint32_t DEFAULT_DISPATCH_BUFFER_SIZE_BLOCKS = 4;
constexpr uint32_t DEFAULT_DISPATCH_BUFFER_BLOCK_SIZE_PAGES = 768 * 1024 / (1 << DEFAULT_DISPATCH_BUFFER_LOG_PAGE_SIZE) / DEFAULT_DISPATCH_BUFFER_SIZE_BLOCKS;
constexpr uint32_t DEFAULT_PREFETCHER_BUFFER_SIZE_PAGES = 768 * 1024 / (1 << DEFAULT_DISPATCH_BUFFER_LOG_PAGE_SIZE);
constexpr uint32_t MAX_XFER_SIZE_16B = 4 * 1024;
constexpr uint32_t MIN_XFER_SIZE_16B = 1;
constexpr uint32_t DEFAULT_PREFETCHER_PAGE_BATCH_SIZE = 1;

//////////////////////////////////////////////////////////////////////////////////////////
// Test dispatch program performance
//
// Times dispatching program to M cores, N processors, of various sizes, CBs, runtime args
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;

uint32_t iterations_g = DEFAULT_ITERATIONS;
uint32_t warmup_iterations_g = DEFAULT_WARMUP_ITERATIONS;
uint32_t prefetcher_iterations_g = 1;

uint32_t log_dispatch_buffer_page_size_g = DEFAULT_DISPATCH_BUFFER_LOG_PAGE_SIZE;
uint32_t dispatch_buffer_page_size_g = 1 << DEFAULT_DISPATCH_BUFFER_LOG_PAGE_SIZE;
uint32_t dispatch_buffer_block_size_pages_g = DEFAULT_DISPATCH_BUFFER_BLOCK_SIZE_PAGES;
uint32_t dispatch_buffer_size_blocks_g = DEFAULT_DISPATCH_BUFFER_SIZE_BLOCKS;
uint32_t dispatch_buffer_size_g = 0;
uint32_t prefetcher_buffer_size_g = 0;
uint32_t prefetcher_page_batch_size_g = DEFAULT_PREFETCHER_PAGE_BATCH_SIZE;
uint32_t max_xfer_size_bytes_g = MAX_XFER_SIZE_16B << 4;
uint32_t min_xfer_size_bytes_g = MIN_XFER_SIZE_16B << 4;
bool debug_g;
bool lazy_g;
bool fire_once_g;

void init(int argc, char **argv) {
    std::vector<std::string> input_args(argv, argv + argc);

    if (test_args::has_command_option(input_args, "-h") ||
        test_args::has_command_option(input_args, "--help")) {
        log_info(LogTest, "Usage:");
        log_info(LogTest, "  -w: warm-up iterations before starting timer (default {}), ", DEFAULT_WARMUP_ITERATIONS);
        log_info(LogTest, "  -i: host iterations (default {})", DEFAULT_ITERATIONS);
        log_info(LogTest, "  -pi: prefetcher iterations (looping on device) (default {})", 1);
        log_info(LogTest, "  -lps: log of page size of prefetch/dispatch buffer (default {})", DEFAULT_DISPATCH_BUFFER_LOG_PAGE_SIZE);
        log_info(LogTest, "  -bs: dispatcher block size in pages (default {})", DEFAULT_DISPATCH_BUFFER_BLOCK_SIZE_PAGES);
        log_info(LogTest, "  -b: dispatcher buffer size in blocks (default {})", DEFAULT_DISPATCH_BUFFER_SIZE_BLOCKS);
        log_info(LogTest, "  -pbs: prefetcher buffer size pages (default {})", DEFAULT_PREFETCHER_BUFFER_SIZE_PAGES);
        log_info(LogTest, "  -ppbs: prefetcher page batch size (process pages in batches of N to reduce overhead) (default {})", DEFAULT_PREFETCHER_PAGE_BATCH_SIZE);
        log_info(LogTest, "  -max: max transfer size (default {})", MAX_XFER_SIZE_16B << 4);
        log_info(LogTest, "  -min: min transfer size (default {})", MIN_XFER_SIZE_16B << 4);
        log_info(LogTest, "  -f: prefetcher fire once, use to measure dispatcher perf w/ prefetcher out of the way (default disabled)");
        log_info(LogTest, "  -d: wrap all commands in debug commands (default disabled)");
        log_info(LogTest, "  -z: enable dispatch lazy mode (default disabled)");
        exit(0);
    }

    warmup_iterations_g = test_args::get_command_option_uint32(input_args, "-w", DEFAULT_WARMUP_ITERATIONS);
    iterations_g = test_args::get_command_option_uint32(input_args, "-i", DEFAULT_ITERATIONS);
    prefetcher_iterations_g = test_args::get_command_option_uint32(input_args, "-pi", 1);

    log_dispatch_buffer_page_size_g = test_args::get_command_option_uint32(input_args, "-lps", DEFAULT_DISPATCH_BUFFER_LOG_PAGE_SIZE);
    dispatch_buffer_page_size_g = 1 << log_dispatch_buffer_page_size_g;
    dispatch_buffer_block_size_pages_g = test_args::get_command_option_uint32(input_args, "-bs", DEFAULT_DISPATCH_BUFFER_BLOCK_SIZE_PAGES);
    dispatch_buffer_size_blocks_g = test_args::get_command_option_uint32(input_args, "-b", DEFAULT_DISPATCH_BUFFER_SIZE_BLOCKS);
    dispatch_buffer_size_g = dispatch_buffer_page_size_g * dispatch_buffer_block_size_pages_g * dispatch_buffer_size_blocks_g;

    prefetcher_page_batch_size_g = test_args::get_command_option_uint32(input_args, "-ppbs", DEFAULT_PREFETCHER_PAGE_BATCH_SIZE);

    uint32_t pbs_pages = test_args::get_command_option_uint32(input_args, "-pbs", DEFAULT_PREFETCHER_BUFFER_SIZE_PAGES);
    uint32_t terminate_cmd_pages = 1;
    // divide the batch size evenlly, one page for terminate
    pbs_pages = pbs_pages / prefetcher_page_batch_size_g * prefetcher_page_batch_size_g + terminate_cmd_pages;
    prefetcher_buffer_size_g = pbs_pages * dispatch_buffer_page_size_g;

    max_xfer_size_bytes_g = test_args::get_command_option_uint32(input_args, "-max", MAX_XFER_SIZE_16B << 4);
    min_xfer_size_bytes_g = test_args::get_command_option_uint32(input_args, "-min", MIN_XFER_SIZE_16B << 4);
    max_xfer_size_bytes_g = (max_xfer_size_bytes_g >> 4) << 4;
    min_xfer_size_bytes_g = (min_xfer_size_bytes_g >> 4) << 4;

    fire_once_g = test_args::has_command_option(input_args, "-f");
    if (fire_once_g) {
        if (prefetcher_buffer_size_g != dispatch_buffer_size_g + terminate_cmd_pages) {
            log_info(LogTest, "Fire once overriding prefetcher buffer size");
            prefetcher_buffer_size_g = dispatch_buffer_size_g + terminate_cmd_pages * dispatch_buffer_page_size_g;
        }
    }

    debug_g = test_args::has_command_option(input_args, "-d");
    lazy_g = test_args::has_command_option(input_args, "-z");
}

void gen_cmds(vector<uint32_t>& dispatch_cmds,
              CoreCoord worker_core,
              vector<uint32_t>& worker_data,
              uint32_t scratch_addr,
              uint32_t page_size) {

    uint32_t total_size_bytes = 0;
    uint32_t total_data_size_bytes = 0;
    uint32_t buffer_size = prefetcher_buffer_size_g - page_size; // for terminate
    uint32_t cmd_count = 0;

    while (total_size_bytes < buffer_size) {
        total_size_bytes += sizeof(CQDispatchCmd);
        if (debug_g) {
            total_size_bytes += sizeof(CQDispatchCmd);
        }
        uint32_t xfer_size_16B = (std::rand() & (MAX_XFER_SIZE_16B - 1));
        if (total_size_bytes + (xfer_size_16B << 4) > buffer_size) {
            xfer_size_16B = (buffer_size - total_size_bytes) >> 4;
        }
        uint32_t xfer_size_bytes = xfer_size_16B << 4;
        if (xfer_size_bytes > max_xfer_size_bytes_g) xfer_size_bytes = max_xfer_size_bytes_g;
        if (xfer_size_bytes < min_xfer_size_bytes_g) xfer_size_bytes = min_xfer_size_bytes_g;

        gen_dispatcher_write_cmd(dispatch_cmds, worker_data, worker_core,
                                 scratch_addr + total_data_size_bytes, xfer_size_bytes);
        uint32_t page_size_words = page_size / sizeof(uint32_t);
        dispatch_cmds.resize((dispatch_cmds.size() + page_size_words - 1) / page_size_words * page_size_words);    // pad to page

        total_data_size_bytes += xfer_size_bytes;
        total_size_bytes = dispatch_cmds.size() * sizeof(uint32_t);
        cmd_count++;
    }

    gen_dispatcher_terminate_cmd(dispatch_cmds);
    uint32_t page_size_words = page_size / sizeof(uint32_t);
    dispatch_cmds.resize((dispatch_cmds.size() + page_size_words - 1) / page_size_words * page_size_words);    // pad to page
    cmd_count++;

    log_info(LogTest, "Generated {} commands", cmd_count);
}

int main(int argc, char **argv) {
    init(argc, argv);

    uint32_t dispatch_buffer_pages = dispatch_buffer_size_g / dispatch_buffer_page_size_g;

    bool pass = true;
    try {
        int device_id = 0;
        tt_metal::Device *device = tt_metal::CreateDevice(device_id);

        CommandQueue& cq = device->command_queue();

        tt_metal::Program program = tt_metal::CreateProgram();

        CoreCoord spoof_prefetch_core = {0, 0};
        CoreCoord dispatch_core = {0, 1};
        CoreCoord worker_core = {0, 2};

        CoreCoord phys_spoof_prefetch_core = device->worker_core_from_logical_core(spoof_prefetch_core);
        CoreCoord phys_dispatch_core = device->worker_core_from_logical_core(dispatch_core);
        CoreCoord phys_worker_core = device->worker_core_from_logical_core(worker_core);

        // Want different buffers on each core, instead use big buffer and self-manage it
        uint32_t l1_buf_base = L1_UNRESERVED_BASE;
        TT_ASSERT((l1_buf_base & (dispatch_buffer_page_size_g - 1)) == 0);
        if (prefetcher_buffer_size_g + l1_buf_base > 1024 * 1024) {
            log_fatal(LogTest, "Error, prefetcher buffer size too large\n");
            exit(-1);
        }
#if 0
        Buffer l1_buf(device, prefetcher_buffer_size_g, prefetcher_buffer_size_g, BufferType::L1, TensorMemoryLayout::SINGLE_BANK);
#endif
        vector<uint32_t> cmds;
        vector<uint32_t> worker_data;
        gen_cmds(cmds, phys_worker_core, worker_data, l1_buf_base, dispatch_buffer_page_size_g);
        llrt::write_hex_vec_to_core(device->id(), phys_spoof_prefetch_core, cmds, l1_buf_base);

        std::map<string, string> defines = {
            {"PREFETCH_NOC_X", std::to_string(phys_spoof_prefetch_core.x)},
            {"PREFETCH_NOC_Y", std::to_string(phys_spoof_prefetch_core.y)},
            {"DISPATCH_NOC_X", std::to_string(phys_dispatch_core.x)},
            {"DISPATCH_NOC_Y", std::to_string(phys_dispatch_core.y)},
        };
        if (fire_once_g) {
            defines.insert(std::pair<string, string>("FIRE_ONCE", std::to_string(1)));
        }

        constexpr uint32_t dispatch_cb_sem = 0;
        tt_metal::CreateSemaphore(program, {spoof_prefetch_core}, dispatch_buffer_pages);
        tt_metal::CreateSemaphore(program, {dispatch_core}, 0);

        std::vector<uint32_t> dispatch_compile_args =
            {l1_buf_base,
             log_dispatch_buffer_page_size_g,
             dispatch_buffer_size_g / dispatch_buffer_page_size_g,
             dispatch_cb_sem,
             dispatch_buffer_size_blocks_g
            };
        std::vector<uint32_t> spoof_prefetch_compile_args =
            {l1_buf_base,
             log_dispatch_buffer_page_size_g,
             dispatch_buffer_pages,
             dispatch_cb_sem,
             l1_buf_base,
             (uint32_t)(cmds.size() * sizeof(uint32_t)) / dispatch_buffer_page_size_g,
             prefetcher_page_batch_size_g,
            };

        auto sp1 = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/perf_microbenchmark/dispatch/kernels/spoof_prefetch.cpp",
            {spoof_prefetch_core},
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::RISCV_0_default,
                .compile_args = spoof_prefetch_compile_args,
                .defines = defines
            }
        );
        vector<uint32_t> args;
        args.push_back(prefetcher_iterations_g);
        tt_metal::SetRuntimeArgs(program, sp1, spoof_prefetch_core, args);

        auto d1 = tt_metal::CreateKernel(
            program,
            "tt_metal/impl/dispatch/kernels/cq_dispatch.cpp",
            {dispatch_core},
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::RISCV_0_default,
                .compile_args = dispatch_compile_args,
                .defines = defines
            }
        );

        log_info(LogTest, "Dispatch buffer size blocks {}", std::to_string(dispatch_buffer_size_blocks_g));
        log_info(LogTest, "Dispatch buffer block size pages {}", std::to_string(dispatch_buffer_block_size_pages_g));
        log_info(LogTest, "Dispatch buffer page size {}", std::to_string(dispatch_buffer_page_size_g));
        log_info(LogTest, "Dispatch buffer pages {}", std::to_string(dispatch_buffer_pages));
        log_info(LogTest, "Dispatch buffer size {}", std::to_string(dispatch_buffer_page_size_g * dispatch_buffer_pages));
        log_info(LogTest, "Dispatch buffer base {}", std::to_string(l1_buf_base));
        log_info(LogTest, "Dispatch buffer end {}", std::to_string(l1_buf_base + dispatch_buffer_page_size_g * dispatch_buffer_pages));
        log_info(LogTest, "Prefetcher CMD Buffer size {}", std::to_string(prefetcher_buffer_size_g));
        log_info(LogTest, "Worker result data size {} bytes", std::to_string(worker_data.size() * sizeof(uint32_t)));

        // Cache stuff
        for (int i = 0; i < warmup_iterations_g; i++) {
            EnqueueProgram(cq, program, false);
        }
        Finish(cq);

        if (lazy_g) {
            tt_metal::detail::SetLazyCommandQueueMode(true);
        }

        auto start = std::chrono::system_clock::now();
        for (int i = 0; i < iterations_g; i++) {
            EnqueueProgram(cq, program, false);
        }
        if (lazy_g) {
            start = std::chrono::system_clock::now();
        }
        Finish(cq);
        auto end = std::chrono::system_clock::now();

        pass &= validate_results(device, phys_worker_core, worker_data, l1_buf_base);

        std::chrono::duration<double> elapsed_seconds = (end-start);
        log_info(LogTest, "Ran in {}us", elapsed_seconds.count() * 1000 * 1000);
        log_info(LogTest, "Ran in {}us per iteration", elapsed_seconds.count() * 1000 * 1000 / iterations_g);
        if (iterations_g == 1) {
            float bw = (float)worker_data.size() * sizeof(uint32_t) * prefetcher_iterations_g / (elapsed_seconds.count() * 1000.0 * 1000.0 * 1000.0);
            std::stringstream ss;
            ss << std::fixed << std::setprecision(3) << bw;
            log_info(LogTest, "BW: {} GB/s", ss.str());
        } else {
            log_info(LogTest, "BW: -- GB/s (use -i 1 to report bandwidth)");
        }
        pass &= tt_metal::CloseDevice(device);
    } catch (const std::exception& e) {
        pass = false;
        log_fatal(e.what());
    }

    tt::llrt::OptionsG.set_kernels_nullified(false);

    if (pass) {
        log_info(LogTest, "Test Passed");
        return 0;
    } else {
        log_fatal(LogTest, "Test Failed\n");
        return 1;
    }
}
