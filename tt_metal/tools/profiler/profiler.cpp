// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <fstream>
#include <iomanip>
#include <filesystem>


#include "tt_metal/host_api.hpp"
#include "tools/profiler/profiler.hpp"
#include "tools/profiler/profiler_state.hpp"
#include "hostdevcommon/profiler_common.h"
#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"

#define HOST_SIDE_LOG "profile_log_host.csv"
#define DEVICE_SIDE_LOG "profile_log_device.csv"

namespace tt {

namespace tt_metal {

TimerPeriodInt Profiler::timerToTimerInt(TimerPeriod period)
{
    TimerPeriodInt ret;

    ret.start = duration_cast<nanoseconds>(period.start.time_since_epoch()).count();
    ret.stop = duration_cast<nanoseconds>(period.stop.time_since_epoch()).count();
    ret.delta = duration_cast<nanoseconds>(period.stop - period.start).count();

    return ret;
}

void Profiler::dumpHostResults(const std::string& timer_name, const std::vector<std::pair<std::string,std::string>>& additional_fields)
{
    auto timer = name_to_timer_map[timer_name];

    auto timer_period_ns = timerToTimerInt(timer);
    TT_ASSERT (timer_period_ns.start != 0 , "Timer start cannot be zero on : " + timer_name);
    TT_ASSERT (timer_period_ns.stop != 0 , "Timer stop cannot be zero on : " + timer_name);

    std::filesystem::path log_path = output_dir / HOST_SIDE_LOG;
    std::ofstream log_file;

    if (host_new_log|| !std::filesystem::exists(log_path))
    {
        log_file.open(log_path);

        log_file << "Name" << ", ";
        log_file << "Start timer count [ns]"  << ", ";
        log_file << "Stop timer count [ns]"  << ", ";
        log_file << "Delta timer count [ns]";

        for (auto &field: additional_fields)
        {
            log_file  << ", "<< field.first;
        }

        log_file << std::endl;
        host_new_log = false;
    }
    else
    {
        log_file.open(log_path, std::ios_base::app);
    }

    log_file << timer_name << ", ";
    log_file << timer_period_ns.start  << ", ";
    log_file << timer_period_ns.stop  << ", ";
    log_file << timer_period_ns.delta;

    for (auto &field: additional_fields)
    {
        log_file  << ", "<< field.second;
    }

    log_file << std::endl;

    log_file.close();
}

void Profiler::readRiscProfilerResults(
        int device_id,
        vector<std::uint32_t> profile_buffer,
        const CoreCoord &worker_core){

    ZoneScoped;
    uint32_t core_flat_id = get_flat_id(worker_core.x, worker_core.y);
    uint32_t startIndex = core_flat_id * PROFILER_RISC_COUNT * PROFILER_L1_VECTOR_SIZE;

    vector<std::uint32_t> control_buffer;

    control_buffer = tt::llrt::read_hex_vec_from_core(
            device_id,
            worker_core,
            PROFILER_L1_BUFFER_CONTROL,
            PROFILER_L1_CONTROL_BUFFER_SIZE);

    uint32_t bufferCount = control_buffer[kernel_profiler::DRAM_BUFFER_NUM];

    //for (auto& i : control_buffer)
    //{
        //std::cout << i << ", ";
    //}

    //std::cout << std::endl;

    if (bufferCount > PROFILER_DRAM_BUFFER_COUNT)
    {
        bufferCount = PROFILER_DRAM_BUFFER_COUNT;
        std::cout << "DRAM Buffer for " << worker_core.x << "," << worker_core.y << " is full" << std::endl;
    }

    if (bufferCount > 0)
    {
//#define DEBUG_PRINT_L1
#ifdef DEBUG_PRINT_L1
        vector<std::uint32_t> profile_buffer_l1;

        profile_buffer_l1 = tt::llrt::read_hex_vec_from_core(
                device_id,
                worker_core,
                PROFILER_L1_BUFFER_BR,
                PROFILER_L1_BUFFER_SIZE * PROFILER_RISC_COUNT);

        std::cout << worker_core.x << "," << worker_core.y <<  "," << core_flat_id << "," << startIndex <<  std::endl ;
        for (int i= 0; i < 6; i ++)
        {
            std::cout << profile_buffer_l1[i] << ",";
        }
        std::cout <<  std::endl;
        for (int i= 0; i < 6; i ++)
        {
            std::cout << profile_buffer[startIndex + i] << ",";
        }
        std::cout <<  std::endl;
        std::cout <<  std::endl;
#endif
    dumpDeviceResultToFile(
            device_id,
            worker_core.x,
            worker_core.y,
            0,
            (uint64_t(control_buffer[kernel_profiler::FW_RESET_H]) << 32) | control_buffer[kernel_profiler::FW_RESET_L],
            0);

    }
    for (int riscNum = 0; riscNum < PROFILER_RISC_COUNT; riscNum++) {
        for (int bufferNum = 0; bufferNum < bufferCount; bufferNum++){

            uint32_t buffer_shift = bufferNum * PROFILER_L1_VECTOR_SIZE * PROFILER_RISC_COUNT * 120 + startIndex;
            uint32_t time_L_pre = 0;
            uint32_t time_H_index = buffer_shift + riscNum * PROFILER_L1_VECTOR_SIZE + kernel_profiler::SYNC_VAL_H;
            uint32_t time_H = profile_buffer[time_H_index];

            for (int marker = kernel_profiler::SYNC_VAL_L; marker <= kernel_profiler::FW_END; marker++)
            {
                uint32_t time_L_index = buffer_shift + riscNum * PROFILER_L1_VECTOR_SIZE + marker;
                uint32_t time_L = profile_buffer[time_L_index];

                // Was there an overflow
                if (time_L_pre > time_L)
                {
                    time_H ++;
                }
                time_L_pre = time_L;

                if (marker > kernel_profiler::SYNC_VAL_L)
                {
                    dumpDeviceResultToFile(
                            device_id,
                            worker_core.x,
                            worker_core.y,
                            riscNum,
                            (uint64_t(time_H) << 32) | time_L,
                            marker - kernel_profiler::SYNC_VAL_L);
                }
            }
        }
    }

    std::vector<uint32_t> zero_buffer(kernel_profiler::DRAM_BUFFER_NUM + 1, 0);
    tt::llrt::write_hex_vec_to_core(
            device_id,
            worker_core,
            zero_buffer,
            PROFILER_L1_BUFFER_CONTROL);
}

void Profiler::dumpDeviceResultToFile(
        int chip_id,
        int core_x,
        int core_y,
        int risc,
        uint64_t timestamp,
        uint32_t timer_id){
    ZoneScoped;
    auto test  = std::filesystem::path("tt_metal/tools/profiler/logs");
    std::filesystem::path log_path = test / DEVICE_SIDE_LOG;
    std::ofstream log_file;

    std::string riscName[] = {"BRISC", "NCRISC", "TRISC_0", "TRISC_1", "TRISC_2"};


    if (device_new_log || !std::filesystem::exists(log_path))
    {
        log_file.open(log_path);
        log_file << "ARCH: " << get_string_lowercase(device_architecture) << ", CHIP_FREQ[MHz]: " << device_core_frequency << std::endl;
        log_file << "PCIe slot, core_x, core_y, RISC processor type, timer_id, time[cycles since reset]" << std::endl;
        device_new_log = false;
    }
    else
    {
        log_file.open(log_path, std::ios_base::app);
    }

    constexpr int DRAM_ROW = 6;
    if (core_y > DRAM_ROW){
       core_y = core_y - 2;
    }
    else{
       core_y--;
    }
    core_x--;

    uint64_t threadID = core_x*1000000+core_y*10000+risc*100;
    uint64_t eventID = timer_id + threadID;

    if (device_data.find (eventID) != device_data.end())
    {
        ZoneScopedNC("eventFound",tracy::Color::Green);
        device_data.at(eventID).push_back(timestamp);
    }
    else
    {
        ZoneScopedNC("eventNotFound",tracy::Color::Red);
        device_data.emplace(eventID,std::list<uint64_t>{timestamp});
    }

    log_file << chip_id << ", " << core_x << ", " << core_y << ", " << riscName[risc] << ", ";
    log_file << timer_id << ", ";
    log_file << timestamp;
    log_file << std::endl;
    log_file.close();
}

Profiler::Profiler()
{
#if defined(PROFILER)
    ZoneScopedC(tracy::Color::Green);
    host_new_log = true;
    device_new_log = true;
    output_dir = std::filesystem::path("tt_metal/tools/profiler/logs");
    std::filesystem::create_directories(output_dir);

    tracyTTCtx = TracyCLContext();
#endif
}

Profiler::~Profiler()
{
#if defined(PROFILER)
    TracyCLDestroy(tracyTTCtx);
    std::cout << "Destroy global profiler" << std::endl ;
#endif
}

void Profiler::markStart(const std::string& timer_name)
{
#if defined(PROFILER)
    name_to_timer_map[timer_name].start = steady_clock::now();
#endif
}

void Profiler::markStop(const std::string& timer_name, const std::vector<std::pair<std::string,std::string>>& additional_fields)
{
#if defined(PROFILER)
    name_to_timer_map[timer_name].stop = steady_clock::now();
    dumpHostResults(timer_name, additional_fields);
#endif
}

void Profiler::setDeviceNewLogFlag(bool new_log_flag)
{
#if defined(PROFILER)
    device_new_log = new_log_flag;
#endif
}

void Profiler::setHostNewLogFlag(bool new_log_flag)
{
#if defined(PROFILER)
    host_new_log = new_log_flag;
#endif
}

void Profiler::setOutputDir(const std::string& new_output_dir)
{
#if defined(PROFILER)
    std::filesystem::create_directories(new_output_dir);
    output_dir = new_output_dir;
#endif
}

void Profiler::setDeviceArchitecture(tt::ARCH device_arch)
{
#if defined(PROFILER)
    device_architecture = device_arch;
#endif
}

void Profiler::dumpDeviceResults (
        int device_id,
        const vector<CoreCoord> &worker_cores){
#if defined(PROFILER)
    ZoneScoped;
    device_core_frequency = tt::Cluster::instance().get_device_aiclk(device_id);
    std::vector<uint32_t> profile_buffer(PROFILER_FULL_BUFFER_SIZE/sizeof(uint32_t), 0);
    //tt_metal::ReadFromBuffer(profiler_dram_buffer, profile_buffer);

    tt::Cluster::instance().read_sysmem_vec(profile_buffer, PROFILER_HUGE_PAGE_ADDRESS, PROFILER_FULL_BUFFER_SIZE, 0);

    for (const auto &worker_core : worker_cores) {
        readRiscProfilerResults(
            device_id,
            profile_buffer,
            worker_core);
    }
#endif
}

bool getHostProfilerState ()
{
    bool profile_host = false;
#if defined(PROFILER)
    profile_host = true;
#endif
    return profile_host;
}

bool getDeviceProfilerState ()
{
    bool profile_device = false;
#if defined(PROFILER)
    const char *TT_METAL_DEVICE_PROFILER = std::getenv("TT_METAL_DEVICE_PROFILER");
    if (TT_METAL_DEVICE_PROFILER != nullptr && TT_METAL_DEVICE_PROFILER[0] == '1')
    {
        profile_device = true;
    }
#endif
    return profile_device;
}

}  // namespace tt_metal

}  // namespace tt
