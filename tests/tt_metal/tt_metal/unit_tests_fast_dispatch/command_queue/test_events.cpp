// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "command_queue_fixture.hpp"
#include "command_queue_test_utils.hpp"
#include "gtest/gtest.h"
#include "tt_metal/common/bfloat16.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "impl/debug/watcher_server.hpp"
#include "tt_metal/detail/tt_metal.hpp"

using namespace tt::tt_metal;

TEST_F(CommandQueueFixture, TestEventsWrittenToCompletionQueueInOrder) {
    size_t num_buffers = 100;
    uint32_t page_size = 2048;
    vector<uint32_t> page(page_size / sizeof(uint32_t));
    uint32_t expected_event_id = 0;

    auto current_mode = CommandQueue::default_mode();
    for (const CommandQueue::CommandQueueMode mode : {CommandQueue::CommandQueueMode::PASSTHROUGH, CommandQueue::CommandQueueMode::ASYNC}) {
        tt::log_info(tt::LogTest, "Using CQ Mode: {}", mode);
        this->device_->command_queue().set_mode(mode);
        auto start = std::chrono::system_clock::now();

        uint32_t completion_queue_base = this->device_->sysmem_manager().get_completion_queue_read_ptr(0);
        chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(this->device_->id());
        uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->device_->id());
        constexpr uint32_t completion_queue_event_alignment = 32;
        for (size_t i = 0; i < num_buffers; i++) {
            std::shared_ptr<Buffer> buf = std::make_shared<Buffer>(this->device_, page_size, page_size, BufferType::DRAM);
            EnqueueWriteBuffer(this->device_->command_queue(), buf, page, false);
        }
        Finish(this->device_->command_queue());

        std::chrono::duration<double> elapsed_seconds = (std::chrono::system_clock::now() - start);
        tt::log_info(tt::LogTest, "Test with CQ Mode: {} Finished in {:.2f} us", mode, elapsed_seconds.count() * 1000 * 1000);

        // Read completion queue and ensure we see events 0-99 inclusive in order
        uint32_t event;
        for (size_t i = 0; i < num_buffers; i++) {
            uint32_t host_addr = completion_queue_base + i * completion_queue_event_alignment;
            tt::Cluster::instance().read_sysmem(&event, 4, host_addr, mmio_device_id, channel);
            EXPECT_EQ(event, expected_event_id++);
        }

    }
    this->device_->command_queue().set_mode(current_mode);


}

// Basic test, record events, check that Event struct was updated. Enough commands to trigger issue queue wrap.
TEST_F(CommandQueueFixture, TestEventsEnqueueRecordEventIssueQueueWrap) {

    const size_t num_events = 100000; // Enough to wrap issue queue. 768MB and cmds are 22KB each, so 35k cmds.
    uint32_t cmds_issued_per_cq = 0;

    auto current_mode = CommandQueue::default_mode();
    for (const CommandQueue::CommandQueueMode mode : {CommandQueue::CommandQueueMode::PASSTHROUGH, CommandQueue::CommandQueueMode::ASYNC}) {
        tt::log_info(tt::LogTest, "Using CQ Mode: {}", mode);
        this->device_->command_queue().set_mode(mode);
        auto start = std::chrono::system_clock::now();

        for (size_t i = 0; i < num_events; i++) {
            auto event = std::make_shared<Event>(); // type is std::shared_ptr<Event>
            EnqueueRecordEvent(this->device_->command_queue(), event);

            if (mode == CommandQueue::CommandQueueMode::ASYNC) {
                event->wait_until_ready(); // To check Event fields from host, must block until async cq populated event.
            }
            EXPECT_EQ(event->event_id, cmds_issued_per_cq);
            EXPECT_EQ(event->cq_id, this->device_->command_queue().id());
            cmds_issued_per_cq++;
        }
        Finish(this->device_->command_queue());

        std::chrono::duration<double> elapsed_seconds = (std::chrono::system_clock::now() - start);
        tt::log_info(tt::LogTest, "Test with CQ Mode: {} Finished in {:.2f} us", mode, elapsed_seconds.count() * 1000 * 1000);
    }
    this->device_->command_queue().set_mode(current_mode);
}

// Test where Host synchronously waits for event to be completed.
TEST_F(CommandQueueFixture, TestEventsEnqueueRecordEventAndSynchronize) {
    const size_t num_events = 100;
    const size_t num_events_between_sync = 10;

    auto current_mode = CommandQueue::default_mode();
    for (const CommandQueue::CommandQueueMode mode : {CommandQueue::CommandQueueMode::PASSTHROUGH, CommandQueue::CommandQueueMode::ASYNC}) {
        tt::log_info(tt::LogTest, "Using CQ Mode: {}", mode);
        auto start = std::chrono::system_clock::now();
        this->device_->command_queue().set_mode(mode);

        std::vector<std::shared_ptr<Event>> sync_events;

        // A bunch of events recorded, occasionally will sync from host.
        for (size_t i = 0; i < num_events; i++) {
            auto event = sync_events.emplace_back(std::make_shared<Event>());
            EnqueueRecordEvent(this->device_->command_queue(), event);

            // Host synchronize every N number of events.
            if (i > 0 && ((i % num_events_between_sync) == 0)) {
                EventSynchronize(event);
            }
        }

        // A bunch of bonus syncs where event_id is mod on earlier ID's.
        EventSynchronize(sync_events.at(2));
        EventSynchronize(sync_events.at(sync_events.size() - 2));
        EventSynchronize(sync_events.at(5));

        Finish(this->device_->command_queue());

        std::chrono::duration<double> elapsed_seconds = (std::chrono::system_clock::now() - start);
        tt::log_info(tt::LogTest, "Test with CQ Mode: {} Finished in {:.2f} us", mode, elapsed_seconds.count() * 1000 * 1000);
    }
    this->device_->command_queue().set_mode(current_mode);
}
// Negative test. Host syncing on a future event that isn't actually issued.
// Ensure that expected hang is seen, which indicates event sync feature is working properly.
TEST_F(CommandQueueFixture, TestEventsEnqueueRecordEventAndSynchronizeHang) {
    tt::llrt::OptionsG.set_test_mode_enabled(true); // Required for finish hang breakout.

    auto future_event = std::make_shared<Event>();
    EnqueueRecordEvent(this->device_->command_queue(), future_event);
    future_event->wait_until_ready();   // in case async used, must block until async cq populated event.
    future_event->event_id = 0xFFFF;    // Modify event_id to be a future event that isn't issued yet.

    // Launch Host Sync in an async thread, expected to hang, with timeout and kill signal.
    auto future = std::async(std::launch::async, [this, future_event]() {
        return EventSynchronize(future_event);
    });

    bool seen_expected_hang = future.wait_for(std::chrono::seconds(1)) == std::future_status::timeout;
    tt::watcher_server_set_error_flag(seen_expected_hang); // Signal to terminate thread. Don't care about it's exception.

    // Briefly wait before clearing error flag, and wrapping up via finish.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    tt::watcher_server_set_error_flag(false);
    Finish(this->device_->command_queue());

    log_info(tt::LogTest, "Note: Test expects to see a hang if events feature is working. seen_expected_hang: {}", seen_expected_hang);
    EXPECT_TRUE(seen_expected_hang);
}

// Negative test. Device sync. Single CQ here syncing on a future event that isn't actually issued.
// Ensure that expected hang is seen, which indicates event sync feature is working properly.
TEST_F(CommandQueueFixture, TestEventsQueueWaitForEventHang) {
    tt::llrt::OptionsG.set_test_mode_enabled(true); // Required for finish hang breakout.

    auto future_event = std::make_shared<Event>();
    EnqueueRecordEvent(this->device_->command_queue(), future_event);
    future_event->wait_until_ready();   // in case async used, must block until async cq populated event.
    future_event->event_id = 0xFFFF;    // Modify event_id to be a future event that isn't issued yet.
    EnqueueWaitForEvent(this->device_->command_queue(), future_event);

    // Launch Finish in an async thread, expected to hang, with timeout and kill signal.
    auto future = std::async(std::launch::async, [this]() {
        return Finish(this->device_->command_queue());
    });

    bool seen_expected_hang = future.wait_for(std::chrono::seconds(1)) == std::future_status::timeout;
    tt::watcher_server_set_error_flag(seen_expected_hang); // Signal to terminate thread. Don't care about it's exception.

    // Clear error flag before exiting to restore state for next test.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    tt::watcher_server_set_error_flag(false);

    log_info(tt::LogTest, "Note: Test expects to see a hang if events feature is working. seen_expected_hang: {}", seen_expected_hang);
    EXPECT_TRUE(seen_expected_hang);
}

// Readback L1 written to be CQ record_last_completed_event() / wait_for_event() functions for extra confidence they are working.
// Alternatively, could maybe do something like check that completion queue last event ID matches that which was recorded.
void verify_event_completed_and_synced_vals_from_l1(Device* device, uint32_t expected_last_completed, uint32_t expected_last_synced){
    std::vector<uint32_t> last_completed_event_id;
    std::vector<uint32_t> last_synced_event_id;

    auto &completion_q_wr_core = device->command_queue().hw_command_queue().completion_queue_writer_core;
    tt::tt_metal::detail::ReadFromDeviceL1(device, completion_q_wr_core, CQ_COMPLETION_LAST_EVENT, sizeof(uint32_t), last_completed_event_id);
    tt::tt_metal::detail::ReadFromDeviceL1(device, completion_q_wr_core, CQ_COMPLETION_16B_SCRATCH, sizeof(uint32_t), last_synced_event_id);

    log_debug(tt::LogTest, "From Core: {} Got last_completed_event_id: {} (expected: {}) last_synced_event_id: {} (expected: {})",
        completion_q_wr_core.str(), last_completed_event_id.at(0), expected_last_completed, last_synced_event_id.at(0), expected_last_synced);

    EXPECT_EQ(last_completed_event_id.at(0), expected_last_completed);
    EXPECT_EQ(last_synced_event_id.at(0), expected_last_synced);
}

// Device sync. Single CQ here, less interesting than 2CQ but still useful. Ensure no hangs, and verify each CQ recorded
// expected "last completed event id" and "last synced event id" in their L1 (used by WaitForEvent cmd handling).
TEST_F(CommandQueueFixture, TestEventsQueueWaitForEventBasic) {

    const size_t num_events = 50;
    const size_t num_events_between_sync = 5;
    uint32_t num_events_generated = 0;
    uint32_t last_sync_event_idx = 0;

    auto current_mode = CommandQueue::default_mode();
    for (const CommandQueue::CommandQueueMode mode : {CommandQueue::CommandQueueMode::PASSTHROUGH, CommandQueue::CommandQueueMode::ASYNC}) {
        tt::log_info(tt::LogTest, "Using CQ Mode: {}", mode);
        auto start = std::chrono::system_clock::now();
        this->device_->command_queue().set_mode(mode);
        std::vector<std::shared_ptr<Event>> sync_events;

        // A bunch of events recorded, occasionally will sync from device.
        for (size_t i = 0; i < num_events; i++) {
            auto event = sync_events.emplace_back(std::make_shared<Event>());
            EnqueueRecordEvent(this->device_->command_queue(), event);
            num_events_generated++;

            // Device synchronize every N number of events.
            if (i > 0 && ((i % num_events_between_sync) == 0)) {
                log_debug(tt::LogTest, "Going to WaitForEvent for i: {}", i);
                EnqueueWaitForEvent(this->device_->command_queue(), event);
                last_sync_event_idx = sync_events.size() - 1;
                num_events_generated++;
            }
        }

        Finish(this->device_->command_queue());

        // Safe to look at event ID now for async, since complete.
        verify_event_completed_and_synced_vals_from_l1(this->device_, num_events_generated-1, sync_events.at(last_sync_event_idx)->event_id);

        // A bunch of bonus syncs where event_id is mod on earlier ID's.
        EnqueueWaitForEvent(this->device_->command_queue(), sync_events.at(0));
        EnqueueWaitForEvent(this->device_->command_queue(), sync_events.at(sync_events.size() - 5));
        EnqueueWaitForEvent(this->device_->command_queue(), sync_events.at(4));
        num_events_generated += 3;
        Finish(this->device_->command_queue());

        // The last event seen to be completed by wait_for_event would be 2 events ago.
        verify_event_completed_and_synced_vals_from_l1(this->device_, num_events_generated-1, num_events_generated-2);

        std::chrono::duration<double> elapsed_seconds = (std::chrono::system_clock::now() - start);
        tt::log_info(tt::LogTest, "Test with CQ Mode: {} Finished in {:.2f} us", mode, elapsed_seconds.count() * 1000 * 1000);
    }
    this->device_->command_queue().set_mode(current_mode);
}

// Mix of WritesBuffers, RecordEvent, WaitForEvent, EventSynchronize with some checking.
TEST_F(CommandQueueFixture, TestEventsMixedWriteBufferRecordWaitSynchronize) {
    const size_t num_buffers = 100;
    const uint32_t page_size = 2048;
    vector<uint32_t> page(page_size / sizeof(uint32_t));
    uint32_t cmds_issued_per_cq = 0;
    const uint32_t num_cmds_per_cq = 3; // Record, Write, Wait
    uint32_t expected_event_id = 0;

    auto current_mode = CommandQueue::default_mode();
    for (const CommandQueue::CommandQueueMode mode : {CommandQueue::CommandQueueMode::PASSTHROUGH, CommandQueue::CommandQueueMode::ASYNC}) {
        tt::log_info(tt::LogTest, "Using CQ Mode: {}", mode);
        auto start = std::chrono::system_clock::now();
        this->device_->command_queue().set_mode(mode);

        uint32_t completion_queue_base = this->device_->sysmem_manager().get_completion_queue_read_ptr(0);
        chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(this->device_->id());
        uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->device_->id());
        constexpr uint32_t completion_queue_event_alignment = 32;
        for (size_t i = 0; i < num_buffers; i++) {

            log_debug(tt::LogTest, "Mode: {} i: {} - Going to record event, write, wait, synchronize.", mode, i);
            auto event = std::make_shared<Event>(); // type is std::shared_ptr<Event>
            EnqueueRecordEvent(this->device_->command_queue(), event);

            // Cannot count on event being populated with async cq, so only check with passthrough.
            if (mode == CommandQueue::CommandQueueMode::PASSTHROUGH) {
                EXPECT_EQ(event->cq_id, this->cmd_queue->id());
                EXPECT_EQ(event->event_id, cmds_issued_per_cq);
            }

            std::shared_ptr<Buffer> buf = std::make_shared<Buffer>(this->device_, page_size, page_size, BufferType::DRAM);
            EnqueueWriteBuffer(this->device_->command_queue(), buf, page, false);
            EnqueueWaitForEvent(this->device_->command_queue(), event);

            if (i % 10 == 0) {
                EventSynchronize(event);
                // For async, can verify event fields here since previous function already called wait-until-ready.
                if (mode == CommandQueue::CommandQueueMode::ASYNC) {
                    EXPECT_EQ(event->cq_id, this->cmd_queue->id());
                    EXPECT_EQ(event->event_id, cmds_issued_per_cq);
                }
            }
            cmds_issued_per_cq += num_cmds_per_cq;
        }
        Finish(this->device_->command_queue());

        // Read completion queue and ensure we see expected event IDs
        uint32_t event_id;
        for (size_t i = 0; i < num_buffers * num_cmds_per_cq; i++) {
            uint32_t host_addr = completion_queue_base + i * completion_queue_event_alignment;
            tt::Cluster::instance().read_sysmem(&event_id, 4, host_addr, mmio_device_id, channel);
            EXPECT_EQ(event_id, expected_event_id++);
        }

        std::chrono::duration<double> elapsed_seconds = (std::chrono::system_clock::now() - start);
        tt::log_info(tt::LogTest, "Test with CQ Mode: {} Finished in {:.2f} us", mode, elapsed_seconds.count() * 1000 * 1000);
    }
    this->device_->command_queue().set_mode(current_mode);
}
