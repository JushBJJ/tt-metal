// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device/tt_xy_pair.h"
#include "gtest/gtest.h"
#include "tt_dnn/op_library/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "tt_eager/tt_dnn/op_library/ccl/ccl_common.hpp"
#include "tt_eager/tt_dnn/op_library/ccl/ccl_host_datastructures.hpp"

TEST(CclHelpers, CreateEriscDatamoverBuilder_Chan4_PageSize2048_RRBufferSharingMode) {
    std::size_t num_channels = 4;
    uint32_t page_size = 2048;
    ccl::EriscDataMoverBufferSharingMode buffer_sharing_mode = ccl::EriscDataMoverBufferSharingMode::ROUND_ROBIN;
    ccl::EriscDataMoverTerminationMode termination_mode = ccl::EriscDataMoverTerminationMode::MESSAGE_COUNT_REACHED;

    auto edm_builder = create_erisc_datamover_builder(num_channels, page_size, buffer_sharing_mode, termination_mode);
    std::vector<uint32_t> worker_semaphore_addresses = {
        0x1000,
        0x1010,
        0x1020,
        0x1030,
    };
    std::vector<uint32_t> message_counts = {256, 512, 24, 1};
    std::vector<std::vector<ccl::WorkerXY>> const& worker_coords = {
        {ccl::WorkerXY{1, 1}, ccl::WorkerXY{2, 1}},
        {ccl::WorkerXY{3, 1}},
        {ccl::WorkerXY{4, 1}, ccl::WorkerXY{5, 1}, ccl::WorkerXY{6, 1}},
        {ccl::WorkerXY{1, 2}},
    };
    std::vector<bool> is_sender_channel{true, false, true, false};

    std::vector<ccl::EriscDatamoverBuilder::ChannelBufferInterface> channel_buffer_interfaces;
    channel_buffer_interfaces.reserve(num_channels);
    for (std::size_t i = 0; i < num_channels; i++) {
        ccl::EriscDatamoverBuilder::ChannelBufferInterface const& channel_buffer_interface =
            (is_sender_channel[i])
                ? edm_builder.add_sender_channel(worker_semaphore_addresses[i], message_counts[i], worker_coords[i])
                : edm_builder.add_receiver_channel(worker_semaphore_addresses[i], message_counts[i], worker_coords[i]);
        channel_buffer_interfaces.push_back(channel_buffer_interface);
        ASSERT_TRUE(channel_buffer_interface.eth_buffer_l1_address > 0);
        ASSERT_TRUE(channel_buffer_interface.eth_semaphore_l1_address > 0);
    }

    auto const& active_channels = edm_builder.get_active_channels();
    ASSERT_EQ(active_channels.size(), num_channels);
    for (std::size_t i = 0; i < active_channels.size(); ++i) {
        ASSERT_EQ(active_channels[i].channel, i);
        ASSERT_EQ(active_channels[i].is_sender, is_sender_channel.at(i));
        ASSERT_EQ(active_channels[i].worker_coords, worker_coords.at(i));
        ASSERT_TRUE(active_channels[i].worker_semaphore_address == worker_semaphore_addresses.at(i));
        ASSERT_TRUE(active_channels[i].num_eth_messages_to_forward == message_counts.at(i));
    }
}

TEST(CclHelpers, EriscDatamoverConfig_GetEdmHandshakeAddress_GT_0) {
    for (std::size_t i = 0; i < 8; i++) {
        ASSERT_TRUE(ccl::EriscDatamoverConfig::get_edm_handshake_address() > 0);
    }
}
TEST(CclHelpers, EriscDatamoverConfig_GetSemaphoresBaseAddress_GT_0) {
    for (std::size_t i = 0; i < 8; i++) {
        ASSERT_TRUE(
            ccl::EriscDatamoverConfig::get_semaphores_base_address(i) >=
            (ccl::EriscDatamoverConfig::get_edm_handshake_address() +
             ccl::EriscDatamoverConfig::handshake_location_size +
             ccl::EriscDatamoverConfig::edm_receiver_first_level_ack_source_word_size));
    }
}

TEST(CclHelpers, EriscDatamoverConfig_GetBuffersBaseAddress_GT_0) {
    for (std::size_t i = 0; i < 8; i++) {
        ASSERT_TRUE(
            ccl::EriscDatamoverConfig::get_buffers_base_address(i) >=
            (ccl::EriscDatamoverConfig::get_edm_handshake_address() +
             ccl::EriscDatamoverConfig::handshake_location_size +
             ccl::EriscDatamoverConfig::edm_receiver_first_level_ack_source_word_size));
    }
}

TEST(CclHelpers, EriscDatamoverConfig_ComputeBufferSize_GT_0) {
    for (std::size_t i = 0; i < 8; i++) {
        ASSERT_TRUE(
            ccl::EriscDatamoverConfig::get_buffers_base_address(i) >=
            (ccl::EriscDatamoverConfig::get_edm_handshake_address() +
             ccl::EriscDatamoverConfig::handshake_location_size +
             ccl::EriscDatamoverConfig::edm_receiver_first_level_ack_source_word_size));
    }
}

/////////////////////////////////////////
// TEST AdvanceSliceRowMajor
/////////////////////////////////////////
//                                               x_y             x_y             x_y
TEST(CclHelper_AdvanceSliceRowMajor, InnerOffset_0_0__InnerShape_1_1__OuterShape_2_2__NumActiveSlices_1) {
    const auto expected = tt::tt_metal::ccl::coord_t(1, 0);
    auto const& result = tt::tt_metal::ccl::advance_slice_row_major({0, 0}, {1, 1}, {2, 2}, 1);
    ASSERT_EQ(result.x, expected.x);
    ASSERT_EQ(result.y, expected.y);
}
TEST(CclHelper_AdvanceSliceRowMajor, InnerOffset_1_0__InnerShape_1_1__OuterShape_2_2__NumActiveSlices_1) {
    const auto expected = tt::tt_metal::ccl::coord_t(0, 1);
    auto const& result = tt::tt_metal::ccl::advance_slice_row_major({1, 0}, {1, 1}, {2, 2}, 1);
    ASSERT_EQ(result.x, expected.x);
    ASSERT_EQ(result.y, expected.y);
}
TEST(CclHelper_AdvanceSliceRowMajor, InnerOffset_0_1__InnerShape_1_1__OuterShape_2_2__NumActiveSlices_1) {
    const auto expected = tt::tt_metal::ccl::coord_t(1, 1);
    auto const& result = tt::tt_metal::ccl::advance_slice_row_major({0, 1}, {1, 1}, {2, 2}, 1);
    ASSERT_EQ(result.x, expected.x);
    ASSERT_EQ(result.y, expected.y);
}
TEST(CclHelper_AdvanceSliceRowMajor, InnerOffset_0_0__InnerShape_1_1__OuterShape_2_2__NumActiveSlices_2) {
    const auto expected = tt::tt_metal::ccl::coord_t(0, 1);
    auto const& result = tt::tt_metal::ccl::advance_slice_row_major({0, 0}, {1, 1}, {2, 2}, 2);
    ASSERT_EQ(result.x, expected.x);
    ASSERT_EQ(result.y, expected.y);
}
TEST(CclHelper_AdvanceSliceRowMajor, InnerOffset_1_0__InnerShape_1_1__OuterShape_2_2__NumActiveSlices_2) {
    const auto expected = tt::tt_metal::ccl::coord_t(1, 1);
    auto const& result = tt::tt_metal::ccl::advance_slice_row_major({1, 0}, {1, 1}, {2, 2}, 2);
    ASSERT_EQ(result.x, expected.x);
    ASSERT_EQ(result.y, expected.y);
}

// Test that we successfully go out of bounds on the last iteration
TEST(CclHelper_AdvanceSliceRowMajor, InnerOffset_0_1__InnerShape_1_1__OuterShape_2_2__NumActiveSlices_2) {
    auto const& result = tt::tt_metal::ccl::advance_slice_row_major({0, 1}, {1, 1}, {2, 2}, 2);
    ASSERT_TRUE(result.x >= 2 || result.y >= 2);
}
TEST(CclHelper_AdvanceSliceRowMajor, InnerOffset_1_1__InnerShape_1_1__OuterShape_2_2__NumActiveSlices_2) {
    auto const& result = tt::tt_metal::ccl::advance_slice_row_major({1, 1}, {1, 1}, {2, 2}, 2);
    ASSERT_TRUE(result.x >= 2 || result.y >= 2);
}

TEST(CclHelper_AdvanceSliceRowMajor, InnerOffset_0_0__InnerShape_1_1__OuterShape_2_2__NumActiveSlices_3) {
    const auto expected = tt::tt_metal::ccl::coord_t(1, 1);
    auto const& result = tt::tt_metal::ccl::advance_slice_row_major({0, 0}, {1, 1}, {2, 2}, 3);
    ASSERT_EQ(result.x, expected.x);
    ASSERT_EQ(result.y, expected.y);
}
TEST(CclHelper_AdvanceSliceRowMajor, InnerOffset_1_1__InnerShape_1_1__OuterShape_2_2__NumActiveSlices_3) {
    const auto expected = tt::tt_metal::ccl::coord_t(1, 1);
    const auto outer_shape = tt::tt_metal::ccl::coord_t(2, 2);
    const auto inner_offset = tt::tt_metal::ccl::coord_t(1, 1);
    const auto inner_shape = tt::tt_metal::ccl::coord_t(1, 1);
    const uint32_t num_parallel_workers = 3;
    auto const& result =
        tt::tt_metal::ccl::advance_slice_row_major(inner_offset, inner_shape, outer_shape, num_parallel_workers);
    ASSERT_TRUE(result.x >= outer_shape.x || result.y >= outer_shape.y);
}
TEST(CclHelper_AdvanceSliceRowMajor, InnerOffset_24_0__InnerShape_24_0__OuterShape_32_4__NumActiveSlices_4) {
    const auto expected = tt::tt_metal::ccl::coord_t(24, 2);
    const auto outer_shape = tt::tt_metal::ccl::coord_t(32, 4);
    const auto inner_offset = tt::tt_metal::ccl::coord_t(24, 0);
    const auto inner_shape = tt::tt_metal::ccl::coord_t(24, 1);
    const uint32_t num_parallel_workers = 4;
    auto const& result =
        tt::tt_metal::ccl::advance_slice_row_major(inner_offset, inner_shape, outer_shape, num_parallel_workers);
    ASSERT_EQ(result.x, expected.x);
    ASSERT_EQ(result.y, expected.y);
}

/////////////////////////////////////////
// Test InterleavedRingReduceScatterTensorSlicer
/////////////////////////////////////////
TEST(Ccl_InterleavedRingReduceScatterTensorSlicer, ComputeWorkerSliceOffsets_AllWorkersSameRow) {
    auto worker_slice_shapes = std::vector<tt_xy_pair>(4, {2, 2});
    tt_xy_pair tensor_slice_shape = {8, 4};
    auto const& worker_slice_offsets = ccl::InterleavedRingReduceScatterTensorSlicer::compute_worker_slice_offsets(
        worker_slice_shapes, tensor_slice_shape);
    ASSERT_EQ(worker_slice_offsets.at(0), tt_xy_pair(0, 0));
    ASSERT_EQ(worker_slice_offsets.at(1), tt_xy_pair(2, 0));
    ASSERT_EQ(worker_slice_offsets.at(2), tt_xy_pair(4, 0));
    ASSERT_EQ(worker_slice_offsets.at(3), tt_xy_pair(6, 0));
}
TEST(Ccl_InterleavedRingReduceScatterTensorSlicer, ComputeWorkerSliceOffsets_1WorkerWrapToNextRowAligned) {
    auto worker_slice_shapes = std::vector<tt_xy_pair>(4, {2, 2});
    tt_xy_pair tensor_slice_shape = {6, 4};
    auto const& worker_slice_offsets = ccl::InterleavedRingReduceScatterTensorSlicer::compute_worker_slice_offsets(
        worker_slice_shapes, tensor_slice_shape);
    ASSERT_EQ(worker_slice_offsets.at(0), tt_xy_pair(0, 0));
    ASSERT_EQ(worker_slice_offsets.at(1), tt_xy_pair(2, 0));
    ASSERT_EQ(worker_slice_offsets.at(2), tt_xy_pair(4, 0));
    ASSERT_EQ(worker_slice_offsets.at(3), tt_xy_pair(0, 2));
}
TEST(Ccl_InterleavedRingReduceScatterTensorSlicer, ComputeWorkerSliceOffsets_1WorkerWrapToNextRowMisaligned) {
    {
        auto worker_slice_shapes = std::vector<tt_xy_pair>(4, {2, 2});
        tt_xy_pair tensor_slice_shape = {5, 4};
        auto const& worker_slice_offsets = ccl::InterleavedRingReduceScatterTensorSlicer::compute_worker_slice_offsets(
            worker_slice_shapes, tensor_slice_shape);
        ASSERT_EQ(worker_slice_offsets.at(0), tt_xy_pair(0, 0));
        ASSERT_EQ(worker_slice_offsets.at(1), tt_xy_pair(2, 0));
        ASSERT_EQ(worker_slice_offsets.at(2), tt_xy_pair(4, 0));
        ASSERT_EQ(worker_slice_offsets.at(3), tt_xy_pair(0, 2));
    }
}
TEST(Ccl_InterleavedRingReduceScatterTensorSlicer, ComputeWorkerSliceOffsets_MultipleWorkersWrapToNextRowAligned) {
    auto worker_slice_shapes = std::vector<tt_xy_pair>(8, {2, 2});
    tt_xy_pair tensor_slice_shape = {10, 4};
    auto const& worker_slice_offsets = ccl::InterleavedRingReduceScatterTensorSlicer::compute_worker_slice_offsets(
        worker_slice_shapes, tensor_slice_shape);
    ASSERT_EQ(worker_slice_offsets.at(0), tt_xy_pair(0, 0));
    ASSERT_EQ(worker_slice_offsets.at(1), tt_xy_pair(2, 0));
    ASSERT_EQ(worker_slice_offsets.at(2), tt_xy_pair(4, 0));
    ASSERT_EQ(worker_slice_offsets.at(3), tt_xy_pair(6, 0));
    ASSERT_EQ(worker_slice_offsets.at(4), tt_xy_pair(8, 0));
    ASSERT_EQ(worker_slice_offsets.at(5), tt_xy_pair(0, 2));
    ASSERT_EQ(worker_slice_offsets.at(6), tt_xy_pair(2, 2));
    ASSERT_EQ(worker_slice_offsets.at(7), tt_xy_pair(4, 2));
}

TEST(Ccl_InterleavedRingReduceScatterTensorSlicer, ComputeWorkerSliceOffsets_MultipleWorkersWrapToNextRowMisaligned) {
    auto worker_slice_shapes = std::vector<tt_xy_pair>(8, {2, 2});
    tt_xy_pair tensor_slice_shape = {9, 4};
    auto const& worker_slice_offsets = ccl::InterleavedRingReduceScatterTensorSlicer::compute_worker_slice_offsets(
        worker_slice_shapes, tensor_slice_shape);
    ASSERT_EQ(worker_slice_offsets.at(0), tt_xy_pair(0, 0));
    ASSERT_EQ(worker_slice_offsets.at(1), tt_xy_pair(2, 0));
    ASSERT_EQ(worker_slice_offsets.at(2), tt_xy_pair(4, 0));
    ASSERT_EQ(worker_slice_offsets.at(3), tt_xy_pair(6, 0));
    ASSERT_EQ(worker_slice_offsets.at(4), tt_xy_pair(8, 0));
    ASSERT_EQ(worker_slice_offsets.at(5), tt_xy_pair(0, 2));
    ASSERT_EQ(worker_slice_offsets.at(6), tt_xy_pair(2, 2));
    ASSERT_EQ(worker_slice_offsets.at(7), tt_xy_pair(4, 2));
}

TEST(Ccl_InterleavedRingReduceScatterTensorSlicer, ComputeWorkerSliceOffsets_NMinus1WorkersWrapToNextRowAligned) {
    auto worker_slice_shapes = std::vector<tt_xy_pair>(3, {4, 4});
    tt_xy_pair tensor_slice_shape = {4, 12};
    auto const& worker_slice_offsets = ccl::InterleavedRingReduceScatterTensorSlicer::compute_worker_slice_offsets(
        worker_slice_shapes, tensor_slice_shape);
    ASSERT_EQ(worker_slice_offsets.at(0), tt_xy_pair(0, 0));
    ASSERT_EQ(worker_slice_offsets.at(1), tt_xy_pair(0, 4));
    ASSERT_EQ(worker_slice_offsets.at(2), tt_xy_pair(0, 8));
}

TEST(Ccl_InterleavedRingReduceScatterTensorSlicer, ComputeWorkerSliceOffsets_NMinus1WorkersWrapToNextRowMisaligned) {
    auto worker_slice_shapes = std::vector<tt_xy_pair>(3, {4, 3});
    tt_xy_pair tensor_slice_shape = {3, 12};
    auto const& worker_slice_offsets = ccl::InterleavedRingReduceScatterTensorSlicer::compute_worker_slice_offsets(
        worker_slice_shapes, tensor_slice_shape);
    ASSERT_EQ(worker_slice_offsets.at(0), tt_xy_pair(0, 0));
    ASSERT_EQ(worker_slice_offsets.at(1), tt_xy_pair(0, 3));
    ASSERT_EQ(worker_slice_offsets.at(2), tt_xy_pair(0, 6));
}

TEST(
    Ccl_InterleavedTensorWorkerSlice_ComputeNumWorkerSliceIterations,
    InnerOffset_0_0__InnerShape_24_1__OuterShape_32_4__NumActiveSlices_4) {
    auto worker_slice = ccl::InterleavedTensorWorkerSlice(
        tt_xy_pair(99999, 99999),  // tensor shape shouldn't affect the result
        tt_xy_pair(32, 4),
        tt_xy_pair(24, 1),
        tt_xy_pair(0, 0));
    uint32_t num_workers = 4;
    auto num_iterations = worker_slice.compute_num_worker_slice_iterations(num_workers);
    auto expected = 2;
    ASSERT_EQ(num_iterations, expected);
}

TEST(
    Ccl_InterleavedTensorWorkerSlice_ComputeNumWorkerSliceIterations,
    InnerOffset_24_0__InnerShape_24_1__OuterShape_32_4__NumActiveSlices_4) {
    auto worker_slice = ccl::InterleavedTensorWorkerSlice(
        tt_xy_pair(99999, 99999),  // tensor shape shouldn't affect the result
        tt_xy_pair(32, 4),
        tt_xy_pair(24, 1),
        tt_xy_pair(24, 0));
    uint32_t num_workers = 4;
    auto num_iterations = worker_slice.compute_num_worker_slice_iterations(num_workers);
    auto expected = 2;
    ASSERT_EQ(num_iterations, expected);
}

TEST(
    Ccl_InterleavedTensorWorkerSlice_ComputeNumWorkerSliceIterations,
    InnerOffset_0_1__InnerShape_24_1__OuterShape_32_4__NumActiveSlices_4) {
    auto worker_slice = ccl::InterleavedTensorWorkerSlice(
        tt_xy_pair(99999, 99999),  // tensor shape shouldn't affect the result
        tt_xy_pair(32, 4),
        tt_xy_pair(24, 1),
        tt_xy_pair(0, 1));
    uint32_t num_workers = 4;
    auto num_iterations = worker_slice.compute_num_worker_slice_iterations(num_workers);
    auto expected = 2;
    ASSERT_EQ(num_iterations, expected);
}

TEST(
    Ccl_InterleavedTensorWorkerSlice_ComputeNumWorkerSliceIterations,
    InnerOffset_24_1__InnerShape_24_1__OuterShape_32_4__NumActiveSlices_4) {
    auto worker_slice = ccl::InterleavedTensorWorkerSlice(
        tt_xy_pair(99999, 99999),  // tensor shape shouldn't affect the result
        tt_xy_pair(32, 4),
        tt_xy_pair(24, 1),
        tt_xy_pair(24, 0));
    uint32_t num_workers = 4;
    auto num_iterations = worker_slice.compute_num_worker_slice_iterations(num_workers);
    auto expected = 2;
    ASSERT_EQ(num_iterations, expected);
}
