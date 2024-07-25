// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/deprecated/tt_dnn/op_library/cb_utils.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/math.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/work_split.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt;
using namespace tt::constants;

namespace ttnn::operations::embedding_backward::detail {

operation::ProgramWithCallbacks embedding_backward_multi_core(const Tensor &index_tensor, const Tensor &grad_tensor, Tensor &output) {
    ////////////////////////////////////////////////////////////////////////////
    //                 Buffer Setup
    ////////////////////////////////////////////////////////////////////////////

    tt_metal::Buffer *index_tensor_buffer = index_tensor.buffer();
    tt_metal::Buffer *grad_tensor_buffer = grad_tensor.buffer();
    tt_metal::Buffer *out_buffer = output.buffer();

    Device *device = grad_tensor.device();
    auto dst_addr = out_buffer->address();

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////

    Program program{};

    bool grad_is_dram = grad_tensor_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool index_is_dram = index_tensor_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool out_is_dram = out_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;

    uint32_t grad_element_size_bytes = grad_tensor.element_size();
    uint32_t index_element_size_bytes = index_tensor.element_size();

    tt::DataFormat grad_cb_data_format = datatype_to_dataformat_converter(grad_tensor.get_dtype());
    uint32_t grad_single_tile_size = tt::tt_metal::detail::TileSize(grad_cb_data_format);

    tt::DataFormat index_cb_data_format = datatype_to_dataformat_converter(index_tensor.get_dtype());
    uint32_t index_single_tile_size = tt::tt_metal::detail::TileSize(index_cb_data_format);
    uint32_t index_page_size = index_tensor.get_legacy_shape()[-1] * index_element_size_bytes;

    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.get_dtype());
    uint32_t output_single_tile_size = tt::tt_metal::detail::TileSize(output_cb_data_format);

    uint32_t embedding_dim = grad_tensor.get_legacy_shape()[-1];
    uint32_t embedding_tiles = embedding_dim / TILE_WIDTH;

    uint32_t batch_size = index_tensor.get_legacy_shape()[0];
    uint32_t seq_len = index_tensor.get_legacy_shape()[-1];

    // We split work based on the number of tiles in the embedding dimension
    auto grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(grid_size, embedding_tiles);
    uint32_t max_tiles_per_core = std::max(num_tiles_per_core_group_1, num_tiles_per_core_group_2);

    log_info(LogType::LogOp, "Embedding tiles: {}", embedding_tiles);
    log_info(LogType::LogOp, "Num parallel cores: {}", num_cores);
    log_info(LogType::LogOp, "Max tiles per core: {}", max_tiles_per_core);

    ////////////////////////////////////////////////////////////////////////////
    //                 Circular buffers
    ////////////////////////////////////////////////////////////////////////////

    // To read from grad tensor
    create_cb(CB::c_in0, program, all_cores, grad_single_tile_size, max_tiles_per_core, grad_cb_data_format);

    // To store index values for a single tile
    create_cb(CB::c_in1, program, all_cores, index_single_tile_size, 1, index_cb_data_format);

    // To read from output tensor
    create_cb(CB::c_in2, program, all_cores, output_single_tile_size, max_tiles_per_core, grad_cb_data_format);

    // To store mask values for a single tile
    create_cb(CB::c_intermed0, program, all_cores, index_single_tile_size, 1, index_cb_data_format);

    // To store reshuffled rows of an input tile to be added to the output tile
    create_cb(CB::c_intermed1, program, all_cores, grad_single_tile_size, 1, grad_cb_data_format);

    // For tiles to be written to the output
    create_cb(CB::c_out0, program, all_cores, output_single_tile_size, max_tiles_per_core, output_cb_data_format);

    ////////////////////////////////////////////////////////////////////////////
    //                 Kernels
    ////////////////////////////////////////////////////////////////////////////

    // reader

    bool index_stick_size_is_power_of_two = is_power_of_two_at_least_32(index_page_size);
    uint32_t index_log2_stick_size = index_stick_size_is_power_of_two ? log2(index_page_size) : 0;

    std::vector<uint32_t> reader_compile_time_args = {
        grad_is_dram,
        index_is_dram,
        out_is_dram,
        index_page_size,
        index_stick_size_is_power_of_two,
        index_log2_stick_size,
        index_tensor.get_dtype() == DataType::BFLOAT16};

    auto reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/embedding_backward/device/kernels/dataflow/reader_embedding_backward.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    std::vector<uint32_t> reader_runtime_args = {
        grad_tensor_buffer->address(),
        index_tensor_buffer->address(),
        out_buffer->address(),
        seq_len / TILE_WIDTH,
        batch_size,
        embedding_tiles,  // how many pages to skip to get to the next row
        0,                // offset to the first tile in a row
        0,                // how many tiles to process in a row
    };

    ////////////////////////////////////////////////////////////////////////////
    //                 Run-time arguments
    ////////////////////////////////////////////////////////////////////////////

    auto cores = corerange_to_cores(all_cores);
    uint32_t offset = 0;
    for (auto core : cores) {
        reader_runtime_args[6] = offset;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            reader_runtime_args[7] = num_tiles_per_core_group_1;
        } else {
            reader_runtime_args[7] = num_tiles_per_core_group_2;
        }
        SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);

        offset += reader_runtime_args[7];
    }

    auto override_runtime_args_callback = [reader_kernel_id, cores, device](
                                              const Program &program,
                                              const std::vector<Buffer *> &input_buffers,
                                              const std::vector<Buffer *> &output_buffers) {
        auto grad_dram_buffer = input_buffers.at(0);
        auto index_dram_buffer = input_buffers.at(1);
        auto output_dram_buffer = output_buffers.at(0);

        for (const auto &core : cores) {
            {
                auto &runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
                runtime_args[0] = grad_dram_buffer->address();
                runtime_args[1] = index_dram_buffer->address();
                runtime_args[2] = output_dram_buffer->address();
            }
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

}  // namespace ttnn::operations::embedding_backward::detail
