// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/nlp_tms/nlp_tms.hpp"
#include "tt_dnn/op_library/work_split.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

using namespace tt::constants;
using namespace tt;

namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks multi_core_tutorial_nlp_concat_heads(const Tensor &a, Tensor& output, CoreCoord compute_with_storage_grid_size) {

    const auto& ashape = a.get_legacy_shape();

    tt_metal::Device *device = a.device();

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(a.get_dtype());

    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);
    tt_metal::Buffer *in0_buffer = a.buffer();


    ////////////////////////////////////////////////////////////////////////////
    //                      TM Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    uint32_t per_tensor_tiles = ashape[1] * ashape[3] / TILE_WIDTH; // 142

    // Per output tensor args
    // Output shape is: [B, 1, s, 4544]
    uint32_t in0_h_tiles = ashape[2] / TILE_HEIGHT;
    uint32_t in0_w_tiles = ashape[3] / TILE_WIDTH; // head_dim
    uint32_t in0_c = per_tensor_tiles / in0_w_tiles; // num_heads
    uint32_t in0_HtWt = in0_h_tiles * in0_w_tiles;
    uint32_t in0_CHtWt = in0_c * in0_HtWt;

    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    // Block is a unit of work; ie. num of per_tensor_tiles per core
    uint32_t num_blocks = ashape[0] * ashape[2] / TILE_HEIGHT;
    uint32_t num_cores = 0, num_blocks_per_core_group_1 = 0, num_blocks_per_core_group_2 = 0;
    CoreRangeSet all_cores = CoreRangeSet({}), core_group_1 = CoreRangeSet({}), core_group_2 = CoreRangeSet({});
    bool row_major = false;

    std::tie(num_cores, all_cores, core_group_1, core_group_2, num_blocks_per_core_group_1, num_blocks_per_core_group_2) = split_work_to_cores(compute_with_storage_grid_size, num_blocks);

    uint32_t g1_numcores = core_group_1.num_cores();
    uint32_t g2_numcores = core_group_2.num_cores();

    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Buffer *out_buffer = output.buffer();
    TT_ASSERT(out_buffer != nullptr, "Output buffer should be allocated on device!");


    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program program = tt_metal::CreateProgram();
    uint32_t src0_cb_index = 0, out_cb_index = 16;

    bool in0_is_dram = in0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool out_is_dram = out_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;

    KernelHandle reader_kernel_id = 0, writer_kernel_id = 0;
    std::vector<uint32_t> reader_compile_time_args = {
        // interleaved accessor args
        (std::uint32_t) in0_is_dram,
        (std::uint32_t) in0_h_tiles,
        (std::uint32_t) in0_w_tiles,
        (std::uint32_t) in0_c,
        (std::uint32_t) in0_HtWt,
    };
    std::vector<uint32_t> writer_compile_time_args = {
        // interleaved accessor args
        (std::uint32_t) src0_cb_index,
        (std::uint32_t) out_is_dram,
    };
    reader_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/nlp_tms/kernels/dataflow/reader_tm_tile_layout_tutorial_nlp_concat_heads.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    writer_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/nlp_tms/kernels/dataflow/writer_tm_tile_layout_tutorial_nlp_concat_heads.cpp",
        all_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));


    // Create circular buffers
    CBHandle cb_src0 = 0, cb_out = 0;
    uint32_t cb_src0_num_tiles = per_tensor_tiles;
    cb_src0_num_tiles *= 2; // double buffer

    tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(cb_src0_num_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
		.set_page_size(src0_cb_index, single_tile_size);
    cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    const auto cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, row_major);
    for (uint32_t i = 0, num_blocks_written = 0; i < cores.size(); ++i){
        const CoreCoord &core = cores[i];
        uint32_t num_blocks_per_core = i < g1_numcores ? num_blocks_per_core_group_1  : num_blocks_per_core_group_2;

        uint32_t in0_h_dim = num_blocks_written % in0_h_tiles;
        uint32_t in0_tensor_tile_id = num_blocks_written / in0_h_tiles * in0_CHtWt + in0_h_dim * in0_w_tiles;

        std::vector<uint32_t> reader_runtime_args = {
            (std::uint32_t) in0_buffer->address(),
            num_blocks_per_core, // num_blocks
            in0_h_dim, // in0_h_dim
            in0_tensor_tile_id, // in0_tensor_tile_id
        };

        std::vector<uint32_t> writer_runtime_args = {
            (std::uint32_t) out_buffer->address(), // out_tensor_addr
            num_blocks_per_core * per_tensor_tiles,
            num_blocks_written * per_tensor_tiles,
        };

        tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
        tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);
        num_blocks_written += num_blocks_per_core;
    }

    auto override_runtime_arguments_callback = [
            reader_kernel_id,
            writer_kernel_id,
            cb_src0,
            cb_out,
            cores
        ]
    (
        const void* operation,
        Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>&,
        const std::vector<Tensor>& output_tensors
    ) {

        const auto src_buffer = input_tensors.at(0).buffer();
        const auto dst_buffer = output_tensors.at(0).buffer();
        const bool in_sharded = input_tensors.at(0).is_sharded();
        const bool out_sharded = output_tensors.at(0).is_sharded();

        for (const auto& core : cores) {
            auto &runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = src_buffer->address();
        }

        for (const auto& core : cores) {
            auto &runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
        }
    };
    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_arguments_callback};
}

} // namespace tt_metal

} // namespace tt
