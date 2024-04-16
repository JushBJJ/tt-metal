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

operation::ProgramWithCallbacks multi_core_nlp_create_qkv_heads(const Tensor &input_tensor, const uint32_t num_q_heads, const uint32_t head_dim, Tensor& output, CoreCoord compute_with_storage_grid_size) {

    const auto& input_shape = input_tensor.get_legacy_shape();

    tt_metal::Device *device = input_tensor.device();

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());

    const bool read_from_input_tensor_kv = input_tensor_kv.has_value();

    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);
    tt_metal::Buffer *in0_buffer = input_tensor.buffer();
    TT_ASSERT(in0_buffer->size() % single_tile_size == 0);

    tt_metal::Buffer *in1_buffer;
    uint32_t in1_buffer_addr = 0;
    tt_metal::BufferType in1_buffer_type = tt_metal::BufferType::DRAM;
    if (read_from_input_tensor_kv) {
        in1_buffer = input_tensor_kv.value().buffer();
        TT_ASSERT(in1_buffer->size() % single_tile_size == 0);
        in1_buffer_addr = in1_buffer->address();
        in1_buffer_type = in1_buffer->buffer_type();
    }


    ////////////////////////////////////////////////////////////////////////////
    //                      TM Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    uint32_t in0_w_tiles = input_shape[3] / TILE_WIDTH;

    // Per output tensor args
    // Output shape for Q is: [B, num_q_heads, s, head_dim], shuffled from [B, 1, s, num_q_heads * head_dim]
    uint32_t q_out_h_tiles = input_shape[2] / TILE_HEIGHT;
    uint32_t q_out_w_tiles = head_dim / TILE_WIDTH; // tiles along head_dim
    uint32_t q_out_HtWt = q_out_h_tiles * q_out_w_tiles;
    uint32_t q_out_CHtWt = num_q_heads * q_out_HtWt;
    uint32_t q_num_tiles = num_q_heads * q_out_w_tiles;

    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    // Block is a unit of work; ie. num of in0_w_tiles per core
    uint32_t num_blocks = input_shape[0] * input_shape[1] * input_shape[2] / TILE_HEIGHT;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_blocks_per_core_group_1, num_blocks_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, num_blocks);


    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Tensor& q = output;

    tt_metal::Buffer *q_buffer = q.buffer();
    TT_ASSERT(q_buffer != nullptr, "Output q buffer should be allocated on device!");


    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program program = tt_metal::CreateProgram();

    bool tile_dtype_is_bfloat16 = input_tensor.get_dtype() == tt::tt_metal::DataType::BFLOAT16;
    bool in0_is_dram = in0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;

    // TODO: Q, K, V doesn't necessarily need to be the same output mem config
    bool out_is_dram = q_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {
            // interleaved accessor args
            (std::uint32_t) in0_is_dram,
            (std::uint32_t) q_num_tiles,
    };
    std::vector<uint32_t> writer_compile_time_args = {
            // interleaved accessor args
            (std::uint32_t) out_is_dram,
            (std::uint32_t) q_out_h_tiles,
            (std::uint32_t) q_out_w_tiles,
            (std::uint32_t) q_out_HtWt,
            (std::uint32_t) num_q_heads, // q_out_c
    };

    std::map<string, string> reader_defines;
    std::map<string, string> writer_defines;

    auto reader_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/nlp_tms/kernels/dataflow/reader_tm_tile_layout_nlp_create_q_heads.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args, reader_defines));
    auto writer_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/nlp_tms/kernels/dataflow/writer_tm_tile_layout_nlp_create_q_heads.cpp",
        all_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args, writer_defines));


    // Create circular buffers
    uint32_t micro_block_size = 1; // Num tiles to read/wait for in reader and writer
    uint32_t cb_num_tiles = micro_block_size * 4; // Quadruple buffer everything

    // TODO: Investigate perf allocating full in0_w_tiles with double buffer
    // uint32_t cb1_num_tiles = in0_w_tiles * 2; // double buffer; this runs out of space for generic shapes
    uint32_t src1_cb_index = 1; // cb0 is needed for compute if we want to use generic transpose_wh compute kernel
    uint32_t cb1_num_tiles = cb_num_tiles;
    tt_metal::CircularBufferConfig cb_src1_config = tt_metal::CircularBufferConfig(cb1_num_tiles * single_tile_size, {{src1_cb_index, cb_data_format}})
		.set_page_size(src1_cb_index, single_tile_size);
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);

    for (uint32_t i = 0, num_blocks_written = 0; i < num_cores; i++){
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_blocks_per_core = 0;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_blocks_per_core = num_blocks_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_blocks_per_core = num_blocks_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        std::vector<uint32_t> reader_runtime_args = {
            (std::uint32_t) in0_buffer->address(),
            num_blocks_per_core,
            num_blocks_written * in0_w_tiles,
        };

        uint32_t q_out_h_dim = num_blocks_written % q_out_h_tiles;
        uint32_t q_out_tensor_tile_id = num_blocks_written / q_out_h_tiles * q_out_CHtWt + q_out_h_dim * q_out_w_tiles;

        std::vector<uint32_t> writer_runtime_args = {
            (std::uint32_t) q_buffer->address(), // q_tensor_addr
            num_blocks_per_core, // num_blocks
            q_out_h_dim, // q_out_h_dim
            q_out_tensor_tile_id, // q_out_tensor_tile_id
        };

        tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
        tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);
        num_blocks_written += num_blocks_per_core;
    }

    auto override_runtime_arguments_callback = [
            reader_kernel_id,
            writer_kernel_id,
            num_cores,
            num_cores_y,
            read_from_input_tensor_kv=read_from_input_tensor_kv
        ]
    (
        const void* operation,
        Program &program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        const std::vector<Tensor>& output_tensors
    ) {

        auto src_buffer = input_tensors.at(0).buffer();

        auto dst_buffer_query = output_tensors.at(0).buffer();

        for (uint32_t i = 0, num_blocks_written = 0; i < num_cores; i++){
            CoreCoord core = {i / num_cores_y, i % num_cores_y};

            {
                auto &runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
                runtime_args[0] = src_buffer->address();
            }

            {
                auto &runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
                runtime_args[0] = dst_buffer_query->address();
            }
        }
    };

    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_arguments_callback};
}

} // namespace tt_metal

} // namespace tt
