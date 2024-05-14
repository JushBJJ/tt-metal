// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_eager/tt_dnn/op_library/topk/topk_op.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

namespace tt {
namespace tt_metal{

operation::ProgramWithCallbacks single_core_topk_interleaved(const Tensor &input_tensor, const uint32_t k, Tensor &value_tensor, Tensor &index_tensor) {
    Program program{};

    CoreRange core({0, 0}, {0, 0});
    tt::DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());
    tt::DataFormat value_cb_data_format = tt_metal::datatype_to_dataformat_converter(value_tensor.get_dtype());
    tt::DataFormat index_cb_data_format = tt_metal::datatype_to_dataformat_converter(index_tensor.get_dtype());

    TT_ASSERT(input_cb_data_format == value_cb_data_format, "Output value tensor and input tensor must have the same datatype!");
    TT_ASSERT(index_cb_data_format == tt::DataFormat::UInt16, "Index tensor must be in Uint16");

    uint32_t input_tile_size = tile_size(input_cb_data_format);
    uint32_t value_tile_size = tile_size(value_cb_data_format);
    uint32_t index_tile_size = tile_size(index_cb_data_format);

    auto input_buffer = input_tensor.buffer();
    auto values_buffer = value_tensor.buffer();
    auto index_buffer = index_tensor.buffer();

    bool input_is_dram = input_buffer->buffer_type() == tt_metal::BufferType::DRAM;
    bool values_is_dram = values_buffer->buffer_type() == tt_metal::BufferType::DRAM;
    bool index_is_dram = index_buffer->buffer_type() == tt_metal::BufferType::DRAM;

    uint32_t num_input_tiles = input_tensor.volume()/TILE_HW;
    uint32_t num_value_tiles = value_tensor.volume()/TILE_HW;

    auto input_shape = input_tensor.get_legacy_shape();
    uint32_t Ht = (input_shape[0]*input_shape[1]*input_shape[2])/TILE_HEIGHT;
    uint32_t Wt = input_shape[3]/TILE_WIDTH;
    // for streaming in input
    uint32_t num_cb_unit = 2;

    uint32_t input_cb_index = CB::c_in0;
    tt_metal::CircularBufferConfig input_cb_config = tt_metal::CircularBufferConfig(num_cb_unit * value_tile_size, {{input_cb_index, input_cb_data_format}})
		.set_page_size(input_cb_index, input_tile_size);
    auto cb_input_tensor = tt_metal::CreateCircularBuffer(program, core, input_cb_config);

    // populate this as input is streamed
    uint32_t index_cb_index = CB::c_in1;
    tt_metal::CircularBufferConfig index_input_intermed0_config = tt_metal::CircularBufferConfig(num_cb_unit * index_tile_size, {{index_cb_index, index_cb_data_format}})
		.set_page_size(index_cb_index, index_tile_size);
    auto cb_index_tensor = tt_metal::CreateCircularBuffer(program, core, index_input_intermed0_config);


    // transpose and populate a CB with one row of tiles at a time - precisely one row of space, since we only work on it when it's full we shouldn't need to double buffer...I think, will have to ask
    uint32_t input_transposed_cb_index = CB::c_intermed0;
    tt_metal::CircularBufferConfig input_transposed_cb_config = tt_metal::CircularBufferConfig(num_cb_unit * (input_shape[-1]/TILE_WIDTH) * value_tile_size, {{input_transposed_cb_index, input_cb_data_format}})
		.set_page_size(input_transposed_cb_index, input_tile_size);
    auto cb_input_transposed_tiles = tt_metal::CreateCircularBuffer(program, core, input_transposed_cb_config);

    uint32_t index_transposed_cb_index = CB::c_intermed1;
    tt_metal::CircularBufferConfig index_transposed_cb_config = tt_metal::CircularBufferConfig(num_cb_unit * (input_shape[-1]/TILE_WIDTH) * index_tile_size, {{index_transposed_cb_index, index_cb_data_format}})
		.set_page_size(index_transposed_cb_index, index_tile_size);
    auto cb_index_transposed_tiles = tt_metal::CreateCircularBuffer(program, core, index_transposed_cb_config);



    uint32_t values_cb_index = CB::c_out0; // output operands start at index 16
    tt_metal::CircularBufferConfig values_cb_config = tt_metal::CircularBufferConfig(num_cb_unit * value_tile_size, {{values_cb_index, value_cb_data_format}})
        .set_page_size(values_cb_index, value_tile_size);
    auto cb_values_tensor = tt_metal::CreateCircularBuffer(program, core, values_cb_config);


    uint32_t output_ind_cb_index = CB::c_out1; // output operands start at index 16
    tt_metal::CircularBufferConfig output_ind_cb_config = tt_metal::CircularBufferConfig(num_cb_unit * index_tile_size, {{output_ind_cb_index, index_cb_data_format}})
        .set_page_size(output_ind_cb_index, index_tile_size);
    auto cb_output_ind_tensor = tt_metal::CreateCircularBuffer(program, core, output_ind_cb_config);

    std::vector<uint32_t> reader_compile_time_args = {
                                                        input_cb_index,
                                                        index_cb_index,
                                                        (uint32_t)input_is_dram,
                                                        Ht,
                                                        Wt};
    tt_metal::KernelHandle unary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/topk/kernels/reader_create_index_tensor.cpp",
        core,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args));


    SetRuntimeArgs(
        program,
        unary_reader_kernel_id,
        core,
        {
            input_buffer->address(),
        }
    );

    std::vector<uint32_t> writer_compile_time_args = {
                                                        values_cb_index,
                                                        output_ind_cb_index,
                                                        (std::uint32_t) values_is_dram,
                                                        (std::uint32_t) index_is_dram,
                                                        num_value_tiles};
    tt_metal::KernelHandle binary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/topk/kernels/writer_binary_interleaved.cpp",
        core,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    SetRuntimeArgs(
        program,
        binary_writer_kernel_id,
        core,
        {
            values_buffer->address(),
            index_buffer->address(),

        }
    );

    std::vector<uint32_t> compute_args = {
                                        input_cb_index,
                                        index_cb_index,
                                        input_transposed_cb_index,
                                        index_transposed_cb_index,
                                        values_cb_index,
                                        output_ind_cb_index,
                                        Ht,
                                        Wt,
                                        k,
                                        (std::uint32_t) std::log2(k),
                                        (std::uint32_t) std::log2(Wt),
                                        };
    tt_metal::KernelHandle topk_compute_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/topk/kernels/topk.cpp",
        core,
        tt_metal::ComputeConfig{.compile_args = compute_args}
    );


    auto override_runtime_args_callback = [unary_reader_kernel_id, binary_writer_kernel_id](
        const Program &program,
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto input_buffer = input_buffers.at(0);
        auto values_buffer = output_buffers.at(0);
        auto index_buffer = output_buffers.at(1);

        CoreCoord core = {0, 0};

        {
            auto &reader_runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
            reader_runtime_args[0] = input_buffer->address();

            auto &writer_runtime_args = GetRuntimeArgs(program, binary_writer_kernel_id, core);
            writer_runtime_args[0] = values_buffer->address();
            writer_runtime_args[1] = index_buffer->address();
        }

    };

    return {std::move(program), override_runtime_args_callback};
}

}
}
