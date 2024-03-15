// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>

#include "tt_dnn/op_library/non_zero_indices/non_zero_indices_op.hpp"
#include "tt_dnn/op_library/work_split.hpp"
#include "tt_dnn/op_library/math.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks non_zero_indices_single_core(const Tensor &input, const Tensor &out_num_indices, const Tensor &out_indices) {
    tt_metal::Program program{};
    Device *device = input.device();


    uint32_t alignment_base = 32/input.element_size();
    //we want per core to be aligned to aligment_base per core

    uint32_t aligned_elements = div_up(input.get_legacy_shape()[-1] , alignment_base) * alignment_base;
    uint32_t actual_elements = input.get_legacy_shape()[-1];

    CoreCoord core = {0,0};

    uint32_t input_cb_index = 0;
    uint32_t output_cb_index_0 = 1;
    uint32_t output_cb_index_1 = 2;

    tt::DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(input.get_dtype());
    tt::DataFormat output_cb_data_format = tt_metal::datatype_to_dataformat_converter(DataType::UINT32);
    bool src_is_dram = input.buffer()->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool out_is_dram_0 = out_num_indices.buffer()->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool out_is_dram_1 = out_indices.buffer()->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;

    uint32_t page_size = actual_elements * input.element_size();
    uint32_t rounded_page_size = round_up_to_mul32(page_size);
    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(2* rounded_page_size, {{input_cb_index, input_cb_data_format}})
            .set_page_size(input_cb_index, rounded_page_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    tt_metal::CircularBufferConfig cb_dst0_config =
        tt_metal::CircularBufferConfig(2* 32, {{output_cb_index_0, output_cb_data_format}})
            .set_page_size(output_cb_index_0, 32);
    auto cb_dst0 = tt_metal::CreateCircularBuffer(program, core, cb_dst0_config);

    uint32_t dst_page_size = actual_elements * 4;
    uint32_t dst_rounded_page_size = round_up_to_mul32(dst_page_size);
    tt_metal::CircularBufferConfig cb_dst1_config =
        tt_metal::CircularBufferConfig(2* dst_rounded_page_size , {{output_cb_index_1, output_cb_data_format}})
            .set_page_size(output_cb_index_1, dst_rounded_page_size);
    auto cb_dst1 = tt_metal::CreateCircularBuffer(program, core, cb_dst1_config);


    std::map<string, string> defines;

    if(input.get_dtype() == DataType::FLOAT32) {
        defines["NUM_BYTES"] = "4";
    }
    else if(input.get_dtype() == DataType::UINT32) {
        defines["NUM_BYTES"] = "4";
    }
    else if(input.get_dtype() == DataType::BFLOAT16) {
        defines["NUM_BYTES"] = "2";
    }
    else if(input.get_dtype() == DataType::BFLOAT8_B) {
        defines["NUM_BYTES"] = "1";
    }
    // Create Kernel
    std::vector<uint32_t> compile_time_args = {
        (std::uint32_t) input_cb_index,
        (std::uint32_t) output_cb_index_0,
        (std::uint32_t) output_cb_index_1,
        (std::uint32_t) src_is_dram,
        (std::uint32_t) out_is_dram_0,
        (std::uint32_t) out_is_dram_1,
    };

    std::vector<uint32_t> run_time_args = {
        (std::uint32_t) input.buffer()->address(),
        (std::uint32_t) out_num_indices.buffer()->address(),
        (std::uint32_t) out_indices.buffer()->address(),
        (std::uint32_t) aligned_elements,
        (std::uint32_t) actual_elements,
        (std::uint32_t) input.element_size()
    };

    auto kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/non_zero_indices/kernels/dataflow/non_zero_indices_sc_reader.cpp",
        core,
        tt_metal::ReaderDataMovementConfig(
            compile_time_args, defines));

    tt_metal::SetRuntimeArgs(program, kernel_id, core, run_time_args);


    auto override_runtime_args_callback = [kernel_id, core,  page_size](
        const void* operation,
        const Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>&,
        const std::vector<Tensor>& output_tensors
    ) {
        auto output_0 = output_tensors.at(0);
        auto output_1 = output_tensors.at(1);
        auto input = input_tensors.at(1);
        uint32_t alignment_base = 32/input.element_size();
        uint32_t aligned_elements = div_up(input.get_legacy_shape()[-1] , alignment_base) * alignment_base;
        uint32_t actual_elements = input.get_legacy_shape()[-1];
        auto runtime_args = tt_metal::GetRuntimeArgs(program, kernel_id, core);
        runtime_args[0] = input.buffer()->address();
        runtime_args[1] = output_0.buffer()->address();
        runtime_args[2] = output_1.buffer()->address();
        runtime_args[3] = aligned_elements;
        runtime_args[4] = actual_elements;
        runtime_args[5] = input.element_size();
        tt_metal::SetRuntimeArgs(program, kernel_id, core, runtime_args);
    };
    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_args_callback};
}

}  // namespace tt_metal

}  // namespace tt
