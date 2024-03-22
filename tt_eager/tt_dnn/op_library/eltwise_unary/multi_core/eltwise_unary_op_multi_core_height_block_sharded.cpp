// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>

#include "tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "tt_dnn/op_library/work_split.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks eltwise_unary_multi_core_height_or_block_sharded(const Tensor &input, Tensor &output, const std::vector<UnaryWithParam> op_chain){
    Program program = CreateProgram();
    Device *device = input.device();

    auto shard_spec = input.shard_spec().value();
    auto all_cores = shard_spec.grid;
    uint32_t ncores = shard_spec.num_cores();

    auto out_shard_spec = output.shard_spec().value();
    TT_FATAL(out_shard_spec.num_cores() == ncores, "Output tensor should have same number of cores {} as input tensor {}", out_shard_spec.num_cores(), ncores);

    DataFormat act_df = tt_metal::datatype_to_dataformat_converter(input.get_dtype());
    DataFormat out_df = tt_metal::datatype_to_dataformat_converter(output.get_dtype());

    uint32_t input_tile_size = tt::tt_metal::detail::TileSize(act_df);
    uint32_t output_tile_size = tt::tt_metal::detail::TileSize(out_df);

    TT_FATAL(input_tile_size == output_tile_size, "Input and output tile size should be same");
    uint32_t shard_size_in_bytes = shard_spec.numel() * datum_size(act_df);

    uint32_t num_tile_per_core = (shard_size_in_bytes + input_tile_size - 1) / input_tile_size; //ceil value
    //num_tile_per_core = std::max(num_tile_per_core, 1u);
    //TT_FATAL(input_tile_size <= shard_size_in_bytes, "Input tile size should be less than shard size");

    uint32_t ncores_x, ncores_nhw;
    if (input.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
        ncores_x = device->compute_with_storage_grid_size().x;
        ncores_nhw = ncores;
    } else if (input.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
        ncores_x = all_cores.ranges().begin()->end.x + 1;
        ncores_nhw = all_cores.ranges().begin()->end.y + 1;
    } else {
        TT_FATAL(false, "Unsupported sharding layout");
    }
    //all_cores = CoreCoordinateRange({{0, 0}, {ncores_x - 1, ncores_nhw - 1}});
    //CoreRange temp_core({0, 0}, {0, 0});
    //all_cores = temp_core
    uint32_t in_cb_id = CB::c_in0;
    uint32_t buffering_factor = 1;  // data is already fully buffered in the CBs since its sharded
    uint32_t aligned_input_tile_nbytes = round_up_to_mul32(input_tile_size); //will have issue if the page is not multiple of 32
    uint32_t in_cb_pagesize = aligned_input_tile_nbytes;
    uint32_t in_cb_npages = num_tile_per_core * buffering_factor;
    CircularBufferConfig cb_src0_config = CircularBufferConfig(
                                            in_cb_pagesize * in_cb_npages,
                                            {{in_cb_id, act_df}})
                                          .set_page_size(in_cb_id, in_cb_pagesize)
                                          .set_globally_allocated_address(*input.buffer());
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    // output sharded CB
    uint32_t out_cb_id = CB::c_out0;
    CircularBufferConfig out_cb_config = CircularBufferConfig(
                                            in_cb_pagesize * in_cb_npages,
                                            {{out_cb_id, out_df}})
                                          .set_page_size(out_cb_id, in_cb_pagesize)
                                          .set_globally_allocated_address(*output.buffer());
    auto out_cb = tt_metal::CreateCircularBuffer(program, all_cores, out_cb_config);

    log_debug(LogOp, "input_cb: {}, npages: {}, pagesize: {}", in_cb_id, in_cb_npages, in_cb_pagesize);
    log_debug(LogOp, "ncores: {}, ncores_x: {}", ncores, ncores_x);
    log_debug(LogOp, "input_tile_size: {} num_tile_per_core: {} ", input_tile_size, num_tile_per_core);

    auto src_buffer = input.buffer();
    auto dst_buffer = output.buffer();

    bool src_is_dram = src_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    TT_FATAL(src_is_dram == 0, "Input buffer should be in L1");
    std::vector<uint32_t> reader_compile_time_args = {
        in_cb_id,
        out_cb_id
    };

    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    TT_FATAL(dst_is_dram == 0, "Output buffer should be in L1");

    std::map<string, string> kernel_defines;
    tt_metal::KernelHandle unary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/kernels/dataflow/reader_unary_sharded.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args, kernel_defines));

    vector<uint32_t> compute_kernel_args_group_1 = {
         //1, // per_core_block_cnt
         num_tile_per_core, // per_core_block_size
         input_tile_size
    };

    // bool fp32_dest_acc_en = false;
    // bool math_approx_mode = std::all_of(op_chain.begin(), op_chain.end(), [](const auto& u) {return eltwise_unary_op_utils::get_op_approx_mode(u.op_type);});
    // std::map<string, string> unary_defines = eltwise_unary_op_utils::get_block_defines(op_chain);
    // auto eltwise_unary_kernel_group_1_id = tt_metal::CreateKernel(
    //     program,
    //     "tt_metal/kernels/compute/eltwise_sfpu.cpp",
    //     all_cores,
    //     tt_metal::ComputeConfig{
    //         .math_fidelity = MathFidelity::HiFi4,
    //         .fp32_dest_acc_en = fp32_dest_acc_en,
    //         .math_approx_mode = math_approx_mode,
    //         .compile_args = compute_kernel_args_group_1,
    //         .defines = unary_defines
    //     }
    // );


    uint32_t start_input_stick_id = 0;
    if (input.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
        //ncores_nhw = 1; ncores_x = 1;
        for (int32_t core = 0; core < ncores_nhw; ++core) {
            for (int32_t core_x = 0; core_x < ncores_x; ++core_x) {
                CoreCoord core_coord(core_x, core); // logical
                //writer_rt_args[6] = start_input_stick_id;
                ///SetRuntimeArgs(program, writer_kernel, core_coord, writer_rt_args);
                tt_metal::SetRuntimeArgs(
                    program,
                    unary_reader_kernel_id,
                    core_coord,
                    {
                        (uint32_t)(num_tile_per_core),
                        (uint32_t)(input_tile_size)
                    }
                );
            }
            //start_input_stick_id += input_nsticks_per_core;
        }
    } else if (input.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
        for (int32_t core = 0; core < ncores_nhw; ++core) {
            CoreCoord core_coord(core % ncores_x, core / ncores_x); // logical
            // writer_rt_args[6] = start_input_stick_id;
            tt_metal::SetRuntimeArgs(
                    program,
                    unary_reader_kernel_id,
                    core_coord,
                    {
                        (uint32_t)(num_tile_per_core),
                        (uint32_t)(input_tile_size)
                    }
                );
            // start_input_stick_id += input_nsticks_per_core;
        }
    } else {
        TT_FATAL(false, "Unsupported memory layout");
    }

    // if (input.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
    //     ncores_nhw = 1; ncores_x = 1;
    //     for (int32_t core = 0; core < ncores_nhw; ++core) {
    //         for (int32_t core_x = 0; core_x < ncores_x; ++core_x) {
    //             CoreCoord core_coord(core_x, core); // logical
    //             tt_metal::SetRuntimeArgs( program, unary_reader_kernel_id, core_coord, {(uint32_t)(num_tile_per_core),} );
    //             tt_metal::SetRuntimeArgs( program, eltwise_unary_kernel_group_1_id, core_coord, {} );
    //         }
    //     }
    // }

    auto override_runtime_args_callback = [unary_reader_kernel_id, in_cb_id, out_cb_id](
        const void* operation,
        Program &program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>&,
        const std::vector<Tensor>& output_tensors
    ) {

        auto src_buffer = input_tensors.at(0).buffer();
        auto dst_buffer = output_tensors.at(0).buffer();

        UpdateDynamicCircularBufferAddress(program, in_cb_id, *src_buffer);
        UpdateDynamicCircularBufferAddress(program, out_cb_id, *dst_buffer);
    };

    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_args_callback};
}

}  // namespace tt_metal

}  // namespace tt
