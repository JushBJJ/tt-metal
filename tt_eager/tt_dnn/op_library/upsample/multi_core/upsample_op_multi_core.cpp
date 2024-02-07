// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <math.h>

#include "tt_dnn/op_library/upsample/upsample_op.hpp"
#include "tt_dnn/op_library/math.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/common/math.hpp"

#include "tt_metal/tt_stl/reflection.hpp"

using namespace tt::constants;

namespace tt {
namespace tt_metal {

operation::ProgramWithCallbacks upsample_multi_core(const Tensor &input, Tensor& output, uint32_t scale_factor_h, uint32_t scale_factor_w) {
    Program program = CreateProgram();
    Device *device = input.device();

    DataFormat input_cb_data_format = datatype_to_dataformat_converter(input.dtype());
    DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());

    // NOTE: input is assumed to have channels last format: {N, H, W, C}, {N, 1, H * W, C}, {1, 1, N * H * W, C}
    // NOTE: Bfp8_b/TILE is not yet supported

    uint32_t input_stick_nbytes = input.shape()[-1] * input.element_size();
    uint32_t output_stick_nbytes = output.shape()[-1] * output.element_size();
    TT_FATAL(input_stick_nbytes == output_stick_nbytes, "Input and output sticks should have same size");

    uint32_t output_nsticks = output.volume() / output.shape()[-1];
    uint32_t input_nsticks = input.volume() / input.shape()[-1];

    auto output_shape = output.shape();

    // sharding
    auto shard_spec = input.shard_spec().value();
    auto all_cores = shard_spec.grid;
    uint32_t ncores = shard_spec.num_cores();
    uint32_t ncores_w = device->compute_with_storage_grid_size().x;

    // TODO: Support non-multiple case
    TT_FATAL(input_nsticks % ncores == 0, "Input sticks should be divisible by number of cores");
    TT_FATAL(output_nsticks % ncores == 0, "Output sticks should be divisible by number of cores");
    uint32_t input_nsticks_per_core = input_nsticks / ncores;
    uint32_t output_nsticks_per_core = output_nsticks / ncores;

    uint32_t in_w = input.shape()[2];
    uint32_t out_w = output.shape()[2];

    // extra limitation to avoid post upsample step of resharding
    TT_ASSERT(input_nsticks_per_core % in_w == 0, "Restriction: Input sticks per core should be divisible by input width. TODO to remove this restriction");

    // CBs

    uint32_t buffering_factor = 1;  // data is already fully buffered in the CBs

    // input data is in a sharded CB
    uint32_t in_cb_id = CB::c_in0;
    uint32_t aligned_input_stick_nbytes = round_up_to_mul32(input_stick_nbytes);
    uint32_t in_cb_pagesize = aligned_input_stick_nbytes;
    uint32_t in_cb_npages = input_nsticks_per_core * buffering_factor;
    CircularBufferConfig cb_src0_config = CircularBufferConfig(
                                            in_cb_pagesize * in_cb_npages,
                                            {{in_cb_id, input_cb_data_format}})
                                          .set_page_size(in_cb_id, in_cb_pagesize)
                                          .set_globally_allocated_address(*input.buffer());
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    // output sharded CB with upsampled data
    uint32_t out_cb_id = CB::c_out0;
    uint32_t aligned_output_stick_nbytes = round_up_to_mul32(output_stick_nbytes);
    uint32_t out_cb_pagesize = aligned_output_stick_nbytes;
    uint32_t out_cb_npages = output_nsticks_per_core * buffering_factor;
    CircularBufferConfig out_cb_config = CircularBufferConfig(
                                            out_cb_pagesize * out_cb_npages,
                                            {{out_cb_id, output_cb_data_format}})
                                          .set_page_size(out_cb_id, out_cb_pagesize)
                                          .set_globally_allocated_address(*output.buffer());
    auto out_cb = tt_metal::CreateCircularBuffer(program, all_cores, out_cb_config);

    log_debug(LogOp, "input_cb: {}, npages: {}, pagesize: {}", in_cb_id, in_cb_npages, in_cb_pagesize);
    log_debug(LogOp, "output_cb: {}, npages: {}, pagesize: {}", out_cb_id, out_cb_npages, out_cb_pagesize);
    log_debug(LogOp, "input_stick_nbytes: {}, output_stick_nbytes: {}", input_stick_nbytes, output_stick_nbytes);
    log_debug(LogOp, "ncores: {}, ncores_w: {}", ncores, ncores_w);
    log_debug(LogOp, "input_nsticks_per_core: {}, output_nsticks_per_core: {}", input_nsticks_per_core, output_nsticks_per_core);

    // Kernels

    std::vector<uint32_t> writer_compile_time_args = {
        in_cb_id,
        out_cb_id,
    };
    auto writer_kernel_fname = std::string("tt_eager/tt_dnn/op_library/upsample/kernels/dataflow/writer_upsample_multi_core_sharded.cpp");
    auto writer_kernel = CreateKernel(
        program,
        writer_kernel_fname,
        all_cores,
        WriterDataMovementConfig{.compile_args = writer_compile_time_args});

    // no reader kernel
    // no compute kernel

    // runtime args

    uint32_t writer_nargs = 7;
    vector<uint32_t> writer_rt_args(writer_nargs);
    writer_rt_args[0] = input_stick_nbytes;
    writer_rt_args[1] = input_nsticks_per_core;
    writer_rt_args[2] = scale_factor_h;
    writer_rt_args[3] = scale_factor_w;
    writer_rt_args[4] = in_w;
    writer_rt_args[5] = out_w;
    writer_rt_args[6] = 0;  // set for each core below

    uint32_t start_input_stick_id = 0;
    for (int32_t core = 0; core < ncores; ++core) {
        CoreCoord core_coord(core % ncores_w, core / ncores_w); // logical

        writer_rt_args[6] = start_input_stick_id;
        SetRuntimeArgs(program, writer_kernel, core_coord, writer_rt_args);

        start_input_stick_id += input_nsticks_per_core;
    }

    auto override_runtime_args_callback = [writer_kernel, in_cb_id, out_cb_id](
        const void* operation,
        Program &program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>&,
        const std::vector<Tensor>& output_tensors
    ) {

        auto src_buffer = input_tensors.at(0).buffer();
        auto dst_buffer = output_tensors.at(0).buffer();

        CoreCoord core = {0, 0};

        UpdateDynamicCircularBufferAddress(program, in_cb_id, *src_buffer);
        UpdateDynamicCircularBufferAddress(program, out_cb_id, *dst_buffer);
    };

    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_args_callback};
}

}  // namespace tt_metal
}  // namespace tt
