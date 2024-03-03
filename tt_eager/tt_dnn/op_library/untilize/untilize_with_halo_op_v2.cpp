// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <math.h>

#include <algorithm>

#include "tensor/owned_buffer_functions.hpp"
#include "tt_dnn/op_library/math.hpp"
#include "tt_dnn/op_library/sharding_utilities.hpp"
#include "tt_dnn/op_library/sliding_window_op_infra/utils.hpp"
#include "tt_dnn/op_library/untilize/untilize_op.hpp"
#include "tt_dnn/op_library/work_split.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/common/env_lib.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::constants;

namespace tt {
namespace tt_metal {

using range_t = std::array<int32_t, 2>;
const int32_t NEIGHBORHOOD_DIST = 2;    // => ncores to left and ncores to right

namespace untilize_with_halo_v2_helpers {

int32_t my_max(const std::vector<int32_t>& in) {
    int32_t mmax = 0;
    for (int32_t v : in) {
        mmax = mmax > v ? mmax : v;
    }
    return mmax;
}

} // namespace untilize_with_halo_v2_helpers

operation::ProgramWithCallbacks untilize_with_halo_multi_core_v2(
    const Tensor& input_tensor,
    const uint32_t pad_val,
    const uint32_t ncores_nhw,
    const uint32_t max_out_nsticks_per_core,
    const Tensor& padding_config,
    const Tensor& local_config,
    const Tensor& remote_config,
    Tensor& output_tensor) {
    Program program = CreateProgram();

    auto program_id = program.get_id();
    log_debug(LogOp, "Created Program with ID: {}", program_id);

    Device *device = input_tensor.device();
    Buffer *src_buffer = input_tensor.buffer();
    Buffer *dst_buffer = output_tensor.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    bool skip_untilize = input_tensor.layout() == Layout::ROW_MAJOR;

    Shape input_shape = input_tensor.shape();
    Shape output_shape = output_tensor.shape();

    DataFormat in_df = datatype_to_dataformat_converter(input_tensor.dtype());
    DataFormat out_df = datatype_to_dataformat_converter(output_tensor.dtype());
    uint32_t out_nbytes = datum_size(out_df);

    auto grid_size = device->compute_with_storage_grid_size();
    std::map<CoreCoord, CoreCoord> left_neighbor_core, right_neighbor_core;
    utils::init_neighbor_core_xy_mapping(grid_size, left_neighbor_core, right_neighbor_core, input_tensor.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED);

    uint32_t ncores_x = grid_size.x;
    uint32_t ncores_y = grid_size.y;

    CoreRangeSet all_cores = input_tensor.shard_spec().value().grid;
    uint32_t ncores = all_cores.num_cores();
    uint32_t ncores_c = 1;
    if (input_tensor.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
        auto core_range = *(all_cores.ranges().begin());
        ncores = core_range.end.x - core_range.start.x + 1;
        ncores_c = core_range.end.y - core_range.start.y + 1;
    }
    log_debug(LogOp, "ncores_c: {}", ncores_c);
    TT_ASSERT(ncores_nhw == ncores);

    auto shard_shape = input_tensor.shard_spec().value().shape;
    uint32_t ntiles_per_block = shard_shape[1] / TILE_WIDTH;
    uint32_t nblocks_per_core = shard_shape[0] / TILE_HEIGHT;
    uint32_t input_npages = ntiles_per_block * nblocks_per_core;

    uint32_t out_stick_nbytes = shard_shape[1] * out_nbytes;

    uint32_t in_page_size = detail::TileSize(in_df);
    uint32_t out_tile_size = detail::TileSize(out_df);

    if (skip_untilize) {
        uint32_t in_nbytes = datum_size(in_df);
        in_page_size = shard_shape[1] * in_nbytes;
        input_npages = shard_shape[0];
    }

    // Construct CBs
    // //

    uint32_t src_cb_id = CB::c_in0;
    uint32_t pad_cb_id = CB::c_in1;
    uint32_t untilize_out_cb_id = CB::c_out0;
    uint32_t out_cb_id = CB::c_out1;

    // input CB (sharded)
    auto src_cb_config = CircularBufferConfig(input_npages * in_page_size, {{src_cb_id, in_df}})
                            .set_page_size(src_cb_id, in_page_size)
                            .set_globally_allocated_address(*src_buffer);
    auto src_cb = CreateCircularBuffer(program, all_cores, src_cb_config);
    log_debug(LogOp, "CB {} :: npages = {}, pagesize = {}, total = {}", src_cb_id, input_npages, in_page_size, input_npages * in_page_size);

    uint32_t input_to_writer_cb_id = src_cb_id;
    if (!skip_untilize) {
        input_to_writer_cb_id = untilize_out_cb_id;

        // output of untilize from compute kernel goes into this CB
        uint32_t output_ntiles = ntiles_per_block * nblocks_per_core;
        auto untilize_out_cb_config = CircularBufferConfig(output_ntiles * out_tile_size, {{untilize_out_cb_id, out_df}})
                                        .set_page_size(untilize_out_cb_id, out_tile_size);
        auto untilize_out_cb = CreateCircularBuffer(program, all_cores, untilize_out_cb_config);
        log_debug(LogOp, "CB {} :: npages = {}, pagesize = {}, total = {}", untilize_out_cb_id, output_ntiles, out_tile_size, output_ntiles * out_tile_size);
    }

    // output shard, after inserting halo and padding, goes into this CB as input to next op.
    uint32_t out_cb_pagesize = out_stick_nbytes;
    uint32_t out_cb_npages = max_out_nsticks_per_core;
    auto out_cb_config = CircularBufferConfig(out_cb_npages * out_cb_pagesize, {{out_cb_id, out_df}})
                            .set_page_size(out_cb_id, out_cb_pagesize)
                            .set_globally_allocated_address(*dst_buffer);
    auto out_cb = CreateCircularBuffer(program, all_cores, out_cb_config);
    log_debug(LogOp, "CB {} :: npages = {}, pagesize = {}, total = {}", out_cb_id, out_cb_npages, out_cb_pagesize, out_cb_npages * out_cb_pagesize);

    // CB for pad val buffer (stick sized)
    uint32_t pad_cb_pagesize = out_stick_nbytes;
    uint32_t pad_cb_npages = 1;
    auto pad_cb_config = CircularBufferConfig(pad_cb_pagesize * pad_cb_npages, {{pad_cb_id, out_df}})
                            .set_page_size(pad_cb_id, pad_cb_pagesize);
    auto pad_cb = CreateCircularBuffer(program, all_cores, pad_cb_config);
    log_debug(LogOp, "CB {} :: npages = {}, pagesize = {}, total = {}", pad_cb_id, pad_cb_npages, pad_cb_pagesize, pad_cb_npages * pad_cb_pagesize);

    // Additional CBs for sharded data kernel configs
    uint32_t padding_config_cb_id = CB::c_in2;
    uint32_t local_config_cb_id = CB::c_in3;
    uint32_t remote_config_cb_id = CB::c_in4;

    DataFormat kernel_config_df = DataFormat::RawUInt16;        // NOTE: UInt16 is not supported for CB types
    uint32_t config_nbytes = datum_size(kernel_config_df) * 2;  // each config is a pair "start, size", so double the size
    uint32_t pagesize = 0;

    // Gather data
    if (!skip_untilize) {
        // compute kernel
        std::vector<uint32_t> compute_ct_args = {nblocks_per_core, ntiles_per_block};
        std::string compute_kernel("tt_eager/tt_dnn/op_library/untilize/kernels/compute/pack_untilize.cpp");
        if (ntiles_per_block > MAX_PACK_UNTILIZE_WIDTH) {
            log_debug(
                LogOp,
                "Falling back to slow untilize since ntiles_per_block {} > MAX_PACK_UNTILIZE_WIDTH {}",
                ntiles_per_block,
                MAX_PACK_UNTILIZE_WIDTH);
            compute_kernel = std::string("tt_eager/tt_dnn/op_library/untilize/kernels/compute/untilize.cpp");
        }
        KernelHandle untilize_kernel_id =
            CreateKernel(program, compute_kernel, all_cores, ComputeConfig{.compile_args = compute_ct_args});
    }

    TT_ASSERT(padding_config.dtype() == DataType::UINT16);
    TT_ASSERT(local_config.dtype() == DataType::UINT16);
    TT_ASSERT(remote_config.dtype() == DataType::UINT16);

    Buffer* padding_config_buffer = padding_config.buffer();
    auto padding_config_cb_config =
        CircularBufferConfig(padding_config_buffer->size(), {{padding_config_cb_id, kernel_config_df}})
            .set_page_size(padding_config_cb_id, padding_config_buffer->page_size())
            .set_globally_allocated_address(*padding_config_buffer);
    CBHandle padding_config_cb = CreateCircularBuffer(program, all_cores, padding_config_cb_config);
    log_debug(LogOp, "CB {} :: npages = {}, pagesize = {}, total = {}", padding_config_cb_id, padding_config_buffer->size() / padding_config_buffer->page_size(), padding_config_buffer->page_size(), padding_config_buffer->size());

    Buffer* local_config_buffer = local_config.buffer();
    auto local_config_cb_config =
        CircularBufferConfig(local_config_buffer->size(), {{local_config_cb_id, kernel_config_df}})
            .set_page_size(local_config_cb_id, local_config_buffer->page_size())
            .set_globally_allocated_address(*local_config_buffer);
    CBHandle local_config_cb = CreateCircularBuffer(program, all_cores, local_config_cb_config);
    log_debug(LogOp, "CB {} :: npages = {}, pagesize = {}, total = {}", local_config_cb_id, local_config_buffer->size() / local_config_buffer->page_size(), local_config_buffer->page_size(), local_config_buffer->size());

    Buffer* remote_config_buffer = remote_config.buffer();
    auto remote_config_cb_config =
        CircularBufferConfig(remote_config_buffer->size(), {{remote_config_cb_id, kernel_config_df}})
            .set_page_size(remote_config_cb_id, remote_config_buffer->page_size())
            .set_globally_allocated_address(*remote_config_buffer);
    CBHandle remote_config_cb = CreateCircularBuffer(program, all_cores, remote_config_cb_config);
    log_debug(LogOp, "CB {} :: npages = {}, pagesize = {}, total = {}", remote_config_cb_id, remote_config_buffer->size() / remote_config_buffer->page_size(), remote_config_buffer->page_size(), remote_config_buffer->size());

    bool const is_block_sharded = input_tensor.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED;

    // reader kernel
    std::vector<uint32_t> reader_ct_args = {
        0,  // padding_config_cb_id
        0,  // local_config_cb_id
        0,  // remote_config_cb_id
        src_cb_id,
        input_to_writer_cb_id,
        out_cb_id,
        pad_cb_id,
        pad_val,
        input_npages,
        out_stick_nbytes,
        is_block_sharded,
    };

    reader_ct_args[0] = 0;
    reader_ct_args[1] = local_config_cb_id;
    reader_ct_args[2] = 0;

    KernelHandle reader_kernel_id0 = CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/untilize/kernels/dataflow/halo_gather.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = reader_ct_args});

    reader_ct_args[0] = padding_config_cb_id;
    reader_ct_args[1] = 0;
    reader_ct_args[2] = remote_config_cb_id;

    KernelHandle reader_kernel_id1 = CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/untilize/kernels/dataflow/halo_gather.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_ct_args});

    auto override_runtime_arguments_callback =
        [src_cb, out_cb, padding_config_cb, local_config_cb, remote_config_cb](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>&,
            const std::vector<Tensor>& output_tensors) {
            auto src_buffer = input_tensors.at(0).buffer();
            auto padding_config_buffer = input_tensors.at(1).buffer();
            auto local_config_buffer = input_tensors.at(2).buffer();
            auto remote_config_buffer = input_tensors.at(3).buffer();
            auto dst_buffer = output_tensors.at(0).buffer();

            UpdateDynamicCircularBufferAddress(program, src_cb, *src_buffer);
            UpdateDynamicCircularBufferAddress(program, out_cb, *dst_buffer);
            UpdateDynamicCircularBufferAddress(program, padding_config_cb, *padding_config_buffer);
            UpdateDynamicCircularBufferAddress(program, local_config_cb, *local_config_buffer);
            UpdateDynamicCircularBufferAddress(program, remote_config_cb, *remote_config_buffer);
        };

    return {
        .program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

void validate_untilize_with_halo_v2_config_tensor(const Tensor& tensor) {
    TT_FATAL(tensor.buffer() != nullptr, "Input tensors need to be allocated buffers on device");
    TT_FATAL(tensor.layout() == Layout::ROW_MAJOR);
    TT_FATAL(tensor.memory_config().is_sharded());
    TT_FATAL(tensor.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED);
}

void UntilizeWithHaloV2::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);

    // validate input data tensor
    if (input_tensor.layout() == Layout::ROW_MAJOR) {
        // skip the untilize, only do halo
        log_debug(LogOp, "Input is ROW_MAJOR, no need to untilize.");
    } else {
        TT_FATAL(input_tensor.volume() % TILE_HW == 0);
    }
    TT_FATAL(input_tensor.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED || input_tensor.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED);
    TT_FATAL(input_tensor.shard_spec().has_value());
}

std::vector<Shape> UntilizeWithHaloV2::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input = input_tensors.at(0);
    const auto& input_shape = input.shape();
    Shape output_shape = input_shape;

    uint32_t nbatch = input_shape[0];
    uint32_t total_nsticks = ncores_nhw_ * max_out_nsticks_per_core_;

    // output_shape[0] remains same
    // output_shape[1] remains same
    // output_shape[2] changes
    // output_shape[3] remains same
    output_shape[2] = (uint32_t) ceil((float) total_nsticks / nbatch);

    log_debug(LogOp, "output_shape: [{} {} {} {}]", output_shape[0], output_shape[1], output_shape[2], output_shape[3]);
    log_debug(LogOp, "max_out_nsticks_per_core: {}", max_out_nsticks_per_core_);
    log_debug(LogOp, "ncores_nhw: {}", ncores_nhw_);

    return {output_shape};
}

std::vector<Tensor> UntilizeWithHaloV2::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    // NOTE: output is always ROW_MAJOR
    DataType output_dtype = input_tensor.dtype() == DataType::BFLOAT8_B ? DataType::BFLOAT16 : input_tensor.dtype();
    auto shard_spec = input_tensor.shard_spec().value();
    // log_debug(LogOp, "INPUT SHARD SPEC: {}", shard_spec);
    auto output_shape = this->compute_output_shapes(input_tensors).at(0);

    if (input_tensor.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
        auto core_range = *(shard_spec.grid.ranges().begin());
        TT_FATAL(ncores_nhw_ == core_range.end.x - core_range.start.x + 1);
    } else {
        TT_FATAL(ncores_nhw_ == shard_spec.grid.num_cores(), "ncores_nhw ({}) != shard_spec.grid.num_cores ({})", ncores_nhw_, shard_spec.grid.num_cores());
    }
    auto out_shard_spec = shard_spec;
    out_shard_spec.shape[0] = div_up(output_shape[0] * output_shape[2], ncores_nhw_);
    out_shard_spec.halo = true;
    // log_debug(LogOp, "OUTPUT SHARD SPEC: {}", out_shard_spec);
    auto mem_config = out_mem_config_;
    mem_config.shard_spec = out_shard_spec;
    return {create_sharded_device_tensor(output_shape, output_dtype, Layout::ROW_MAJOR, input_tensor.device(), mem_config)};
}

operation::ProgramWithCallbacks UntilizeWithHaloV2::create_program(const std::vector<Tensor>& inputs, std::vector<Tensor> &outputs) const {
    const auto& input_tensor = inputs.at(0);
    const auto& padding_config = inputs.at(1);
    const auto& local_config = inputs.at(2);
    const auto& remote_config = inputs.at(3);
    auto& output_tensor = outputs.at(0);

    return {untilize_with_halo_multi_core_v2(
        input_tensor,
        pad_val_,
        ncores_nhw_,
        max_out_nsticks_per_core_,
        padding_config,
        local_config,
        remote_config,
        output_tensor)};
}

Tensor untilize_with_halo_v2(
    const Tensor& input_tensor,
    const Tensor& padding_config,
    const Tensor& local_config,
    const Tensor& remote_config,
    const uint32_t pad_val,
    const uint32_t ncores_nhw,
    const uint32_t max_out_nsticks_per_core,
    const MemoryConfig& mem_config) {
    TT_ASSERT(input_tensor.memory_config().is_sharded());
    TT_ASSERT(input_tensor.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED || input_tensor.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED);
    // NOTE: for HEIGHT_SHARDED, ncores_nhw == ncores
    //       for BLOCK_SHARDED, ncores_nhw is just the ncores along height dim (last tensor dim is split along width)

    return operation::run_without_autoformat(
               UntilizeWithHaloV2{pad_val, ncores_nhw, max_out_nsticks_per_core, mem_config},
               {
                   input_tensor,
                   padding_config,
                   local_config,
                   remote_config,
               })
        .at(0);
}

}  // namespace tt_metal

}  // namespace tt
