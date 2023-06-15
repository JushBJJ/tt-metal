#include "tt_dnn/op_library/eltwise_binary/eltwise_binary_op.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {
Program eltwise_binary_single_core(const Tensor &a, const Tensor &b, Tensor& output, BinaryOpType::Enum op_type) {

    Program program{};
    CoreRange core = {.start={0, 0}, .end={0, 0}};

    // TODO: Build some sort of dispatcher based on location of op operands
    TT_ASSERT(not a.on_host() and not b.on_host(), "Operands to eltwise binary need to be on device!");
    TT_ASSERT(a.device() == b.device(), "Operands to eltwise binary need to be on the same device!");
    TT_ASSERT(a.buffer() != nullptr and b.buffer() != nullptr, "Operands to eltwise binary need to be allocated in buffers on device!");

    uint32_t single_tile_size = 2 * TILE_HW;

    tt_metal::Buffer *src0_dram_buffer = a.buffer();
    tt_metal::Buffer *src1_dram_buffer = b.buffer();

    TT_ASSERT(src0_dram_buffer->size() == src1_dram_buffer->size(), "Operand to eltwise binary need to be the same size!");

    TT_ASSERT(a.volume() % TILE_HW == 0);
    uint32_t num_tiles = a.volume() / TILE_HW;

    auto dram_src0_noc_xy = src0_dram_buffer->noc_coordinates();
    auto dram_src1_noc_xy = src1_dram_buffer->noc_coordinates();

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();

    tt_metal::Buffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");
    auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates();

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    auto cb_src0 = tt_metal::CreateCircularBuffers(
        program,
        device,
        src0_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        DataFormat::Float16_b
    );

    uint32_t src1_cb_index = 1;
    auto cb_src1 = tt_metal::CreateCircularBuffers(
        program,
        device,
        src1_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        DataFormat::Float16_b
    );

    uint32_t ouput_cb_index = 16; // output operands start at index 16
    uint32_t num_output_tiles = 2;
    auto cb_output = tt_metal::CreateCircularBuffers(
        program,
        device,
        ouput_cb_index,
        core,
        num_output_tiles,
        num_output_tiles * single_tile_size,
        DataFormat::Float16_b
    );

    tt_metal::DataMovementKernel *binary_reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        //"tt_metal/kernels/dataflow/reader_binary.cpp",
        "tt_metal/kernels/dataflow/reader_dual_8bank.cpp",
        core,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_1_default);

    tt_metal::DataMovementKernel *unary_writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        //"tt_metal/kernels/dataflow/writer_unary.cpp",
        "tt_metal/kernels/dataflow/writer_unary_8bank.cpp",
        core,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_0_default);

    vector<uint32_t> compute_kernel_args = {
        num_tiles, // per_core_block_cnt
        1, // per_core_block_size
    };

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto eltwise_binary_kernel = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/eltwise_binary.cpp",
        core,
        compute_kernel_args,
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode
    );

    eltwise_binary_op_utils::add_defines(eltwise_binary_kernel, op_type);


    tt_metal::WriteRuntimeArgsToDevice(
        device,
        binary_reader_kernel,
        core,
        {src0_dram_buffer->address(),
        (std::uint32_t)dram_src0_noc_xy.x,
        (std::uint32_t)dram_src0_noc_xy.y,
        num_tiles,
        src1_dram_buffer->address(),
        (std::uint32_t)dram_src1_noc_xy.x,
        (std::uint32_t)dram_src1_noc_xy.y,
        num_tiles});

    tt_metal::WriteRuntimeArgsToDevice(
        device,
        unary_writer_kernel,
        core,
        {dst_dram_buffer->address(),
        (std::uint32_t)dram_dst_noc_xy.x,
        (std::uint32_t)dram_dst_noc_xy.y,
        num_tiles});
    // output does not hold any data, contains pointer to buffer on device with the data
    return program;
}

}  // namespace tt_metal

}  // namespace tt
