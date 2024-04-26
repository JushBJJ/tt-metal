// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/op_library/binary/binary_op.hpp"

#include "third_party/magic_enum/magic_enum.hpp"
#include "tt_dnn/op_library/eltwise_binary/eltwise_binary_op.hpp"
#include "tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "tt_dnn/op_library/work_split.hpp"
#include "tt_eager/tt_dnn/op_library/bcast/bcast_op.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"

namespace ttnn {

namespace operations {

namespace binary {

enum class BinaryProgramType {
    ElementWiseSingleCore,
    ElementWiseMultiCore,
    BroadcastWidthMultiCore,
    BroadcastHeightMultiCore,
    BroadcastHeightAndWidthMultiCore,
};

inline BinaryProgramType get_program_type(const Binary& operation, const std::vector<Tensor>& input_tensors) {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);

    const auto& input_shape_a = input_tensor_a.get_shape();
    const auto& input_shape_b = input_tensor_b.get_shape();

    auto batch_size_0_a = input_shape_a.rank() >= 4 ? input_shape_a[-4] : 1;
    auto batch_size_1_a = input_shape_a.rank() >= 3 ? input_shape_a[-3] : 1;
    auto height_a = input_shape_a[-2];
    auto width_a = input_shape_a[-1];

    auto batch_size_0_b = input_shape_b.rank() >= 4 ? input_shape_b[-4] : 1;
    auto batch_size_1_b = input_shape_b.rank() >= 3 ? input_shape_b[-3] : 1;
    auto height_b = input_shape_b[-2];
    auto width_b = input_shape_b[-1];

    /*
    fmt::print("input_shape_a: {}, input_shape_b: {}\n", input_shape_a, input_shape_b);
    fmt::print(
        "batch_size_0_a: {}, batch_size_1_a: {}, height_a: {}, width_a: {}\n",
        batch_size_0_a,
        batch_size_1_a,
        height_a,
        width_a);
    fmt::print(
        "batch_size_0_b: {}, batch_size_1_b: {}, height_b: {}, width_b: {}\n",
        batch_size_0_b,
        batch_size_1_b,
        height_b,
        width_b);
    */

    if (batch_size_0_a == batch_size_0_b and batch_size_1_a == batch_size_1_b and height_a == height_b and
        width_a == width_b) {
        uint32_t num_tiles = input_tensor_a.volume() / tt::constants::TILE_HW;
        if (num_tiles > 1 or input_tensor_a.memory_config().is_sharded() or
            input_tensor_b.memory_config().is_sharded() or operation.program_config.memory_config.is_sharded()) {
            // fmt::print("BinaryProgramType::ElementWiseMultiCore\n");
            return BinaryProgramType::ElementWiseMultiCore;
        } else {
            // fmt::print("BinaryProgramType::ElementWiseSingleCore\n");
            return BinaryProgramType::ElementWiseSingleCore;
        }
    } else if (height_b == 1 or width_b == 1) {
        if (operation.program_config.dtype != input_tensor_a.get_dtype()) {
            TT_THROW("ttnn::operations::binary::Binary: cannot change dtype when broadcasting");
        }
        if (height_b == 1 and width_b == 1) {
            // fmt::print("BinaryProgramType::BroadcastHeightAndWidthMultiCore\n");
            return BinaryProgramType::BroadcastHeightAndWidthMultiCore;
        } else if (height_b == 1) {
            // fmt::print("BinaryProgramType::BroadcastHeightMultiCore\n");
            return BinaryProgramType::BroadcastHeightMultiCore;
        } else if (width_b == 1) {
            // fmt::print("BinaryProgramType::BroadcastWidthMultiCore\n");
            return BinaryProgramType::BroadcastWidthMultiCore;
        }
    }
    TT_THROW("ttnn::operations::binary::Binary: unsupported broadcast");
}

inline void validate_shape(const ttnn::Shape& shape) {
    // Validate that intended and tile-padded shapes are either the same
    // or tile-padded dim is the indended dim padded to the next multiple of ttnn::TILE_SIZE

    for (int i = 0; i < shape.rank(); i++) {
        TT_ASSERT(
            shape[i] == shape.with_tile_padding()[i] or
                pad_to_multiple_of_tile_size(shape[i]) == shape.with_tile_padding()[i],
            "{}: pad_to_multiple_of_tile_size({}) -> {} != {}",
            i,
            shape[i],
            pad_to_multiple_of_tile_size(shape[i]),
            shape.with_tile_padding()[i]);
    }
}

void Binary::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);

    const auto& input_shape_a = input_tensor_a.get_shape();
    const auto& input_shape_b = input_tensor_b.get_shape();

    auto batch_size_0_a = input_shape_a.rank() >= 4 ? input_shape_a[-4] : 1;
    auto batch_size_1_a = input_shape_a.rank() >= 3 ? input_shape_a[-3] : 1;
    auto height_a = input_shape_a[-2];
    auto width_a = input_shape_a[-1];

    auto batch_size_0_b = input_shape_b.rank() >= 4 ? input_shape_b[-4] : 1;
    auto batch_size_1_b = input_shape_b.rank() >= 3 ? input_shape_b[-3] : 1;
    auto height_b = input_shape_b[-2];
    auto width_b = input_shape_b[-1];

    // Input shape b must be the same as or broadcastable to input shape a
    if (batch_size_0_a != batch_size_0_b) {
        TT_ASSERT(
            batch_size_0_a > batch_size_0_b and batch_size_0_b == 1,
            "ttnn::operations::binary::Binary: batch size mismatch");
    }
    if (batch_size_1_a != batch_size_1_b) {
        TT_ASSERT(
            batch_size_1_a > batch_size_1_b and batch_size_1_b == 1,
            "ttnn::operations::binary::Binary: batch size mismatch");
    }
    if (height_a != height_b) {
        TT_ASSERT(height_a > height_b and height_b == 1, "ttnn::operations::binary::Binary: height mismatch");
    }
    if (width_a != width_b) {
        TT_ASSERT(width_a > width_b and width_b == 1, "ttnn::operations::binary::Binary: width mismatch");
    }
    validate_shape(input_shape_a);
    validate_shape(input_shape_b);

    TT_FATAL(
        input_tensor_a.device() == input_tensor_b.device(),
        "Operands to eltwise binary need to be on the same device!");
    TT_FATAL(
        (input_tensor_a.get_layout() == Layout::TILE && input_tensor_b.get_layout() == Layout::TILE),
        "Inputs to eltwise binary must be tilized");
    if (this->program_config.in_place) {
        TT_FATAL(input_tensor_a.memory_config().memory_layout == this->program_config.memory_config.memory_layout);
        TT_FATAL(input_tensor_a.memory_config().buffer_type == this->program_config.memory_config.buffer_type);
        TT_FATAL(input_tensor_a.get_dtype() == this->program_config.dtype);
    }
    if (input_tensor_a.memory_config().is_sharded()) {
        if (input_tensor_a.memory_config().memory_layout != TensorMemoryLayout::HEIGHT_SHARDED) {
            // If we aren't height sharded, we require all sharding schemes to match until we add blocked
            // reader/writers for width and block sharding
            TT_FATAL((input_tensor_b.memory_config().is_sharded()));
            TT_FATAL(input_tensor_a.shard_spec().value().grid.ranges().size() == 1);
        }
        if (input_tensor_b.memory_config().is_sharded()) {
            TT_FATAL(input_tensor_a.memory_config().memory_layout == input_tensor_b.memory_config().memory_layout);
            TT_FATAL(input_tensor_a.shard_spec().value() == input_tensor_b.shard_spec().value());
        }
        if (this->program_config.memory_config.is_sharded()) {
            TT_FATAL(input_tensor_a.memory_config().memory_layout == this->program_config.memory_config.memory_layout);
        } else {
            TT_FATAL(this->program_config.memory_config.memory_layout == TensorMemoryLayout::INTERLEAVED);
        }
    } else if (input_tensor_b.memory_config().is_sharded()) {
        TT_FATAL(input_tensor_b.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED);
        TT_FATAL(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);
        if (this->program_config.memory_config.is_sharded()) {
            TT_FATAL(input_tensor_b.memory_config().memory_layout == this->program_config.memory_config.memory_layout);
        } else {
            TT_FATAL(this->program_config.memory_config.memory_layout == TensorMemoryLayout::INTERLEAVED);
        }
    } else {
        TT_FATAL(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);
        TT_FATAL(input_tensor_b.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);
        if (this->program_config.memory_config.is_sharded()) {
            TT_FATAL(this->program_config.memory_config.memory_layout == TensorMemoryLayout::HEIGHT_SHARDED);
            uint32_t num_blocks = input_tensor_a.volume() / input_tensor_a.get_legacy_shape()[-1] / TILE_HEIGHT;
            auto core_grid = input_tensor_a.device()->compute_with_storage_grid_size();
            uint32_t num_cores = core_grid.x * core_grid.y;
            TT_FATAL(num_blocks < num_cores or num_blocks % num_cores == 0);

        } else {
            TT_FATAL(this->program_config.memory_config.memory_layout == TensorMemoryLayout::INTERLEAVED);
        }
    }
}

std::vector<tt::tt_metal::Shape> Binary::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    return {input_tensor_a.get_legacy_shape()};
}

std::vector<Tensor> Binary::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    if (this->program_config.in_place) {
        return {};
    }
    auto program_type = get_program_type(*this, input_tensors);

    if (program_type == BinaryProgramType::ElementWiseSingleCore or
        program_type == BinaryProgramType::ElementWiseMultiCore) {
        if (this->program_config.memory_config.is_sharded()) {
            ShardSpec shard_spec{CoreRangeSet({}), {0, 0}};
            if (input_tensor_a.memory_config().is_sharded()) {
                shard_spec = input_tensor_a.shard_spec().value();
            } else if (input_tensor_b.memory_config().is_sharded()) {
                shard_spec = input_tensor_b.shard_spec().value();
            } else {
                uint32_t num_blocks = input_tensor_a.volume() / input_tensor_a.get_legacy_shape()[-1] / TILE_HEIGHT;
                auto core_grid = input_tensor_a.device()->compute_with_storage_grid_size();
                uint32_t num_grid_cores = core_grid.x * core_grid.y;
                uint32_t target_num_cores = num_blocks < num_grid_cores ? num_blocks : num_grid_cores;
                shard_spec.grid = num_cores_to_corerange_set(target_num_cores, core_grid, true);
                shard_spec.shape = {num_blocks / target_num_cores * TILE_HEIGHT, input_tensor_a.get_legacy_shape()[-1]};
                shard_spec.orientation = ShardOrientation::ROW_MAJOR;
            }
            auto memory_config = this->program_config.memory_config;
            memory_config.shard_spec = shard_spec;
            return {create_sharded_device_tensor(
                this->compute_output_shapes(input_tensors).at(0),
                this->program_config.dtype,
                Layout::TILE,
                input_tensor_a.device(),
                memory_config)};
        }
    } else {
        if (this->program_config.memory_config.is_sharded()) {
            ShardSpec shard_spec{CoreRangeSet({}), {0, 0}};
            if (input_tensor_a.memory_config().is_sharded()) {
                // Derive output shard_spec based on input
                shard_spec = input_tensor_a.shard_spec().value();
            }
            auto memory_config = this->program_config.memory_config;
            memory_config.shard_spec = shard_spec;
            return {create_sharded_device_tensor(
                this->compute_output_shapes(input_tensors).at(0),
                input_tensor_a.get_dtype(),
                Layout::TILE,
                input_tensor_a.device(),
                memory_config)};
        }
    }
    return operation::generic_create_output_tensors(
        *this, input_tensors, this->program_config.dtype, Layout::TILE, this->program_config.memory_config);
}

operation::ProgramWithCallbacks Binary::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    const auto& output_tensor = this->program_config.in_place ? input_tensor_a : output_tensors.at(0);

    static std::map<BinaryOpType, tt::tt_metal::BcastOpMath> binary_op_type_to_bcast_op_math{
        {BinaryOpType::ADD, tt::tt_metal::BcastOpMath::ADD},
        {BinaryOpType::SUB, tt::tt_metal::BcastOpMath::SUB},
        {BinaryOpType::MUL, tt::tt_metal::BcastOpMath::MUL},
    };

    auto program_type = get_program_type(*this, input_tensors);
    switch (program_type) {
        case BinaryProgramType::ElementWiseMultiCore:
            return eltwise_binary_multi_core(
                input_tensor_a,
                input_tensor_b,
                output_tensor,
                this->program_config.op_type,
                this->program_config.fused_activations);
        case BinaryProgramType::ElementWiseSingleCore:
            return eltwise_binary_single_core(
                input_tensor_a,
                input_tensor_b,
                output_tensor,
                this->program_config.op_type,
                this->program_config.fused_activations);
        case BinaryProgramType::BroadcastHeightAndWidthMultiCore:
            return bcast_multi_core_hw(
                input_tensor_a,
                input_tensor_b,
                output_tensor,
                binary_op_type_to_bcast_op_math.at(this->program_config.op_type),
                tt::tt_metal::BcastOpDim::HW);
        case BinaryProgramType::BroadcastHeightMultiCore:
            return bcast_multi_core_h(
                input_tensor_a,
                input_tensor_b,
                output_tensor,
                binary_op_type_to_bcast_op_math.at(this->program_config.op_type),
                tt::tt_metal::BcastOpDim::H);
        case BinaryProgramType::BroadcastWidthMultiCore:
            return bcast_multi_core_w(
                input_tensor_a,
                input_tensor_b,
                output_tensor,
                binary_op_type_to_bcast_op_math.at(this->program_config.op_type),
                tt::tt_metal::BcastOpDim::W);
        default: TT_THROW("Invalid program type");
    }
}

const operation::Hash Binary::compute_program_hash(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    auto program_type = get_program_type(*this, input_tensors);
    operation::Hash hash = tt::stl::hash::hash_objects(
        0,
        typeid(*this).hash_code(),
        this->program_config,
        program_type,
        input_tensor_a.get_dtype(),
        input_tensor_a.memory_config(),
        input_tensor_b.get_dtype(),
        input_tensor_b.memory_config());
    return hash;
}

operation::OpPerformanceModel Binary::create_op_performance_model(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    // GS specific parameters
    // 80 B/cycle unpacker BW shared
    // 128 datums per cycle math, but unpacker cant keep up
    constexpr int num_cores = 9 * 12;

    int total_bytes = 0;
    for (const auto& t : input_tensors) {
        total_bytes += t.volume() * t.element_size();
    }
    int ideal_eltwise_cycles = total_bytes / 80 / num_cores;

    operation::OpPerformanceModel result(input_tensors, output_tensors, ideal_eltwise_cycles);
#if 0
        tt::log_info(tt::LogOp, "Binary PerfModel:");
        tt::log_info(tt::LogOp, "\t Data (Bytes): {}", total_bytes);
        tt::log_info(tt::LogOp, "\t ideal_eltwise_cycles: {}", ideal_eltwise_cycles);
#endif
    return result;
}

}  // namespace binary

}  // namespace operations

}  // namespace ttnn
