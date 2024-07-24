// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/ccl/all_gather/device/all_gather_op.hpp"
#include "ttnn/experimental/tt_dnn/op_library/math.hpp"

#include "tt_metal/host_api.hpp"

#include "tensor/tensor_utils.hpp"

#include "eth_l1_address_map.h"


#include "ttnn/operations/ccl/all_gather_matmul/device/all_gather_matmul_op.hpp"

/* All Gather Matmul fusion includes */
#include "ttnn/cpp/ttnn/operations/ccl/all_gather/device/all_gather_op.hpp"
#include "ttnn/cpp/ttnn/operations/matmul/device/matmul_op.hpp"
#include "ttnn/cpp/ttnn/operations/matmul/matmul.hpp"

namespace ttnn {

void AllGatherMatmul::validate(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<const ttnn::Tensor>>& optional_input_tensors) const {

    // All Gather validate
    this->all_gather_struct.validate({input_tensors.at(0)});

    // Matmul validate.
    this->matmul_struct.validate({input_tensors.at(1), input_tensors.at(2)}, optional_input_tensors);
}

std::vector<tt::tt_metal::Shape> AllGatherMatmul::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {

    // All Gather shape
    tt::tt_metal::Shape all_gather_output_shape = this->all_gather_struct.compute_output_shapes({input_tensors.at(0)}).at(0);

    // Matmul shape
    tt::tt_metal::Shape matmul_output_shapes = this->matmul_struct.compute_output_shapes({input_tensors.at(1), input_tensors.at(2)}).at(0);

    return {all_gather_output_shape, matmul_output_shapes};
}

std::vector<Tensor> AllGatherMatmul::create_output_tensors(const std::vector<Tensor> &input_tensors) const {

    // All Gather output tensor
    auto& all_gather_output_tensor = input_tensors.at(1); // this->all_gather_out_tensor = this->all_gather_struct.create_output_tensors(input_tensors).at(0);

    // Matmul output tensor
    ttnn::Tensor matmul_output_tensor = this->matmul_struct.create_output_tensors({input_tensors.at(1), input_tensors.at(2)}).at(0);

    return {all_gather_output_tensor, matmul_output_tensor};
}

operation::ProgramWithCallbacks AllGatherMatmul::create_program(const std::vector<Tensor> & input_tensors, const std::vector<std::optional<const ttnn::Tensor>>& optional_input_tensors, std::vector<Tensor> &output_tensors) const {

    // Return the AllGatherMatmul program with callbacks
    return all_gather_struct.create_program({input_tensors.at(0)}, output_tensors);

}

namespace operations {
namespace ccl {

std::vector <ttnn::Tensor> all_gather_matmul(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const uint32_t dim,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const bool transpose_a,
    const bool transpose_b,
    const std::optional<const DataType> dtype,
    const std::optional<const tt::operations::primary::MatmulProgramConfig> program_config,
    const std::optional<const std::string>& activation,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const ttnn::CoreGrid> core_grid
) {

    TT_FATAL(std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr, "This op is only supported for Fast Dispatch");

    auto devices = input_tensor.get_workers();
    std::vector<Tensor> output_tensors = {ttnn::Tensor(operation::get_workers_for_op_output({input_tensor, weight_tensor})),
                                            ttnn::Tensor(operation::get_workers_for_op_output({input_tensor, weight_tensor}))};
    std::vector<std::optional<const ttnn::Tensor>> optional_input_tensors = {std::nullopt};


    operation::launch_op(
        [dim, num_links, memory_config, transpose_a, transpose_b, dtype, program_config, activation, compute_kernel_config, core_grid, devices](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const ttnn::Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {

            const auto& input_tensor = input_tensors.at(0);
            const auto& weight_tensor = input_tensors.at(1);
            uint32_t num_devices = devices.size();

            /* All Gather stuff */
            uint32_t device_index = 0; // Initialize device index
            uint32_t receiver_device_id = 0; // Initialize receiver device ID
            uint32_t sender_device_id = 0; // Initialize sender device ID

            for (uint32_t i = 0; i < num_devices; ++i) {
                if (devices[i] == input_tensor.device()) {
                    device_index = i;
                    receiver_device_id = devices[(i + 1) % num_devices]->id(); // Next device in the ring
                    sender_device_id = devices[(i + num_devices - 1) % num_devices]->id(); // Previous device in the ring
                    break;
                }
            }

            ttnn::AllGather all_gather_struct{
                dim, num_links, num_devices, device_index, receiver_device_id, sender_device_id, memory_config.value_or(input_tensor.memory_config())};


            /* Matmul stuff */
            auto arch = input_tensor.device()->arch();
            const bool has_user_grid = core_grid.has_value();
            const bool has_program_config = program_config.has_value();
            const auto increase_fidelity = !has_program_config && !has_user_grid;
            auto math_fidelity = increase_fidelity ? MathFidelity::HiFi2 : MathFidelity::LoFi;
            auto kernel_config_val = init_device_compute_kernel_config(arch, compute_kernel_config, math_fidelity);
            bool broadcast_batch = get_broadcast_batch(input_tensor, weight_tensor, program_config);
            bool user_run_batched = ttnn::operations::matmul::detail::is_input_batched(weight_tensor.get_shape());
            TT_FATAL(!(has_user_grid && has_program_config), "Cannot use both user core grid/coordinates and a program config");
            std::optional<CoreCoord> user_core_coord;
            if (core_grid.has_value()) {
                user_core_coord = CoreCoord(core_grid->x, core_grid->y);
            }

            tt::operations::primary::Matmul matmul_struct{
                program_config, broadcast_batch, memory_config.value_or(input_tensor.memory_config()), dtype.value_or(input_tensor.get_dtype()), compute_kernel_config,
                /*untilize_out=*/false, user_core_coord, ttnn::operations::matmul::get_fused_activation(activation),
                user_run_batched, transpose_a, transpose_b
            };


            /* Create the dummy all gather output tensor used as input (activation) to the matmul */
            ttnn::Tensor all_gather_out_tensor = all_gather_struct.create_output_tensors({input_tensor}).at(0);

            return operation::run(
                ttnn::AllGatherMatmul{
                    /* All Gather Params */
                    all_gather_struct,
                    /* Matmul params */
                    matmul_struct},
                {input_tensor, all_gather_out_tensor, weight_tensor}, optional_input_tensors);
        },
        {input_tensor, weight_tensor}, output_tensors, optional_input_tensors);
    return {output_tensors.at(0), output_tensors.at(1)};
}


} // namespace ccl
} // namespace operations

}  // namespace ttnn
