// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/ccl/all_gather/device/all_gather_op.hpp"
#include "ttnn/experimental/tt_dnn/op_library/math.hpp"

#include "tt_metal/host_api.hpp"

#include "tensor/tensor_utils.hpp"

#include "eth_l1_address_map.h"


#include "ttnn/operations/ccl/all_gather_matmul/device/all_gather_matmul_op.hpp"


namespace ttnn {

void AllGather::validate(const std::vector<Tensor> &input_tensors) const {
    TT_FATAL(input_tensors.size() == 1);
    const auto& input_tensor = input_tensors[0];
    const auto& layout = input_tensors[0].get_layout();
    const auto& dtype = input_tensors[0].get_dtype();
    const auto& page_size = input_tensors[0].buffer()->page_size();
    TT_FATAL(page_size % input_tensors[0].buffer()->alignment() == 0, "All Gather currently requires aligned pages");

    // TODO: This can be removed by passing two page sizes, actual and aligned to be used for address offsets
    // Buffer sizes also need to take this aligned page size into consideration
    // TODO: Validate ring
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to all_gather need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr , "Operands to all_gather need to be allocated in buffers on device!");
    TT_FATAL(this->num_links > 0);
    TT_FATAL(this->num_links <= input_tensor.device()->compute_with_storage_grid_size().y, "Worker cores used by links are parallelizaed over rows");
    TT_FATAL(this->receiver_device_id.has_value() || this->sender_device_id.has_value());
    if (this->receiver_device_id == this->sender_device_id) {
        TT_FATAL(input_tensor.device()->get_ethernet_sockets(this->receiver_device_id.value()).size() >= 2 * this->num_links, "2 Device all gather requires at least 2 eth connections per link");
    } else {
        TT_FATAL(this->topology == all_gather_op::Topology::Linear || (this->receiver_device_id.has_value() && input_tensor.device()->get_ethernet_sockets(this->receiver_device_id.value()).size() >= this->num_links), "All gather requires at least 1 eth connection per link between sender device {} and receiver device {}", this->sender_device_id, this->receiver_device_id);
        TT_FATAL(this->topology == all_gather_op::Topology::Linear || (this->sender_device_id.has_value() &&input_tensor.device()->get_ethernet_sockets(this->sender_device_id.value()).size() >= this->num_links), "All gather requires at least 1 eth connection per link between sender device {} and receiver device {}", this->sender_device_id, this->receiver_device_id);
    }

    TT_FATAL(input_tensor.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED ||
        input_tensor.memory_config().memory_layout == TensorMemoryLayout::WIDTH_SHARDED ||
        input_tensor.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED);

    // Sharding Config checks
    bool input_sharded = input_tensor.is_sharded();
    if (input_sharded) {
        // TODO(snijjar)
    }
}

std::vector<tt::tt_metal::Shape> AllGather::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    auto shape = input_tensors[0].get_legacy_shape();
    shape[this->dim] *= this->ring_size;
    return std::vector<tt::tt_metal::Shape>(input_tensors.size(), shape);
}

std::vector<Tensor> AllGather::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors[0];
    if(this->output_mem_config.is_sharded()) {
        return {create_device_tensor(
            this->compute_output_shapes(input_tensors).at(0),
            input_tensor.get_dtype(),
            input_tensor.get_layout(),
            input_tensor.device(),
            this->output_mem_config
            )};
    } else {
        return operation::generic_create_output_tensors(*this, input_tensors, input_tensor.get_dtype(), input_tensor.get_layout(), this->output_mem_config);
    }
}

operation::ProgramWithCallbacks AllGather::create_program(const std::vector<Tensor> & input_tensors, std::vector<Tensor> &output_tensors) const {
    AllGatherMode all_gather_mode = choose_all_gather_mode(input_tensors.at(0), output_tensors.at(0), dim);
    switch (all_gather_mode) {
        case AllGatherMode::RING_INTERLEAVED:
        case AllGatherMode::SINGLE_TILE_HIGH_WIDTH_SHARDED:
            return all_gather_multi_core_with_workers(input_tensors[0], output_tensors[0], this->dim, this->num_links, this->ring_size, this->ring_index, this->receiver_device_id, this->sender_device_id, this->topology);
        break;
        case AllGatherMode::FULL_WORKER_GRID_SHARDED:
            TT_THROW("Unsupported AllGatherMode");
        break;
        default:
            TT_THROW("Unsupported AllGatherMode");
    };
}

namespace operations {
namespace ccl {

std::tuple <Tensor, Tensor> all_gather_matmul(
    const Tensor& input_tensor,
    const Tensor& weight_tensor,

    const uint32_t dim,
    const uint32_t num_links = 1,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,

    std::optional<const Tensor> bias = std::nullopt,

    const struct AllGatherMatmul& parametrs = AllGatherMatmul{}
) {

    TT_FATAL(std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr, "This op is only supported for Fast Dispatch");

    auto devices = input_tensor.get_workers();
    std::vector<Tensor> output_tensors;  // Should be size 2: One for all_gather and one for matmul

    std::vector<std::optional<const Tensor>> optional_input_tensors = {};

    if (bias.has_value()) {
        optional_input_tensors.push_back(bias.value());
        output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor, weight_tensor}, {bias.value()})), Tensor(operation::get_workers_for_op_output({input_tensor, weight_tensor}, {bias.value()}));
    } else {
        optional_input_tensors.push_back(std::nullopt);
        output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor, weight_tensor})), Tensor(operation::get_workers_for_op_output({input_tensor, weight_tensor}));
    }

    operation::launch_op(
        [dim, num_links, memory_config, parameters, devices](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors) mutable -> std::vector<Tensor> {

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

            /* Matmul stuff */
            auto arch = input_tensor.device()->arch();
            const bool has_user_grid = parameters.user_core_coord.has_value();
            const bool has_program_config = parameters.program_config.has_value();
            const auto increase_fidelity = !has_program_config && !has_user_grid;
            auto math_fidelity = increase_fidelity ? MathFidelity::HiFi2 : MathFidelity::LoFi;
            auto kernel_config_val = init_device_compute_kernel_config(arch, parameters.compute_kernel_config, math_fidelity);
            bool broadcast_batch = parameters.bcast_batch.value_or(get_broadcast_batch(input_tensor, weight_tensor, parameters.program_config));
            TT_FATAL(!(has_user_grid && has_program_config), "Cannot use both user core grid/coordinates and a program config");


            return operation::run(
                ttnn::AllGather{
                    /* All Gather Params */
                    dim, num_links, num_devices, device_index, receiver_device_id, sender_device_id, memory_config.value_or(input_tensor.memory_config()), all_gather_op::Topology::Ring,
                    /* Matmul params */
                    parameters.program_config, broadcast_batch, parameters.mm_output_dtype.value_or(input_tensor.get_dtype()),
                    kernel_config_val, parameters.untilize_out, parameters.user_core_coord, parameters.user_fused_activation, parameters.user_run_batched},
                {input_tensor, weight_tensor}, optional_input_tensors);
        },
        {input_tensor, weight_tensor}, output_tensors, optional_input_tensors);
    return {output_tensors.at(0), output_tensors.at(1)};
}


} // namespace ccl
} // namespace operations

}  // namespace ttnn
