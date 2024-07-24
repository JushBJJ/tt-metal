// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
///
#include <algorithm>

#include "tt_metal/common/core_coord.h"
#include "eth_l1_address_map.h"
#include "impl/buffers/buffer.hpp"
#include "tensor/tensor_impl.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/all_gather/device/all_gather_op.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/experimental/tt_dnn/op_library/math.hpp"
#include "ttnn/experimental/tt_dnn/op_library/work_split.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"
#include <sstream>
#include <type_traits>

#include "ttnn/cpp/ttnn/operations/ccl/all_gather_matmul/device/all_gather_matmul_op.hpp"


using namespace tt::constants;

namespace ttnn {

using namespace ccl;

// For ring all-gather, we can send sub-sections of input tensor in opposite directions
// For linear all-gather though, we must ensure we send full tensors in BOTH directions
//   (in other words, disable the "bidirectional" send flag)
operation::ProgramWithCallbacks all_gather_matmul_multi_core_with_workers(const Tensor& input_tensor, Tensor& output_tensor, const uint32_t dim, const uint32_t num_links, const uint32_t ring_size, const uint32_t ring_index, const std::optional<chip_id_t> receiver_device_id, const std::optional<chip_id_t> sender_device_id, all_gather_op::Topology topology, const CoreCoord core_grid_offset) {

    tt::tt_metal::Program program{};
    return all_gather_multi_core_with_workers_helper(program, input_tensor, output_tensor, dim, num_links, ring_size, ring_index, receiver_device_id, sender_device_id, topology, core_grid_offset);
}

}  // namespace ttnn
