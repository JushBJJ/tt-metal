// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/detail/tt_metal.hpp"
#include "ttnn/types.hpp"
namespace ttnn {

namespace device {

using Device = ttnn::Device;

Device &open_device(int device_id, size_t l1_small_size = DEFAULT_L1_SMALL_SIZE, size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE);
void close_device(Device &device);
void enable_program_cache(Device &device);
void disable_and_clear_program_cache(Device &device);
bool is_wormhole_or_blackhole(tt::ARCH arch);

}  // namespace device

using namespace device;

}  // namespace ttnn
