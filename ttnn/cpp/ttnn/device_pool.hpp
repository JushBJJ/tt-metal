// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/detail/tt_metal.hpp"
#include "types.hpp"

namespace ttnn::device {

using Device = tt::tt_metal::Device;

namespace device_pool {

extern std::vector<Device *> _devices;

} // namespace device_pool

} // namespace ttnn::device
