// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api.h"
#include "tools/profiler/kernel_profiler.hpp"

namespace NAMESPACE {
    void MAIN {
        for (int i = 0; i < LOOP_COUNT; i ++)
        {
            DeviceZoneScopedN("TEST_FULL_BUFFER_COMPUTE");
            {
                DeviceZoneScopedN("TEST_FULL_BUFFER_COMPUTE");
                {
                    DeviceZoneScopedN("TEST_FULL_BUFFER_COMPUTE");
                    {
                        DeviceZoneScopedN("TEST_FULL_BUFFER_COMPUTE");
                        {
                            DeviceZoneScopedN("TEST_FULL_BUFFER_COMPUTE");
                            {
                                DeviceZoneScopedN("TEST_FULL_BUFFER_COMPUTE");
                                {
                                    DeviceZoneScopedN("TEST_FULL_BUFFER_COMPUTE");
                                }
                            }
                        }
                    }
                }
            }
        }
    }
} // NAMESPACE
