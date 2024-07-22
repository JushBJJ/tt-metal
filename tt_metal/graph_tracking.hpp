// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <string_view>

#include "tt_metal/common/core_coord.h"
#include "tt_metal/impl/buffers/buffer.hpp"

namespace tt::tt_metal {
    class GraphTracker {
    public:
        static GraphTracker& instance() {
            static GraphTracker tracker;
            return tracker;
        }
        void track_allocate(Buffer* buffer, uint64_t size, bool bottom_up) {
            auto alloc_id = reinterpret_cast<std::uintptr_t>(buffer);
            tt::log_info("Called Allocate id: {}, size: {}, bottom_up: {}", alloc_id, size, bottom_up);
        }

        void track_deallocate(Buffer* buffer) {
            auto alloc_id = reinterpret_cast<std::uintptr_t>(buffer);
            tt::log_info("Called Deallocate id: {}", alloc_id);
        }

        void track_allocate_cb(const CoreRange &core_range, uint64_t addr, uint64_t size) {
            tt::log_info( "Called allocate circular buffer rangeX: {}:{}, rangeY: {}:{} , addr: {}, size: {}",core_range.start.x, core_range.end.x, core_range.start.y, core_range.end.y, addr, size);
        }

        template<class ReturnType, class... Ts>
        void track_begin_op(std::string_view function_name) {
            tt::log_info( "Called Begin Op:{}", function_name);
        }

        void track_end_op() {
            tt::log_info( "Called End Op");
            tt::log_info( "CB deallocated here?");
        }
    private:
        GraphTracker() = default;
        ~GraphTracker() = default;
    };
}
