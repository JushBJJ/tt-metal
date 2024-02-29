// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/impl/dispatch/kernels/cq_cmds.hpp"

extern bool debug_g;

inline void generate_random_payload(vector<uint32_t>& cmds,
                                    uint32_t length) {

    for (uint32_t i = 0; i < length; i++) {
        uint32_t datum = std::rand();
        cmds.push_back(datum);
    }
}

inline void generate_random_payload(vector<uint32_t>& cmds,
                                    vector<uint32_t>& data,
                                    uint32_t length) {

    for (uint32_t i = 0; i < length; i++) {
        uint32_t datum = std::rand();
        cmds.push_back(datum);
        data.push_back(datum);
    }
}

inline void add_bare_dispatcher_cmd(vector<uint32_t>& cmds,
                                    CQDispatchCmd cmd) {

    uint32_t *ptr = (uint32_t *)&cmd;
    for (int i = 0; i < sizeof(CQDispatchCmd) / sizeof(uint32_t); i++) {
        cmds.push_back(*ptr++);
    }
}

inline void add_dispatcher_cmd(vector<uint32_t>& cmds,
                               vector<uint32_t>& worker_data,
                               CQDispatchCmd cmd,
                               uint32_t length) {

    auto prior_end = cmds.size();

    if (debug_g) {
        CQDispatchCmd debug_cmd;
        debug_cmd.base.cmd_id = CQ_DISPATCH_CMD_DEBUG;
        add_bare_dispatcher_cmd(cmds, debug_cmd);
    }

    add_bare_dispatcher_cmd(cmds, cmd);
    uint32_t length_words = length / sizeof(uint32_t);
    generate_random_payload(cmds, worker_data, length_words);

    if (debug_g) {
        CQDispatchCmd* debug_cmd_ptr;
        debug_cmd_ptr = (CQDispatchCmd *)&cmds[prior_end];
        debug_cmd_ptr->debug.size = (cmds.size() - prior_end) * sizeof(uint32_t) - sizeof(CQDispatchCmd);
        debug_cmd_ptr->debug.stride = sizeof(CQDispatchCmd);
        uint32_t checksum = 0;
        for (uint32_t i = prior_end + sizeof(CQDispatchCmd) / sizeof(uint32_t); i < cmds.size(); i++) {
            checksum += cmds[i];
        }
        debug_cmd_ptr->debug.checksum = checksum;
    }
}

inline void gen_dispatcher_write_cmd(vector<uint32_t>& cmds,
                                     vector<uint32_t>& worker_data,
                                     CoreCoord worker_core,
                                     uint32_t dst_addr,
                                     uint32_t length) {

    CQDispatchCmd cmd;

    cmd.base.cmd_id = CQ_DISPATCH_CMD_WRITE;
    cmd.base.flags = 0;
    cmd.write.dst_noc_addr = worker_core.x | (worker_core.y << 6);
    cmd.write.dst_addr = dst_addr;
    cmd.write.length = length;

    add_dispatcher_cmd(cmds, worker_data, cmd, length);
}

inline void gen_dispatcher_terminate_cmd(vector<uint32_t>& cmds) {

    vector<uint32_t> dummy;
    CQDispatchCmd cmd;
    cmd.base.cmd_id = CQ_DISPATCH_CMD_TERMINATE;
    add_dispatcher_cmd(cmds, dummy, cmd, 0);
}

inline bool validate_results(Device *device, CoreCoord phys_worker_core, const vector<uint32_t>& worker_data, uint64_t l1_buf_base) {

    log_info(tt::LogTest, "Validating {} bytes\n", worker_data.size() * sizeof(uint32_t));
    vector<uint32_t> results =
        tt::llrt::read_hex_vec_from_core(device->id(), phys_worker_core, l1_buf_base, worker_data.size() * sizeof(uint32_t));

    int fail_count = 0;

    for (int i = 0; i < worker_data.size(); i++) {
        if (results[i] != worker_data[i]) {
            if (fail_count == 0) {
                tt::log_fatal("Data mismatch, first 20 failures:\n");
                fprintf(stderr, "[idx] expected->read\n");
            }
            fprintf(stderr, "[%02d] 0x%08x->0x%08x\n", i, (unsigned int)worker_data[i], (unsigned int)results[i]);
            fail_count++;
            if (fail_count > 20) {
                break;
            }
        }
    }

    return fail_count == 0;
}
