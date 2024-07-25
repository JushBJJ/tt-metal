// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/prefix_scan.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::experimental::ssm {

struct ExecutePrefixScan {
    static ttnn::Tensor execute_on_worker_thread(
        const Tensor& a,
        const Tensor& bx,
        const Tensor& h_prev,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<DataType> dtype = std::nullopt,
        const std::optional<MathFidelity> math_fidelity = std::nullopt) {
        auto program = PrefixScan{
            memory_config.value_or(a.memory_config()),
            dtype.value_or(a.dtype()),
            math_fidelity.value_or(MathFidelity::HiFi4)};
        return operation::run(program, {a, bx, h_prev}).at(0);
    }
};

}  // namespace ttnn::operations::experimental::ssm

namespace ttnn {

constexpr auto prefix_scan = ttnn::
    register_operation<"ttnn::experimental::ssm::prefix_scan", operations::experimental::ssm::ExecutePrefixScan>();

}  // namespace ttnn
