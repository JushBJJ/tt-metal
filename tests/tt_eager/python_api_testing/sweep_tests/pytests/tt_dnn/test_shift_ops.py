# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from functools import partial
from math import pi
import copy

from tests.tt_eager.python_api_testing.sweep_tests import (
    comparison_funcs,
    generation_funcs,
)
from tests.tt_eager.python_api_testing.sweep_tests.run_pytorch_ci_tests import (
    run_single_pytorch_test,
)

shapes = [
    [[1, 1, 1, 30], [1, 1, 1, 30]],  # Single core
    [[1, 1, 300, 380], [1, 1, 300, 380]],  # multi core
    [[1, 3, 320, 380], [1, 3, 320, 380]],  # multi core
    [[1, 1, 32, 32], [1, 1, 32, 32]],  # Single core
    [[1, 1, 320, 384], [1, 1, 320, 384]],  # Multi core
    [[1, 3, 320, 384], [1, 3, 320, 384]],  # Multi core
]
input_mem_cfgs = copy.copy(generation_funcs.supported_mem_configs)
output_mem_cfgs = copy.copy(generation_funcs.supported_mem_configs)


@pytest.mark.parametrize(
    "input_shapes",
    shapes,
)
@pytest.mark.parametrize("input_mem_config", input_mem_cfgs)
@pytest.mark.parametrize("output_mem_config", output_mem_cfgs)
class TestShiftVariants:
    @pytest.mark.parametrize("shift_kind", ["left"])
    def test_all_shift_ops(
        self,
        input_shapes,
        shift_kind,
        device,
        function_level_defaults,
        input_mem_config,
        output_mem_config,
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-100, high=100), torch.int64)
        ] + [generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=1, high=50), torch.int64)]
        comparison_func = comparison_funcs.comp_pcc
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
        test_args.update(
            {
                "input_mem_config": [input_mem_config],
                "output_mem_config": output_mem_config,
            }
        )
        run_single_pytorch_test(
            f"eltwise-{shift_kind}-shift",
            input_shapes,
            datagen_func,
            comparison_func,
            device,
        )
