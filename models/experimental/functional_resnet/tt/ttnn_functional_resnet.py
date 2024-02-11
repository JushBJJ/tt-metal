# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn


class BasicBlock:
    def __init__(
        self,
        parameters,
    ) -> None:
        self.conv1 = parameters.conv1
        self.conv2 = parameters.conv2
        if "downsample" in parameters:
            self.downsample = parameters.downsample
        else:
            self.downsample = None

    def __call__(self, x):
        identity = x

        out = self.conv1(x)
        # out = self.bn1(out)

        out = self.conv2(out)
        # out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # out = ttnn.add(out, identity, memory_config=ttnn.get_memory_config(out))
        # out = ttnn.to_memory_config(out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # out = self.relu(out)

        return out

    def torch_call(self, torch_input_tensor):
        input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
        input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16)

        input_tensor = self.conv1.copy_input_to_device(input_tensor)
        output_tensor = self(input_tensor)
        output_tensor = self.conv2.copy_output_from_device(output_tensor)

        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
        output_tensor = torch.reshape(output_tensor, torch_input_tensor.shape)
        output_tensor = output_tensor.to(torch_input_tensor.dtype)
        return output_tensor
