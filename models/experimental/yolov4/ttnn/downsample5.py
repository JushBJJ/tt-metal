# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.experimental.yolov4.ttnn.common import Conv
from models.experimental.yolov4.reference.downsample5 import DownSample5
from tests.ttnn.utils_for_testing import assert_with_pcc
import pytest
import time


class Down5:
    def __init__(self, model) -> None:
        if type(model) is str:
            torch_model = torch.load(model)
        else:
            torch_model = model.torch_model
        self.torch_model = torch_model
        self.conv1 = Conv(
            torch_model, "down5.conv1", [1, 20, 20, 512], (2, 2, 1, 1), reshard=True, height_sharding=False
        )
        self.conv2 = Conv(torch_model, "down5.conv2", [1, 10, 10, 1024], (1, 1, 0, 0), reshard=True, deallocate=False)
        self.conv3 = Conv(torch_model, "down5.conv3", [1, 10, 10, 1024], (1, 1, 0, 0))

        self.res1_conv1 = Conv(
            torch_model, "down5.resblock.module_list.0.0", [1, 10, 10, 512], (1, 1, 0, 0), deallocate=False
        )
        self.res1_conv2 = Conv(torch_model, "down5.resblock.module_list.0.1", [1, 10, 10, 512], (1, 1, 1, 1))
        self.res2_conv1 = Conv(
            torch_model, "down5.resblock.module_list.1.0", [1, 10, 10, 512], (1, 1, 0, 0), deallocate=False
        )
        self.res2_conv2 = Conv(torch_model, "down5.resblock.module_list.1.1", [1, 10, 10, 512], (1, 1, 1, 1))
        self.res3_conv1 = Conv(
            torch_model, "down5.resblock.module_list.2.0", [1, 10, 10, 512], (1, 1, 0, 0), deallocate=False
        )
        self.res3_conv2 = Conv(torch_model, "down5.resblock.module_list.2.1", [1, 10, 10, 512], (1, 1, 1, 1))
        self.res4_conv1 = Conv(
            torch_model, "down5.resblock.module_list.3.0", [1, 10, 10, 512], (1, 1, 0, 0), deallocate=False
        )
        self.res4_conv2 = Conv(torch_model, "down5.resblock.module_list.3.1", [1, 10, 10, 512], (1, 1, 1, 1))

        self.conv4 = Conv(torch_model, "down5.conv4", [1, 10, 10, 512], (1, 1, 0, 0), reshard=True, deallocate=False)

        self.conv5 = Conv(torch_model, "down5.conv5", [1, 10, 10, 1024], (1, 1, 0, 0), height_sharding=False)

    def __call__(self, device, input_tensor):
        output_tensor_split = self.conv1(device, input_tensor)
        output_tensor_left = self.conv2(device, output_tensor_split)

        res1_split = self.conv3(device, output_tensor_split)

        output_tensor = self.res1_conv1(device, res1_split)
        output_tensor = self.res1_conv2(device, output_tensor)
        res2_split = res1_split + output_tensor
        ttnn.deallocate(res1_split)

        output_tensor = self.res2_conv1(device, res2_split)
        output_tensor = self.res2_conv2(device, output_tensor)
        res3_split = res2_split + output_tensor

        ttnn.deallocate(res2_split)

        output_tensor = self.res3_conv1(device, res3_split)
        output_tensor = self.res3_conv2(device, output_tensor)
        res4_split = res3_split + output_tensor

        ttnn.deallocate(res3_split)

        output_tensor = self.res4_conv1(device, res4_split)
        output_tensor = self.res4_conv2(device, output_tensor)
        output_tensor = res4_split + output_tensor

        ttnn.deallocate(res4_split)

        output_tensor = self.conv4(device, output_tensor)

        output_tensor = ttnn.experimental.tensor.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor_left = ttnn.experimental.tensor.sharded_to_interleaved(output_tensor_left, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.concat([output_tensor, output_tensor_left], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(output_tensor_left)

        output_tensor = self.conv5(device, output_tensor)
        return output_tensor

    def __str__(self) -> str:
        this_str = ""
        index = 1
        for conv in self.convs:
            this_str += str(index) + " " + str(conv)
            this_str += " \n"
            index += 1
        return this_str


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_down5(device):
    ttnn_model = Down5("tests/ttnn/integration_tests/yolov4/yolov4.pth")

    torch_input = torch.randn((1, 20, 20, 512), dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16)
    torch_input = torch_input.permute(0, 3, 1, 2).float()
    torch_model = DownSample5()

    new_state_dict = {}
    ds_state_dict = {k: v for k, v in ttnn_model.torch_model.items() if (k.startswith("down5."))}

    keys = [name for name, parameter in torch_model.state_dict().items()]
    values = [parameter for name, parameter in ds_state_dict.items()]
    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    result_ttnn = ttnn_model(device, ttnn_input)

    start_time = time.time()
    for x in range(2):
        result_ttnn = ttnn_model(device, ttnn_input)
    print(f"Time taken: {time.time() - start_time}")
    result = ttnn.to_torch(result_ttnn)
    ref = torch_model(torch_input)
    ref = ref.permute(0, 2, 3, 1)
    result = result.reshape(ref.shape)
    assert_with_pcc(result, ref, 0.99)
