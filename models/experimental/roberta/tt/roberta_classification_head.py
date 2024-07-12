# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn

import tt_lib
import ttnn

from models.helper_funcs import Linear as TTLinear
from models.utility_functions import tt2torch_tensor, pad_by_zero
from models.experimental.roberta.roberta_common import torch2tt_tensor


class TtRobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, state_dict, base_address, device):
        super().__init__()
        self.device = device
        self.mem_config = tt_lib.tensor.MemoryConfig(
            tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.L1
        )

        self.dense_weight = pad_by_zero(state_dict[f"{base_address}.dense.weight"], self.device)[0]
        self.dense_bias = pad_by_zero(state_dict[f"{base_address}.dense.bias"], self.device)[0]

        # TODO: Add when supporting training
        # classifier_dropout = (
        #     config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        # )
        # self.dropout = nn.Dropout(classifier_dropout)

        self.out_proj_weight = torch2tt_tensor(state_dict[f"{base_address}.out_proj.weight"], self.device)
        self.out_proj_bias = torch2tt_tensor(state_dict[f"{base_address}.out_proj.bias"], self.device)
        self.dense_linear = TTLinear(
            self.dense_weight.get_legacy_shape()[-1],
            self.dense_weight.get_legacy_shape()[-2],
            self.dense_weight,
            self.dense_bias,
        )
        self.out_proj_linear = TTLinear(
            self.out_proj_weight.get_legacy_shape()[-1],
            self.out_proj_weight.get_legacy_shape()[-2],
            self.out_proj_weight,
            self.out_proj_bias,
        )

    def linear(self, x, weight, bias):
        weight = tt_lib.tensor.transpose(weight, -2, -1)
        x = ttnn.matmul(x, weight, memory_config=self.mem_config)
        x = tt_lib.tensor.bcast(
            x,
            bias,
            tt_lib.tensor.BcastOpMath.ADD,
            tt_lib.tensor.BcastOpDim.H,
            output_mem_config=self.mem_config,
        )
        return x

    def forward(self, features, **kwargs):
        torch_features = tt2torch_tensor(features).squeeze(0)
        torch_x = torch_features[:, 0, :]  # take <s> token (equiv. to [CLS])

        x = torch2tt_tensor(torch_x, self.device)
        # x = self.dropout(x)
        x = self.dense_linear(x)
        x = ttnn.tanh(x, memory_config=self.mem_config)
        # x = torch.tanh(x)

        # x = self.dropout(x)
        x = self.out_proj_linear(x)

        return x
