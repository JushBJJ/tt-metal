import dataclasses
from typing import List

import torch
import torch.nn.functional as F
from torch import nn


@dataclasses.dataclass
class MoeArgs:
    num_experts: int
    num_experts_per_tok: int


class MoeLayer(nn.Module):
    def __init__(self, experts: List[nn.Module], gate: nn.Module, moe_args: MoeArgs):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.args = moe_args

    def forward(self, inputs: torch.Tensor):
        org_input_shape = inputs.shape
        inputs = inputs.view(-1, inputs.shape[-1])
        gate_logits = self.gate(inputs)
        weights, selected_experts = torch.topk(gate_logits, self.args.num_experts_per_tok)
        weights = F.softmax(weights, dim=-1, dtype=torch.float).to(inputs.dtype)
        results = torch.zeros_like(inputs)
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            expert_ouput = expert(inputs[batch_idx])
            results[batch_idx] += weights[batch_idx, nth_expert, None] * expert_ouput
        return results.view(org_input_shape)
