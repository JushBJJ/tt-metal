# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch
import tt_lib
import ttnn

from models.demos.falcon7b.tt.falcon_lm_head import falcon_lm_head_matmul_2d
from models.demos.falcon7b.tt.falcon_model import TtFalconModelShared
from models.demos.falcon7b.tt.model_utils import get_falcon_default_core_grid, get_weights_cached
from models.utility_functions import torch_tensors_to_tt_tensors


class TtFalconCausalLM(TtFalconModelShared):
    def __init__(
        self,
        devices,
        state_dict,
        base_url,
        num_layers,
        config,
        max_position_embeddings,
        model_config,
        tt_cache_path,
        seq_len,
    ):
        assert base_url == "", "base_url should be empty at the root of the model!"
        super().__init__(
            devices=devices,
            state_dict=state_dict,
            base_url=f"transformer",
            num_layers=num_layers,
            config=config,
            max_position_embeddings=max_position_embeddings,
            model_config=model_config,
            tt_cache_path=tt_cache_path,
        )
        self.num_devices = len(devices)
        self.model_config = model_config
        self.seq_len = seq_len

        lm_head_weight = None
        if self.state_dict:
            lm_head_weight = self.state_dict["lm_head.weight"]
            lm_head_weight = torch.transpose(lm_head_weight, -2, -1)

        if self.model_config["PREFILL_OPTIMIZED_MODE"] and self.seq_len > 512:
            # Optimization for lm_head matmul
            num_slices = self.model_config["LM_HEAD_NUM_SLICES"][seq_len]
            if lm_head_weight is not None:
                PADDING = torch.zeros([64, lm_head_weight.shape[1] // num_slices])
                lm_head_weights = torch.chunk(lm_head_weight, num_slices, dim=-1)
                lm_head_weights_padded = [torch.cat([weight, PADDING], 0) for weight in lm_head_weights]
            # Cache sliced weights for lm_head with different seq_len
            self.lm_head_sliced_weights = [
                get_weights_cached(
                    devices,
                    model_config,
                    tt_cache_path,
                    f"lm_head.weight_slice_{i}_of_{num_slices}",
                    weight_config_str="LM_HEAD_MM_WEIGHTS",
                    weights_to_cache=lm_head_weights_padded[i] if lm_head_weight is not None else None,
                )
                for i in range(num_slices)
            ]
            # Generate padding for lm_head > 512
            padding = torch.zeros([1, 1, seq_len, 64])

            tt_paddings = torch_tensors_to_tt_tensors(
                [padding.detach().clone() for _ in range(self.num_devices)],
                tt_lib.tensor.Layout.TILE,
                self.model_config["LM_HEAD_MM_INPUT_DTYPE"],
                self.model_config["LM_HEAD_MM_INPUT_MEMCFG"],
                self.devices,
            )
            self.lm_head_padding = tt_paddings

        self.lm_head_weights = get_weights_cached(
            devices,
            model_config,
            tt_cache_path,
            f"lm_head.weight",
            weight_config_str="LM_HEAD_MM_WEIGHTS",
            weights_to_cache=lm_head_weight,
        )

    def forward(
        self,
        input_ids: tt_lib.tensor.Tensor,
        llm_mode: str,
        attention_mask: tt_lib.tensor.Tensor = None,
        user_id: int = 0,
        layer_past: Optional[Tuple[Tuple[tt_lib.tensor.Tensor]]] = None,
        layer_past_len: int = 0,
        use_cache: bool = False,
    ) -> tt_lib.tensor.Tensor:
        hidden_states, presents = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            llm_mode=llm_mode,
            user_id=user_id,
            layer_past=layer_past,
            layer_past_len=layer_past_len,
            use_cache=use_cache,
        )

        if llm_mode == "prefill":
            seq_len = hidden_states[0].get_legacy_shape()[-2]

            if self.model_config["PREFILL_OPTIMIZED_MODE"] and seq_len > 512:
                lm_logits = []
                for device_id in range(self.num_devices):
                    hidden_states[device_id] = tt_lib.tensor.concat(
                        [hidden_states[device_id], self.lm_head_padding[device_id]], -1
                    )

                    out_slices = []
                    for slice_id in range(self.model_config["LM_HEAD_NUM_SLICES"][seq_len]):
                        out_slices.append(
                            tt_lib.operations.primary.matmul(
                                hidden_states[device_id],
                                self.lm_head_sliced_weights[slice_id][device_id],
                                program_config=self.model_config["LM_HEAD_PROGCFG"][seq_len],
                                output_mem_config=self.model_config["LM_HEAD_MM_OUTPUT_MEMCFG"],
                                output_dtype=self.model_config["LM_HEAD_MM_OUTPUT_DTYPE"],
                                compute_kernel_config=self.model_config["LM_HEAD_KERNEL_CONFIG"],
                            )
                        )

                    out = tt_lib.tensor.concat(out_slices, -1)
                    lm_logits.append(out)

                    # Do we need this?
                    for i in range(self.model_config["LM_HEAD_NUM_SLICES"][seq_len]):
                        out_slices[i].deallocate(True)
            else:
                lm_logits = [
                    ttnn.matmul(
                        hidden_states[device_id],
                        self.lm_head_weights[device_id],
                        memory_config=self.model_config["LM_HEAD_MM_OUTPUT_MEMCFG"],
                        dtype=self.model_config["LM_HEAD_MM_OUTPUT_DTYPE"],
                        core_grid=get_falcon_default_core_grid(hidden_states[device_id].device()),
                        use_1d_systolic_array=True,
                        compute_kernel_config=self.model_config["LM_HEAD_KERNEL_CONFIG"],
                    )
                    for device_id in range(self.num_devices)
                ]
        else:
            lm_logits = [
                tt_lib.tensor.falcon_lm_head_matmul(
                    hidden_states[device_id],
                    self.lm_head_weights[device_id],
                    bias=None,
                    output_mem_config=self.model_config["LM_HEAD_MM_OUTPUT_MEMCFG"],
                    output_dtype=self.model_config["LM_HEAD_MM_OUTPUT_DTYPE"],
                )
                for device_id in range(self.num_devices)
            ]

        return lm_logits, presents
