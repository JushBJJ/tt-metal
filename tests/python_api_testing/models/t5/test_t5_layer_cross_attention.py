from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import torch
import json
import tt_lib
from loguru import logger

from transformers import T5Model
from utility_functions import comp_allclose, comp_pcc
from tt_lib.utils import print_diff_argmax

from python_api_testing.models.t5.t5_utils import torch2tt_tensor, tt2torch_tensor
from python_api_testing.models.t5.t5_layer_cross_attention import TtT5LayerCrossAttention


def run_test_T5LayerCrossAttention_inference(device):
    hf_reference_model = T5Model.from_pretrained("t5-small")
    hf_reference_model.eval()

    config = json.loads(hf_reference_model.config.to_json_string())
    config["is_decoder"] = False

    # Cross attention can be only decoder
    hf_reference_module = hf_reference_model.decoder.block[0].layer[1]
    base_address = f"decoder.block.0.layer.1"

    # Cross attention is only in decoder part
    config["is_decoder"] = True

    # Prepare input
    torch.manual_seed(0)
    test_input = (torch.rand(32, 128, 512) * 2) - 1
    key_value_states = (torch.rand(32, 128, 512) * 2) - 1

    # PyTorch output
    pt_out = hf_reference_module(test_input, key_value_states)[0].unsqueeze(0)

    test_input = test_input.unsqueeze(0)
    key_value_states = key_value_states.unsqueeze(0)

    # T5-small config file: https://huggingface.co/t5-small/resolve/main/config.json
    tt_model = TtT5LayerCrossAttention(config, hf_reference_model.state_dict(), base_address, device)
    tt_out = tt_model(torch2tt_tensor(test_input, device), torch2tt_tensor(key_value_states, device))[0]
    tt_out = tt2torch_tensor(tt_out)

    print(pt_out[0, 0, 1:10, 1:10])
    print(tt_out[0, 0, 1:10, 1:10])

    print_diff_argmax(pt_out, tt_out)
    does_pass, pcc_message = comp_pcc(pt_out, tt_out, 0.98)

    print(comp_allclose(pt_out, tt_out))
    print(pcc_message)

    assert does_pass

    if does_pass:
        logger.info("test_T5LayerCrossAttention_inference Passed!")
    else:
        logger.warning("test_T5LayerCrossAttention_inference Failed!")


def test_T5LayerCrossAttention_inference():
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    run_test_T5LayerCrossAttention_inference(device)
    tt_lib.device.CloseDevice(device)


if __name__ == "__main__":
    test_T5LayerCrossAttention_inference()
