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
from python_api_testing.models.t5.t5_stack import TtT5Stack



def run_test_T5Stack_inference(device):
    hf_reference_model = T5Model.from_pretrained("t5-small")
    hf_reference_model.eval()

    config = json.loads(hf_reference_model.config.to_json_string())
    config["is_decoder"] = False
    config["use_cache"] = False

    if config["is_decoder"]:
        hf_reference_module = hf_reference_model.decoder
        base_address = f"decoder"
    else:
        hf_reference_module = hf_reference_model.encoder
        base_address = f"encoder"

    # Prepare input
    torch.manual_seed(0)
    test_input = (torch.rand(32, 128, 512) * 2) - 1

    # PyTorch output
    pt_out = hf_reference_module(inputs_embeds=test_input)
    pt_out = pt_out.last_hidden_state
    pt_out = pt_out.unsqueeze(0)

    # Move test input to Tt device test_input
    test_input = test_input.unsqueeze(0)
    test_input = torch2tt_tensor(test_input, device)

    # T5-small config file: https://huggingface.co/t5-small/resolve/main/config.json
    tt_model = TtT5Stack(config, hf_reference_model.state_dict(), base_address, device)
    tt_model_outputs = tt_model(inputs_embeds=test_input)
    last_hidden_state = tt_model_outputs[0]
    tt_out = tt2torch_tensor(last_hidden_state)

    print(pt_out[0, 0, 1:10, 1:10])
    print(tt_out[0, 0, 1:10, 1:10])

    print_diff_argmax(pt_out, tt_out)
    does_pass, pcc_message = comp_pcc(pt_out, tt_out, 0.98)

    print(comp_allclose(pt_out, tt_out))
    print(pcc_message)

    assert does_pass

    if does_pass:
        logger.info("test_T5Stack_inference Passed!")
    else:
        logger.warning("test_T5Stack_inference Failed!")


def test_T5Stack_inference():
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    run_test_T5Stack_inference(device)
    tt_lib.device.CloseDevice(device)


if __name__ == "__main__":
    test_T5Stack_inference()
