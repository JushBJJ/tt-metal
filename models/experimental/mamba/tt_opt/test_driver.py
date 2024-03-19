# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List
import sys
import ttnn
import torch
import model_config


from models.experimental.mamba.reference.decode_model import MambaPretrainedModelName


def get_cpu_reference_model():
    from models.experimental.mamba.reference.decode_model import MambaDecode

    return MambaDecode.from_pretrained("state-spaces/mamba-370m")


def get_tt_metal_model(num_users, hidden_size, configs):
    from models.experimental.mamba.tt_opt.full_model import MambaTT

    device_id = 0
    device = ttnn.open_device(device_id=device_id)

    torch.manual_seed(0)
    ttnn.enable_program_cache()

    reference_model = get_cpu_reference_model()
    cache_path = f"/tmp/state-spaces/mamba-370m"
    
    model = MambaTT(reference_model, cache_path, 1, device, num_users, hidden_size, configs)
    return model, device


def run_demo(num_users, hidden_size):
    configs = model_config.create_model_config(num_users, hidden_size)
    model, device = get_tt_metal_model(num_users, hidden_size, configs)

    # evaluate model:
    model.eval()

    with torch.no_grad():
        # create random torch tensor of hidden size and num_users, with datatype bfloat16

        input_data = torch.randn((1, 1, num_users, hidden_size), dtype=torch.bfloat16)

        input_data = ttnn.to_device(
            ttnn.from_torch(input_data, layout=ttnn.TILE_LAYOUT), device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )

        out_data = model(input_data)

    ttnn.close_device(device)


def main():
    num_users = int(sys.argv[1])
    hidden_size = int(sys.argv[2])
    assert num_users == 32
    assert (hidden_size // 8) % 32 == 0
    run_demo(num_users, hidden_size)


if __name__ == "__main__":
    main()
