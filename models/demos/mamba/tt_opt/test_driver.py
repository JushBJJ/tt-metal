# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List
import sys
import ttnn
import torch


from models.demos.mamba.reference.decode_model import MambaPretrainedModelName


def get_cpu_reference_model():
    from models.demos.mamba.reference.decode_model import MambaDecode

    return MambaDecode.from_pretrained("state-spaces/mamba-370m")


def get_tt_metal_model():
    #import tt_lib
    from models.demos.mamba.tt_opt.full_model import MambaTT

    #device = tt_lib.device.CreateDevice(0)
    #tt_lib.device.SetDefaultDevice(device)
    #device = tt_lib.device.GetDefaultDevice()

    device_id = 0
    device = ttnn.open_device(device_id=device_id)

    torch.manual_seed(0)
    ttnn.enable_program_cache()

    reference_model = get_cpu_reference_model()
    model = MambaTT(reference_model, 1, device)
    return model, device








def run_demo(num_users, hidden_size):

    model, device = get_tt_metal_model()


    # evaluate model:
    model.eval()

    with torch.no_grad():
        #create random torch tensor of hidden size and num_users, with datatype bfloat16
        '''
        input_data = torch.randn((1, 1, num_users, hidden_size), dtype=torch.bfloat16)
        
        cfg = ttnn.create_sharded_memory_config(shape=(1,1,num_users,hidden_size), core_grid=ttnn.CoreGrid(y=num_users//32, x=8), strategy=ttnn.ShardStrategy.WIDTH)
        
        input_data = ttnn.to_device(ttnn.from_torch(input_data), layout=ttnn.TILE_LAYOUT, device=device, memory_config=cfg)
        '''
        
        
        out_data = model(None)

    ttnn.close_device(device)


def main():
    num_users = int(sys.argv[1])
    hidden_size = int(sys.argv[2])
    assert num_users == 32
    assert (hidden_size // 8) % 32 == 0
    run_demo(num_users, hidden_size)


if __name__ == "__main__":
    main()
