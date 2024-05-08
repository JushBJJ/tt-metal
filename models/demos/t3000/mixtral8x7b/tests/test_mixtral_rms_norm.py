# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import os
import pytest
from loguru import logger
import ttnn
from models.demos.t3000.mixtral8x7b.tt.model_config import TtModelArgs
from models.demos.t3000.mixtral8x7b.tt.mixtral_rms_norm import TtRMSNormSharded
from models.demos.t3000.mixtral8x7b.reference.model import RMSNorm
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from ttnn import ReplicateTensorToMesh, ConcatMeshToTensor

# Set Mixtral flags for CI, if CI environment is setup
if os.getenv("CI") == "true":
    os.environ["MIXTRAL_CKPT_DIR"] = "/mnt/MLPerf/tt_dnn-models/Mistral/Mixtral-8x7B-v0.1/"
    os.environ["MIXTRAL_TOKENIZER_PATH"] = "/mnt/MLPerf/tt_dnn-models/Mistral/Mixtral-8x7B-v0.1/"
    os.environ["MIXTRAL_CACHE_PATH"] = "/mnt/MLPerf/tt_dnn-models/Mistral/Mixtral-8x7B-v0.1/"


def test_mistral_rms_norm_inference(device_mesh, reset_seeds):
    dtype = ttnn.bfloat8_b

    model_args = TtModelArgs(device_mesh.get_device(0))
    state_dict = torch.load(model_args.consolidated_weights_path(0))

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    partial_state_dict = {k[24:]: v for k, v in state_dict.items() if (k.startswith("layers.0.attention_norm."))}
    reference_model = RMSNorm(dim=model_args.dim)
    reference_model.load_state_dict(partial_state_dict)

    tt_model = TtRMSNormSharded(
        device_mesh=device_mesh,
        state_dict=state_dict,
        args=model_args,
        dtype=dtype,
        layer_num=4,
        weight_key="attention_norm",
    )
    input = torch.rand(1, 1, 32, 4096)
    reference_output = reference_model(input)[0]

    tt_input = ttnn.from_torch(
        input, device=device_mesh, dtype=dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=ReplicateTensorToMesh(device_mesh)
    )
    tt_input = ttnn.to_device(tt_input, device_mesh)
    tt_output = tt_model(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ConcatMeshToTensor(device_mesh, dim=0))[0]
    print(tt_output_torch.shape, reference_output.shape)
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(pcc_message)

    if passing:
        logger.info("Mixtral_rms_norm Passed!")
    else:
        logger.warning("Mixtral_rms_norm Failed!")

    assert passing, f"Mixtral_rms_norm output does not meet PCC requirement {0.99}."
