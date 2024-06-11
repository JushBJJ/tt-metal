# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn

from models.demos.ttnn_falcon7b.tt.falcon_attention import TtFalconAttention
from models.demos.ttnn_falcon7b.tt.model_config import get_model_config, get_tt_cache_path
from models.demos.ttnn_falcon7b.tt.common import (
    create_custom_preprocessor,
    create_attention_mask,
    create_kv_cache,
    create_attention_input,
    create_position_ids,
    strip_state_dict_prefix,
)
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import assert_with_pcc
import transformers

from loguru import logger
from ttnn import ShardTensorToMesh, ReplicateTensorToMesh, ConcatMeshToTensor


PRETRAINED_MODEL_NAME = f"tiiuae/falcon-7b-instruct"


def get_model_prefix(layer_index: int = 0):
    return f"transformer.h.{layer_index}.self_attention"


@pytest.fixture(scope="module")
def torch_model():
    hugging_face_reference_model = transformers.FalconForCausalLM.from_pretrained(
        PRETRAINED_MODEL_NAME, low_cpu_mem_usage=True
    ).eval()
    state_dict = hugging_face_reference_model.state_dict()
    filtered_state_dict = strip_state_dict_prefix(state_dict, get_model_prefix())

    configuration = transformers.FalconConfig.from_pretrained(PRETRAINED_MODEL_NAME)
    torch_model = transformers.models.falcon.modeling_falcon.FalconAttention(configuration).eval()
    torch_model.load_state_dict(filtered_state_dict)
    return torch_model


@pytest.mark.parametrize(
    "llm_mode, device_batch_size, seq_len, kv_cache_len",
    (
        ("prefill", 1, 128, 0),
        ("decode", 32, 1, 128),
    ),
    ids=["prefill_seq128", "decode_batch32"],
)
@pytest.mark.parametrize(
    "model_name, expected_pcc",
    (("tiiuae/falcon-7b-instruct", 0.99),),
)
@pytest.mark.parametrize("model_config_str", ("BFLOAT16-DRAM", "BFLOAT16-L1"))
@pytest.mark.parametrize(
    "enable_async",
    [True, False],
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 72576}], indirect=True)
def test_falcon_attention(
    pcie_device_mesh,
    model_name,
    llm_mode,
    device_batch_size,
    seq_len,
    kv_cache_len,
    expected_pcc,
    model_config_str,
    torch_model,
    enable_async,
):
    for device in pcie_device_mesh.get_device_ids():
        pcie_device_mesh.get_device(device).enable_async(enable_async)
        pcie_device_mesh.get_device(device).enable_program_cache()

    torch.manual_seed(0)
    batch = device_batch_size * pcie_device_mesh.get_num_devices()
    if llm_mode == "decode":
        shard_dim = 2
        concat_dim = 1
    else:
        shard_dim = 0
        concat_dim = 0

    configuration = transformers.FalconConfig.from_pretrained(model_name)
    model_config = get_model_config(model_config_str)
    dtype = model_config["DEFAULT_DTYPE"]
    kv_len = seq_len if llm_mode == "prefill" else kv_cache_len + 1

    attention_input, tt_attention_input = create_attention_input(
        llm_mode,
        dtype,
        batch,
        seq_len,
        configuration.hidden_size,
        pcie_device_mesh,
        mesh_mapper=ShardTensorToMesh(pcie_device_mesh, dim=shard_dim),
    )
    position_ids = create_position_ids(llm_mode, kv_cache_len)
    attention_mask, tt_attention_mask = create_attention_mask(
        llm_mode,
        dtype,
        attention_input,
        batch,
        seq_len,
        configuration.num_attention_heads,
        kv_cache_len,
        pcie_device_mesh,
        mesh_mapper=ShardTensorToMesh(pcie_device_mesh, dim=shard_dim),
    )
    layer_past, tt_layer_past = create_kv_cache(
        llm_mode,
        dtype,
        batch,
        kv_cache_len,
        configuration,
        pcie_device_mesh,
        mesh_mapper=ShardTensorToMesh(pcie_device_mesh, dim=0),
    )

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        device=pcie_device_mesh,
        custom_preprocessor=create_custom_preprocessor(
            model_config,
            tt_cache_path=get_tt_cache_path(f"{model_name}"),
            device=pcie_device_mesh,
            base_file_name=get_model_prefix(),
            weights_mesh_mapper=ReplicateTensorToMesh(pcie_device_mesh),
        ),
    )
    tt_FalconAttention_model = TtFalconAttention(
        configuration.hidden_size,
        configuration.num_attention_heads,
        configuration.max_position_embeddings,
        model_config,
        parameters=parameters,
        core_grid=pcie_device_mesh.get_devices()[0].core_grid,
    )
    logger.info("Compiling Attention Module")
    tt_FalconAttention_model(
        tt_attention_input,
        alibi=None,
        attention_mask=tt_attention_mask,
        llm_mode=llm_mode,
        user_id=0,
        layer_past=tt_layer_past,
        layer_past_len=kv_cache_len,
        use_cache=True,
    )
    logger.info("Capturing Trace")
    trace_id = ttnn.begin_trace_capture(pcie_device_mesh, cq_id=0)
    tt_out, tt_layer_present = tt_FalconAttention_model(
        tt_attention_input,
        alibi=None,
        attention_mask=tt_attention_mask,
        llm_mode=llm_mode,
        user_id=0,
        layer_past=tt_layer_past,
        layer_past_len=kv_cache_len,
        use_cache=True,
    )
    ttnn.end_trace_capture(pcie_device_mesh, trace_id, cq_id=0)
    logger.info("Executing Trace")
    for i in range(50):
        attention_input, tt_attention_input_updated = create_attention_input(
            llm_mode,
            dtype,
            batch,
            seq_len,
            configuration.hidden_size,
            None,
            mesh_mapper=ShardTensorToMesh(pcie_device_mesh, dim=shard_dim),
        )
        pytorch_out, pytorch_layer_present = torch_model(
            attention_input,
            alibi=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            layer_past=layer_past,
            use_cache=True,
        )
        ttnn.copy_host_to_device_tensor(tt_attention_input_updated, tt_attention_input)
        ttnn.execute_trace(pcie_device_mesh, trace_id, cq_id=0)
        tt_out_host = ttnn.to_torch(tt_out, mesh_composer=ConcatMeshToTensor(pcie_device_mesh, dim=concat_dim)).squeeze(
            1
        )

        tt_layer_present_host = (
            ttnn.to_torch(tt_layer_present[0], mesh_composer=ConcatMeshToTensor(pcie_device_mesh, dim=0)).squeeze(1),
            ttnn.to_torch(tt_layer_present[1], mesh_composer=ConcatMeshToTensor(pcie_device_mesh, dim=0)).squeeze(1),
        )

        if llm_mode == "decode":
            tt_out_host = tt_out_host.transpose(0, 1)
        tt_layer_present_host = (
            tt_layer_present_host[0][:, :kv_len, :],
            tt_layer_present_host[1][:, :kv_len, :],
        )

        passed, pcc = assert_with_pcc(pytorch_out, tt_out_host.to(pytorch_out.dtype), expected_pcc)
        logger.success(f"Passed: pcc: {pcc}, expected: {expected_pcc}")
        assert_with_pcc(
            pytorch_layer_present[0].squeeze(1),
            tt_layer_present_host[0].to(pytorch_layer_present[0].dtype),
            expected_pcc,
        )
        assert_with_pcc(
            pytorch_layer_present[1].squeeze(1),
            tt_layer_present_host[1].to(pytorch_layer_present[1].dtype),
            expected_pcc,
        )

    for device in pcie_device_mesh.get_device_ids():
        pcie_device_mesh.get_device(device).enable_async(False)
