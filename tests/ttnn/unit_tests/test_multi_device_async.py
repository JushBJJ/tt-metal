# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import typing
import pytest
import ttnn
from loguru import logger
from tests.ttnn.utils_for_testing import assert_with_pcc
import transformers


#######
# Multi-Device Tensor tests running in async mode
#######


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
def test_ttnn_to_and_from_multi_device_shard(pcie_device_mesh, layout, memory_config):
    """Shard a tensor across devices, compose it back and verify loopback tensor is same as the original tensor"""
    from ttnn import ShardTensorToMesh, ConcatMeshToTensor

    for device in pcie_device_mesh.get_device_ids():
        pcie_device_mesh.get_device(device).enable_async(True)

    for i in range(100):
        torch_tensor = torch.rand((1, 1, 256, 512), dtype=torch.bfloat16)
        ttnn_tensor = ttnn.from_torch(torch_tensor, mesh_mapper=ShardTensorToMesh(pcie_device_mesh, dim=3))
        ttnn_tensor = ttnn.to_layout(ttnn_tensor, layout=layout)
        ttnn_tensor = ttnn.to_device(ttnn_tensor, pcie_device_mesh, memory_config=memory_config)
        ttnn_loop_back_tensor = ttnn.from_device(ttnn_tensor)
        torch_loop_back_tensor = ttnn.to_torch(
            ttnn_loop_back_tensor, mesh_composer=ConcatMeshToTensor(pcie_device_mesh, dim=3)
        )
        assert torch.all(torch_tensor == torch_loop_back_tensor)

    for device in pcie_device_mesh.get_device_ids():
        pcie_device_mesh.get_device(device).enable_async(False)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
# @pytest.mark.parametrize("pcie_device_mesh", [8], indirect=True)
def test_multi_device_check_per_device_shard(pcie_device_mesh, layout, memory_config):
    """This test checks if the tensor is correctly sharded across devices"""
    from ttnn import ShardTensorToMesh, ConcatMeshToTensor

    for device in pcie_device_mesh.get_device_ids():
        pcie_device_mesh.get_device(device).enable_async(True)

    for i in range(50):
        torch_tensor = torch.rand((8, 1, 1024, 1024), dtype=torch.bfloat16)

        ttnn_tensor = ttnn.from_torch(torch_tensor, mesh_mapper=ShardTensorToMesh(pcie_device_mesh, dim=3))
        ttnn_tensor = ttnn.to_layout(ttnn_tensor, layout=layout)
        ttnn_tensor = ttnn.to_device(ttnn_tensor, pcie_device_mesh, memory_config=memory_config)
        ttnn_loop_back_tensor = ttnn.from_device(ttnn_tensor)

        shard_offset, shard_size = 0, 128
        for device_tensor in ttnn.get_device_tensors(ttnn_loop_back_tensor):
            device_tensor_torch = ttnn.to_torch(device_tensor)
            assert torch.all(device_tensor_torch == torch_tensor[..., shard_offset : shard_offset + shard_size])
            shard_offset += shard_size

    for device in pcie_device_mesh.get_device_ids():
        pcie_device_mesh.get_device(device).enable_async(False)


@pytest.mark.parametrize("shape", [(1, 1, 512, 512), (1, 1, 1024, 1024)])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
# @pytest.mark.parametrize("pcie_device_mesh", [8], indirect=True)
def test_multi_device_replicate(pcie_device_mesh, shape, layout, memory_config):
    """Test ReplicateTensorToMesh to broadcast a tensor across multiple devices"""
    from ttnn import ReplicateTensorToMesh, ListMeshToTensor

    for device in pcie_device_mesh.get_device_ids():
        pcie_device_mesh.get_device(device).enable_async(True)

    for i in range(100):
        print("Running iter: " + str(i))
        full_tensor = torch.rand(shape, dtype=torch.bfloat16)

        ttnn_tensor = ttnn.from_torch(
            full_tensor,
            mesh_mapper=ReplicateTensorToMesh(pcie_device_mesh),
            layout=layout,
            memory_config=memory_config,
            device=pcie_device_mesh,
        )
        ttnn_tensor = ttnn.to_device(ttnn_tensor, pcie_device_mesh)
        ttnn_loop_back_tensor = ttnn.from_device(ttnn_tensor)
        loopback_replicated_tensors = ttnn.to_torch(
            ttnn_loop_back_tensor, mesh_composer=ListMeshToTensor(pcie_device_mesh)
        )
        for loopback_replicated_tensor in loopback_replicated_tensors:
            assert torch.all(full_tensor == loopback_replicated_tensor)

    for device in pcie_device_mesh.get_device_ids():
        pcie_device_mesh.get_device(device).enable_async(False)


# @pytest.mark.parametrize("pcie_device_mesh", [8], indirect=True)
@pytest.mark.parametrize("program_cache", [False, True])
@pytest.mark.parametrize("shape", [(1, 1, 512, 512), (1, 3, 1024, 1024)])
def test_multi_device_unary_binary_op_chain(pcie_device_mesh, program_cache, shape):
    """Multidevice API test: Running tensor-parallel multi-device chain of eltwise ops"""
    from ttnn import ShardTensorToMesh, ConcatMeshToTensor

    for device in pcie_device_mesh.get_device_ids():
        pcie_device_mesh.get_device(device).enable_async(True)
        if program_cache:
            pcie_device_mesh.get_device(device).enable_program_cache()

    torch_silu = torch.nn.SiLU()
    for i in range(50):
        torch_input_tensor = torch.rand(shape, dtype=torch.bfloat16)
        torch_output_golden = torch.add(
            torch.subtract(
                torch.exp(torch.nn.functional.relu(torch.nn.functional.gelu(torch_input_tensor))),
                torch.exp(torch_input_tensor),
            ),
            torch_silu(torch_input_tensor),
        )

        ttnn_input_tensor = ttnn.from_torch(
            torch_input_tensor,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ShardTensorToMesh(pcie_device_mesh, dim=3),
            device=pcie_device_mesh,
        )
        ttnn_output_tensor = ttnn.add(
            ttnn.sub(ttnn.exp(ttnn.relu(ttnn.gelu(ttnn_input_tensor))), ttnn.exp(ttnn_input_tensor)),
            ttnn.silu(ttnn_input_tensor),
        )
        ttnn_torch_output_tensor = ttnn.from_device(ttnn_output_tensor)
        ttnn_torch_output_tensor = ttnn.to_torch(
            ttnn_output_tensor, mesh_composer=ConcatMeshToTensor(pcie_device_mesh, dim=3)
        )
        assert_with_pcc(ttnn_torch_output_tensor, torch_output_golden, pcc=0.98)

    for device in pcie_device_mesh.get_device_ids():
        pcie_device_mesh.get_device(device).enable_async(False)


# @pytest.mark.parametrize("pcie_device_mesh", [8], indirect=True)
@pytest.mark.parametrize("program_cache", [False, True])
def test_multi_device_data_parallel_op_chain(pcie_device_mesh, program_cache):
    """Multidevice API: Running data-parallel chain of ops with matmul"""
    from ttnn import ShardTensorToMesh, ConcatMeshToTensor, ReplicateTensorToMesh

    for device in pcie_device_mesh.get_device_ids():
        pcie_device_mesh.get_device(device).enable_async(True)
        if program_cache:
            pcie_device_mesh.get_device(device).enable_program_cache()

    import time

    torch_silu = torch.nn.SiLU()
    start = time.time()
    for i in range(5):
        print("Running iter " + str(i))
        torch_input_a_tensor = torch.rand((16, 1, 512, 512), dtype=torch.bfloat16)
        torch_input_b_tensor = torch.rand((1, 1, 512, 512), dtype=torch.bfloat16)
        torch_output_golden = torch_silu(
            torch.nn.functional.relu(torch.nn.functional.gelu(torch_input_a_tensor @ torch_input_b_tensor))
            @ torch.exp(torch_input_a_tensor)
        )

        ttnn_input_a_tensor = ttnn.from_torch(
            torch_input_a_tensor,
            layout=ttnn.TILE_LAYOUT,
            device=pcie_device_mesh,
            mesh_mapper=ShardTensorToMesh(pcie_device_mesh, dim=0),
        )
        ttnn_input_b_tensor = ttnn.from_torch(
            torch_input_b_tensor,
            layout=ttnn.TILE_LAYOUT,
            device=pcie_device_mesh,
            mesh_mapper=ReplicateTensorToMesh(pcie_device_mesh),
        )
        ttnn_output_tensor = ttnn.from_device(
            ttnn.silu(ttnn.relu(ttnn.gelu(ttnn_input_a_tensor @ ttnn_input_b_tensor)) @ ttnn.exp(ttnn_input_a_tensor))
        )
        ttnn_torch_output_tensor = ttnn.to_torch(
            ttnn_output_tensor, mesh_composer=ConcatMeshToTensor(pcie_device_mesh, dim=0)
        )
        assert_with_pcc(ttnn_torch_output_tensor, torch_output_golden, pcc=0.97)
    print("Time taken: " + str(time.time() - start))

    for device in pcie_device_mesh.get_device_ids():
        pcie_device_mesh.get_device(device).enable_async(False)


# @pytest.mark.parametrize("pcie_device_mesh", [2], indirect=True)
# def test_multi_device_explicit_dealloc(pcie_device_mesh):
#     """Multidevice API: Data Parallel on matmul"""
#     from ttnn import ShardTensorToMesh, ConcatMeshToTensor, ReplicateTensorToMesh
#     import time

#     for device in pcie_device_mesh.get_device_ids():
#         pcie_device_mesh.get_device(device).enable_async(True)
#     start = time.time()
#     for i in range(1):
#         print("Running iter " + str(i))
#         torch_input_a_tensor = torch.rand((512, 1, 2048, 2048), dtype=torch.bfloat16)
#         torch_input_b_tensor = torch.rand((1, 1, 2048, 2048), dtype=torch.bfloat16)
#         torch_output_golden = torch.nn.functional.relu(
#             torch.nn.functional.gelu(torch_input_a_tensor @ torch_input_b_tensor)
#         ) @ torch_input_a_tensor
#         ttnn_input_a_tensor = ttnn.from_torch(
#             torch_input_a_tensor,
#             layout=ttnn.TILE_LAYOUT,
#             device=pcie_device_mesh,
#             mesh_mapper=ShardTensorToMesh(pcie_device_mesh, dim=0),
#         )
#         ttnn_input_b_tensor = ttnn.from_torch(
#             torch_input_b_tensor,
#             layout=ttnn.TILE_LAYOUT,
#             device=pcie_device_mesh,
#             mesh_mapper=ReplicateTensorToMesh(pcie_device_mesh),
#         )
#         ttnn_output_tensor_1 = ttnn_input_a_tensor @ ttnn_input_b_tensor
#         ttnn_output_tensor_2 = ttnn.gelu(ttnn_output_tensor_1)
#         ttnn_output_tensor_1.deallocate()
#         ttnn_input_b_tensor.deallocate()
#         ttnn_output_tensor_3 = ttnn.relu(ttnn_output_tensor_2)
#         ttnn_output_tensor_2.deallocate()
#         ttnn_output_tensor_4 = ttnn_output_tensor_3 @ ttnn_input_a_tensor
#         ttnn_output_tensor_3.deallocate()
#         ttnn_output_tensor = ttnn.from_device(ttnn_output_tensor_4)
#         ttnn_torch_output_tensor = ttnn.to_torch(
#             ttnn_output_tensor, mesh_composer=ConcatMeshToTensor(pcie_device_mesh, dim=0)
#         )
#         assert_with_pcc(ttnn_torch_output_tensor, torch_output_golden, pcc=0.97)
#     print("Time taken: " + str(time.time() - start))

#     for device in pcie_device_mesh.get_device_ids():
#         pcie_device_mesh.get_device(device).enable_async(False)
