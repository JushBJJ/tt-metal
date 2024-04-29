#/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
  exit 1
fi

echo "Running nightly tests for WH B0 only"

env pytest tests/ttnn/integration_tests/unet                # -> failing: issue #7556
env WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -svv tests/ttnn/integration_tests/stable_diffusion/test_cross_attention.py  -k 512
env WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -svv tests/ttnn/integration_tests/stable_diffusion/test_cross_attn_up_block_2d.py  -k 512
# env pytest tests/ttnn/integration_tests/stable_diffusion    # -> failing/hanging: issue #7560

env pytest models/demos/mamba/tests/test_mamba_ssm.py
env pytest models/demos/mamba/tests/test_mamba_block.py
env pytest models/demos/mamba/tests/test_residual_block.py
env pytest models/demos/mamba/tests/test_full_model_loop.py
env pytest models/demos/mamba/tests/test_benchmarks.py
env pytest models/demos/mamba/tests/test_reference_model.py
env pytest models/demos/mamba/tests/test_transforms.py
env pytest models/demos/mamba/tests/test_mamba_demo.py

env pytest models/demos/mistral7b/tests/test_mistral_embedding.py
env pytest models/demos/mistral7b/tests/test_mistral_rms_norm.py
env pytest models/demos/mistral7b/tests/test_mistral_mlp.py
env WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest models/demos/mistral7b/tests/test_mistral_attention.py
env WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest models/demos/mistral7b/tests/test_mistral_decoder.py
