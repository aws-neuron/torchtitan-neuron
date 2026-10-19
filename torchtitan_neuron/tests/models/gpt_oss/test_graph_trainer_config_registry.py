# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

# ruff: noqa: E402

import pytest

pytest.importorskip("nkilib")
pytest.importorskip("spmd_types")
pytest.importorskip("torch_neuronx")

from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig

from torchtitan_neuron.experiments.graph_trainer.gpt_oss import (
    config_registry as graph_config_registry,
    model_registry,
)
from torchtitan_neuron.experiments.graph_trainer.gpt_oss.model import (
    GraphTrainerGptOssModel,
)
from torchtitan_neuron.experiments.graph_trainer.gpt_oss.parallelize import (
    parallelize_gptoss,
)
from torchtitan_neuron.models.gpt_oss.config_registry import gpt_oss_20b_nki_tp4_fsdp16
from torchtitan_neuron.models.gpt_oss.nki_attention import NKIFlashAttentionWrapper
from torchtitan_neuron.models.gpt_oss.nki_moe import NKIMoE


def test_simple_fsdp_gpt_oss_neuron_config_contract():
    base = gpt_oss_20b_nki_tp4_fsdp16()
    config = graph_config_registry.simple_fsdp_gpt_oss_20b_nki_tp4_fsdp16()

    assert config.model_spec.name == "simple_fsdp/gpt_oss"
    assert config.model_spec.flavor == base.model_spec.flavor == "20b"
    assert isinstance(config.model_spec.model, GraphTrainerGptOssModel.Config)
    assert config.model_spec.parallelize_fn is parallelize_gptoss
    assert config.model_spec.post_optimizer_build_fn is register_moe_load_balancing_hook

    assert isinstance(config.compile, GraphTrainerCompileConfig)
    assert config.compile.enable is True
    assert config.compile.backend == "neuron"
    assert config.compile.mode == "jit"
    assert config.compile.components == ["model", "loss", "grad_clip"]
    assert isinstance(config.activation_checkpoint, FullAC.Config)

    for layer in config.model_spec.model.layers:
        assert isinstance(layer.moe, NKIMoE.Config)
        assert isinstance(
            layer.attention.inner_attention,
            NKIFlashAttentionWrapper.Config,
        )


def test_gpt_oss_model_registry_is_neuron_owned():
    model_spec = model_registry("20b")

    assert model_spec.name == "graph_trainer/gpt_oss"
    assert isinstance(model_spec.model, GraphTrainerGptOssModel.Config)
    assert model_spec.parallelize_fn is parallelize_gptoss
