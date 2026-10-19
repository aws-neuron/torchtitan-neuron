# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

# ruff: noqa: E402

import inspect

import pytest

pytest.importorskip("nkilib")
pytest.importorskip("spmd_types")
pytest.importorskip("torch_neuronx")

from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.config import CompileConfig
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.experiments.graph_trainer.qwen3 import (
    config_registry as core_graph_config_registry,
)
from torchtitan.experiments.graph_trainer.qwen3.parallelize import parallelize_qwen3

from torchtitan_neuron.experiments.graph_trainer.qwen3 import (
    config_registry as graph_config_registry,
)
from torchtitan_neuron.models.qwen3 import (
    config_registry as trainer_config_registry,
    nki_moe,
)


def _config_factories():
    factories = []
    for name, factory in inspect.getmembers(
        graph_config_registry,
        inspect.isfunction,
    ):
        if (
            not name.startswith("qwen3_")
            or factory.__module__ != graph_config_registry.__name__
        ):
            continue
        if any(
            parameter.default is inspect.Parameter.empty
            for parameter in inspect.signature(factory).parameters.values()
        ):
            continue
        factories.append(pytest.param(factory, id=name))
    return factories


@pytest.mark.parametrize("config_factory", _config_factories())
def test_graph_trainer_recipes_finalize_model_sharding(config_factory):
    config = config_factory()

    config.model_spec.model.update_from_config(config=config)

    for layer in config.model_spec.model.layers:
        moe = layer.moe
        if isinstance(moe, nki_moe.NKIMoE.Config):
            assert moe.sharding_config is not None
            assert moe.router.gate.sharding_config is not None
            assert moe.experts.sharding_config is not None


def test_neuron_graph_trainer_recipes_are_not_registered_in_core():
    assert not hasattr(
        core_graph_config_registry,
        "graph_trainer_qwen3_235b_scaled",
    )
    assert not hasattr(
        core_graph_config_registry,
        "simple_fsdp_qwen3_32b_tp4_fsdp16_fullac",
    )


def test_relocated_235b_aot_graph_trainer_recipes():
    for config in (
        graph_config_registry.graph_trainer_qwen3_235b_scaled(),
        graph_config_registry.graph_trainer_qwen3_235b_4layers_ep64_nle1_tp4_neuron_ep(),
    ):
        assert config.model_spec.name == "qwen3"
        assert isinstance(config.compile, GraphTrainerCompileConfig)
        assert config.compile.enable is True
        assert config.compile.mode == "aot_fx_trace"


def test_qwen3_30b_simple_fsdp_recipe_reuses_trainer_model():
    base = trainer_config_registry.qwen3_nki_moe_config(
        "30B-A3B",
        compile_config=CompileConfig(enable=False, backend="neuron"),
    )
    config = graph_config_registry.qwen3_30b_a3b_nki_fullac_simplefsdp()

    assert config.model_spec.flavor == base.model_spec.flavor
    assert len(config.model_spec.model.layers) == 48
    assert config.training.local_batch_size == base.training.local_batch_size
    assert config.training.steps == base.training.steps
    assert config.dataloader.dataset == base.dataloader.dataset
    assert config.model_spec.name == "simple_fsdp/qwen3"
    assert isinstance(config.compile, GraphTrainerCompileConfig)
    assert config.compile.enable is True
    assert config.compile.mode == "jit"
    assert config.compile.components == ["model", "loss", "grad_clip"]
    assert config.model_spec.parallelize_fn is parallelize_qwen3
    assert config.model_spec.post_optimizer_build_fn is register_moe_load_balancing_hook
    assert config.override.imports == []


def test_qwen3_235b_six_layer_4k_neuron_ep_recipes():
    trainer = (
        trainer_config_registry.qwen3_235b_6layers_ep64_nle1_tp4_neuron_ep_compile()
    )
    graph = (
        graph_config_registry.qwen3_235b_6layers_ep64_nle1_tp4_neuron_ep_compile_simplefsdp()
    )

    for config in (trainer, graph):
        assert len(config.model_spec.model.layers) == 6
        assert config.training.seq_len == 4096
        assert isinstance(config.activation_checkpoint, SelectiveAC.Config)
        assert config.parallelism.expert_parallel_degree == 64
        assert config.parallelism.tensor_parallel_degree == 4
        assert config.compile.enable is True
        assert config.compile.backend == "neuron"
        assert config.compile.components == ["model", "loss"]
        for layer in config.model_spec.model.layers:
            assert layer.moe.num_experts == 64
            assert layer.moe.router.top_k == 8
            assert isinstance(layer.moe.experts, nki_moe.BWMMGroupedExperts.Config)

    assert graph.model_spec.name == "simple_fsdp/qwen3"


def test_qwen3_235b_ep128_4k_neuron_ep_recipes():
    trainer = (
        trainer_config_registry.qwen3_235b_6layers_ep128_nle1_tp4_neuron_ep_compile()
    )
    graph = (
        graph_config_registry.qwen3_235b_6layers_ep128_nle1_tp4_neuron_ep_compile_simplefsdp()
    )

    for config in (trainer, graph):
        assert len(config.model_spec.model.layers) == 6
        assert config.training.seq_len == 4096
        assert isinstance(config.activation_checkpoint, SelectiveAC.Config)
        assert config.parallelism.expert_parallel_degree == 128
        assert config.parallelism.tensor_parallel_degree == 4
        for layer in config.model_spec.model.layers:
            assert layer.moe.num_experts == 128
            assert layer.moe.router.top_k == 8
            assert isinstance(layer.moe.experts, nki_moe.BWMMGroupedExperts.Config)


def test_qwen3_235b_94layer_ep128_fsdp64_pp1_topology():
    config = (
        trainer_config_registry.qwen3_235b_94layers_ep128_tp4_fsdp64_pp1_4k_neuron_ep_compile()
    )

    assert len(config.model_spec.model.layers) == 94
    assert config.training.seq_len == 4096
    assert config.training.local_batch_size == 1
    assert isinstance(config.activation_checkpoint, FullAC.Config)

    parallel_dims = ParallelDims.from_config(config.parallelism, world_size=256)
    assert parallel_dims.pp == 1
    assert parallel_dims.dp_replicate == 1
    assert parallel_dims.dp_shard == 64
    assert parallel_dims.tp == 4
    assert parallel_dims.ep == 128
    expert_fsdp = (
        parallel_dims.dp_shard * parallel_dims.cp * parallel_dims.tp // parallel_dims.ep
    )
    assert expert_fsdp == 2


def test_qwen3_235b_ep128_pp2_preserves_pipeline_configuration():
    trainer = (
        trainer_config_registry.qwen3_235b_6layers_ep128_nle1_tp4_neuron_ep_pp2_compile()
    )
    graph = (
        graph_config_registry.qwen3_235b_6layers_ep128_nle1_tp4_neuron_ep_pp2_compile_simplefsdp()
    )

    for config in (trainer, graph):
        assert config.parallelism.expert_parallel_degree == 128
        assert config.parallelism.tensor_parallel_degree == 4
        assert config.parallelism.pipeline_parallel_degree == 2
        assert config.parallelism.pipeline_parallel_schedule == "1F1B"
        assert config.parallelism.pipeline_parallel_first_stage_less_layers == 0
        assert config.parallelism.pipeline_parallel_last_stage_less_layers == 0
        assert config.training.local_batch_size == 16

    assert graph.model_spec.name == "simple_fsdp/qwen3"
    assert graph.model_spec.pipelining_fn is not None
