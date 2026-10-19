# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

import inspect

import pytest

pytest.importorskip("nkilib")
pytest.importorskip("spmd_types")
pytest.importorskip("torch_neuronx")

from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.models.common.attention import ScaledDotProductAttention
from torchtitan.models.common.rope import CosSinRoPE
from torchtitan.models.qwen3 import (
    config_registry as core_config_registry,
    model_registry as core_model_registry,
    qwen3_configs,
)

from torchtitan_neuron.components.quantization import torchao_supports_neuron
from torchtitan_neuron.models.qwen3 import config_registry, model_registry, nki_moe


def _config_factories():
    factories = []
    for name, factory in inspect.getmembers(config_registry, inspect.isfunction):
        if (
            not name.startswith("qwen3_")
            or factory.__module__ != config_registry.__name__
        ):
            continue
        if any(
            parameter.default is inspect.Parameter.empty
            for parameter in inspect.signature(factory).parameters.values()
        ):
            continue
        # MXFP8 recipes attach a torchao-backed converter at build time; skip them
        # unless torchao with Neuron support is installed (public torchao lacks it).
        marks = (
            pytest.mark.skipif(
                not torchao_supports_neuron(),
                reason="MXFP8 configs require torchao with Neuron support",
            )
            if "mxfp8" in name
            else ()
        )
        factories.append(pytest.param(factory, id=name, marks=marks))
    return factories


@pytest.fixture
def recipe_config(config_factory):
    if "mxfp8" in config_factory.__name__:
        pytest.importorskip("torchao")
    return config_factory()


@pytest.mark.parametrize("config_factory", _config_factories())
def test_trainer_recipes_finalize_model_sharding(recipe_config):
    config = recipe_config

    config.model_spec.model.update_from_config(config=config)

    for layer in config.model_spec.model.layers:
        moe = layer.moe
        if isinstance(moe, nki_moe.NKIMoE.Config):
            assert moe.sharding_config is not None
            assert moe.router.gate.sharding_config is not None
            assert moe.experts.sharding_config is not None


def test_model_map_contains_moe_presets_only():
    assert set(config_registry._QWEN3_NKI_MOE_PRESETS) == {
        "30B-A3B",
        "235B-A22B",
    }


def test_qwen3_235b_base_recipe_is_neuron_owned_and_importable():
    assert not hasattr(core_config_registry, "qwen3_235b_a22b")

    config = config_registry.qwen3_235b_a22b()

    assert config.model_spec.flavor == "235B-A22B"
    assert config.hf_assets_path == "./assets/hf/Qwen3-235B-A22B"
    assert config.training.seq_len == 4096
    assert isinstance(config.activation_checkpoint, FullAC.Config)


def test_scaled_235b_model_flavor_is_neuron_owned():
    assert "235B-A22B-scaled" not in qwen3_configs

    config = model_registry("235B-A22B-scaled", attn_backend="sdpa").model

    assert config.dim == 1024
    assert len(config.layers) == 2
    assert config.layers[0].moe.num_experts == 16
    assert config.layers[0].moe.router.top_k == 4


def test_eager_and_compile_recipes_select_the_same_nki_model():
    eager = config_registry.qwen3_30b_a3b_nki_fullac_eager()
    compiled = config_registry.qwen3_30b_a3b_nki_fullac_compile()

    for config in (eager, compiled):
        assert config.model_spec.flavor == "30B-A3B-nki"
        assert len(config.model_spec.model.layers) == 48
        assert config.training.steps == 10
        assert config.dataloader.dataset == "c4_test"

    assert eager.compile.enable is False
    assert eager.override.imports == []
    assert compiled.compile.enable is True
    assert compiled.compile.backend == "neuron"
    assert compiled.compile.components == ["model", "loss", "grad_clip"]
    assert compiled.override.imports == []


@pytest.mark.parametrize(
    "config_factory",
    [
        case
        for case in _config_factories()
        if "nki" not in case.id and "235b" not in case.id
    ],
)
def test_dense_recipes_select_core_prefix_rope(recipe_config):
    config = recipe_config

    for layer in config.model_spec.model.layers:
        assert type(layer.attention.rope) is CosSinRoPE.Config
        assert layer.attention.rope.use_prefix_cache_for_positions is True
    assert config.override.imports == []
    if config.compile.enable:
        assert "grad_clip" in config.compile.components


def test_neuron_recipe_does_not_change_core_model_positions():
    config_registry.qwen3_8b_tp4_fsdp16_sac()
    model = core_model_registry("8B").model

    assert all(
        not layer.attention.rope.use_prefix_cache_for_positions
        for layer in model.layers
    )


def _lr(config):
    return config.optimizer.param_groups[0].optimizer_kwargs["lr"]


def _assert_chunked_cross_entropy_loss(config):
    assert isinstance(config.loss, ChunkedLossWrapper.Config)
    assert isinstance(config.loss.loss_fn, CrossEntropyLoss.Config)


def _assert_sdpa(config):
    assert isinstance(
        config.model_spec.model.layers[0].attention.inner_attention,
        ScaledDotProductAttention.Config,
    )


def test_qwen3_8b_dense_eager_contract():
    config = config_registry.qwen3_8b_tp1_fsdp64_fullac_eager()

    assert config.model_spec.name == "qwen3"
    assert config.model_spec.flavor == "8B"
    _assert_chunked_cross_entropy_loss(config)
    _assert_sdpa(config)
    assert config.parallelism.tensor_parallel_degree == 1
    assert config.parallelism.data_parallel_shard_degree == -1
    assert config.training.local_batch_size == 1
    assert config.training.global_batch_size == 64
    assert _lr(config) == 8e-4
    assert isinstance(config.activation_checkpoint, FullAC.Config)
    assert config.compile.enable is False
    assert config.override.imports == []


def test_qwen3_8b_dense_compile_contract():
    config = config_registry.qwen3_8b_tp1_fsdp64_fullac_compile()

    assert config.model_spec.name == "qwen3"
    assert config.model_spec.flavor == "8B"
    _assert_chunked_cross_entropy_loss(config)
    _assert_sdpa(config)
    assert config.parallelism.tensor_parallel_degree == 1
    assert config.parallelism.data_parallel_shard_degree == -1
    assert config.training.local_batch_size == 1
    assert config.training.global_batch_size == 64
    assert _lr(config) == 8e-4
    assert isinstance(config.activation_checkpoint, FullAC.Config)
    assert config.compile.enable is True
    assert config.compile.backend == "neuron"
    assert config.compile.components == ["model", "loss", "grad_clip"]
    assert config.override.imports == []


def test_qwen3_32b_dense_compile_contract():
    config = config_registry.qwen3_32b_tp4_fsdp16_fullac_compile()

    assert config.model_spec.name == "qwen3"
    assert config.model_spec.flavor == "32B"
    _assert_chunked_cross_entropy_loss(config)
    _assert_sdpa(config)
    assert config.parallelism.tensor_parallel_degree == 4
    assert config.parallelism.data_parallel_shard_degree == -1
    assert config.training.local_batch_size == 1
    assert config.training.global_batch_size == 16
    assert _lr(config) == 8e-4
    assert isinstance(config.activation_checkpoint, FullAC.Config)
    assert config.compile.enable is True
    assert config.compile.backend == "neuron"
    assert config.compile.components == ["model", "loss", "grad_clip"]
    assert config.override.imports == []


def test_qwen3_235b_production_recipe():
    config = config_registry.qwen3_235b_a22b_neuron()

    assert config.model_spec.flavor == "235B-A22B-neuron"
    assert len(config.model_spec.model.layers) == 94
    assert config.training.seq_len == 4096
    assert config.parallelism.tensor_parallel_degree == 4
    assert config.parallelism.data_parallel_shard_degree == 32
    assert config.parallelism.data_parallel_replicate_degree == 4
    assert config.parallelism.pipeline_parallel_degree == 2
    assert config.parallelism.expert_parallel_degree == 128
    assert config.compile.enable is True
    assert config.compile.backend == "neuron"
    for layer in config.model_spec.model.layers:
        assert isinstance(layer.moe.experts, nki_moe.BWMMGroupedExperts.Config)
