# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

import pytest


pytest.importorskip("spmd_types")

from torchtitan.components.loss import (  # noqa: E402
    ChunkedLossWrapper,
    CrossEntropyLoss,
)
from torchtitan.distributed.activation_checkpoint import FullAC  # noqa: E402
from torchtitan.experiments.graph_trainer.chunked_loss import (  # noqa: E402
    ChunkedLossWrapperWithParamGrads,
)
from torchtitan.experiments.graph_trainer.configs import (  # noqa: E402
    GraphTrainerCompileConfig,
)
from torchtitan.experiments.graph_trainer.qwen3.model import (  # noqa: E402
    GraphTrainerQwen3Model,
)
from torchtitan.models.common.attention import ScaledDotProductAttention  # noqa: E402
from torchtitan.models.qwen3 import (  # noqa: E402
    config_registry as core_qwen3_config_registry,
)

from torchtitan_neuron.experiments.graph_trainer.qwen3 import (  # noqa: E402
    config_registry as graph_trainer_qwen3_config_registry,
)
from torchtitan_neuron.models.qwen3 import (  # noqa: E402
    config_registry as qwen3_config_registry,
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


def test_neuron_dense_recipes_are_not_registered_in_core():
    assert not hasattr(
        core_qwen3_config_registry,
        "qwen3_32b_tp4_fsdp16_fullac_compile",
    )


def test_qwen3_8b_dense_eager_contract():
    config = qwen3_config_registry.qwen3_8b_tp1_fsdp64_fullac_eager()

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
    config = qwen3_config_registry.qwen3_8b_tp1_fsdp64_fullac_compile()

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


def test_simple_fsdp_qwen3_8b_config_contract():
    config = (
        graph_trainer_qwen3_config_registry.simple_fsdp_qwen3_8b_tp1_fsdp64_fullac()
    )

    assert config.model_spec.name == "simple_fsdp/qwen3"
    assert config.model_spec.flavor == "8B"
    assert isinstance(config.model_spec.model, GraphTrainerQwen3Model.Config)
    _assert_sdpa(config)

    assert config.parallelism.tensor_parallel_degree == 1
    assert config.parallelism.data_parallel_shard_degree == -1
    assert config.training.local_batch_size == 1
    assert config.training.global_batch_size == 64
    assert isinstance(config.activation_checkpoint, FullAC.Config)
    assert isinstance(config.compile, GraphTrainerCompileConfig)
    assert config.compile.enable is True
    assert config.compile.backend == "neuron"
    assert config.compile.mode == "jit"
    assert config.compile.components == ["model", "loss", "grad_clip"]

    assert isinstance(config.loss, ChunkedLossWrapperWithParamGrads.Config)
    assert isinstance(config.loss.loss_fn, CrossEntropyLoss.Config)
    assert config.override.imports == []
    assert all(
        layer.attention.rope.use_prefix_cache_for_positions
        for layer in config.model_spec.model.layers
    )


def test_qwen3_32b_dense_compile_contract():
    config = qwen3_config_registry.qwen3_32b_tp4_fsdp16_fullac_compile()

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
