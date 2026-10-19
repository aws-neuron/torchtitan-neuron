# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

import pytest


pytest.importorskip("spmd_types")

from torchtitan.experiments.graph_trainer.configs import (  # noqa: E402
    GraphTrainerCompileConfig,
)
from torchtitan.experiments.graph_trainer.llama3.model import (  # noqa: E402
    GraphTrainerLlama3Model,
)
from torchtitan.models.common.attention import ScaledDotProductAttention  # noqa: E402
from torchtitan.models.llama3.config_registry import llama3_8b  # noqa: E402

from torchtitan_neuron.experiments.graph_trainer.llama3 import (  # noqa: E402
    config_registry as graph_trainer_llama3_config_registry,
)


def test_simple_fsdp_llama3_8b_neuron_override_config_contract():
    base = llama3_8b()
    config = graph_trainer_llama3_config_registry.simple_fsdp_llama3_8b()

    assert config.model_spec.name == "simple_fsdp/llama3"
    assert config.model_spec.flavor == base.model_spec.flavor == "8B"
    assert isinstance(config.model_spec.model, GraphTrainerLlama3Model.Config)
    assert len(config.model_spec.model.layers) == len(base.model_spec.model.layers)
    assert isinstance(
        config.model_spec.model.layers[0].attention.inner_attention,
        ScaledDotProductAttention.Config,
    )

    assert isinstance(config.compile, GraphTrainerCompileConfig)
    assert config.compile.enable is True
    assert config.compile.backend == "neuron"
    assert config.compile.mode == "jit"
    assert config.compile.components == ["model", "loss", "grad_clip"]

    assert config.override.imports == []
    assert all(
        layer.attention.rope.use_prefix_cache_for_positions
        for layer in config.model_spec.model.layers
    )


@pytest.mark.parametrize(
    "config_factory",
    [
        graph_trainer_llama3_config_registry.simple_fsdp_llama3_8b,
        graph_trainer_llama3_config_registry.simple_fsdp_llama3_8b_fullac,
        graph_trainer_llama3_config_registry.simple_fsdp_llama3_70b_fullac,
    ],
)
def test_simple_fsdp_llama3_uses_sdpa_backend(config_factory):
    config = config_factory()

    assert config.model_spec is not None
    assert all(
        isinstance(
            layer.attention.inner_attention,
            ScaledDotProductAttention.Config,
        )
        for layer in config.model_spec.model.layers
    )
