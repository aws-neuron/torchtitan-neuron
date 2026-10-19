# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

import pytest

pytest.importorskip("spmd_types")

from torchtitan.distributed.activation_checkpoint import (  # noqa: E402
    FullAC,
    SelectiveAC,
)
from torchtitan.models.common.attention import ScaledDotProductAttention  # noqa: E402

from torchtitan_neuron.models.llama3 import config_registry  # noqa: E402

# (recipe, full_ac, compile, n_layers)
_RECIPES = [
    (config_registry.llama3_8b_sac_eager, False, False, 32),
    (config_registry.llama3_8b_fullac_eager, True, False, 32),
    (config_registry.llama3_70b_fullac_eager, True, False, 80),
    (config_registry.llama3_8b_sac_compile, False, True, 32),
    (config_registry.llama3_8b_fullac_compile, True, True, 32),
    (config_registry.llama3_70b_fullac_compile, True, True, 80),
    (config_registry.llama3_70b_16layers_fullac_compile, True, True, 16),
]


@pytest.mark.parametrize(
    ("recipe", "full_ac", "compile", "n_layers"),
    _RECIPES,
    ids=[r[0].__name__ for r in _RECIPES],
)
def test_neuron_dense_recipe_contract(recipe, full_ac, compile, n_layers):
    config = recipe()
    layers = config.model_spec.model.layers

    assert len(layers) == n_layers
    for layer in layers:
        assert isinstance(
            layer.attention.inner_attention, ScaledDotProductAttention.Config
        )
        assert layer.attention.rope.use_prefix_cache_for_positions is True
    assert isinstance(
        config.activation_checkpoint, FullAC.Config if full_ac else SelectiveAC.Config
    )
    assert config.dataloader.dataset == "c4_test"
    assert config.metrics.log_freq == 1
    assert config.compile.enable is compile
    if compile:
        assert config.compile.backend == "neuron"
        assert config.compile.components == ["model", "loss", "grad_clip"]
    assert config.override.imports == []
