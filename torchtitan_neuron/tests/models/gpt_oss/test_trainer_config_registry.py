# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

# ruff: noqa: E402

import pytest
import torch

pytest.importorskip("nkilib")
pytest.importorskip("spmd_types")
pytest.importorskip("torch_neuronx")

from torchtitan.models.gpt_oss import model_registry

from torchtitan_neuron.models.gpt_oss.config_registry import gpt_oss_20b_nki_tp4_fsdp16
from torchtitan_neuron.models.gpt_oss.nki_attention import NKIFlashAttentionWrapper
from torchtitan_neuron.models.gpt_oss.nki_moe import NKIMoE, to_nki_moe_config

_PARAM_SHAPES = {
    "mlp1_weight": (4, 8, 2, 6),
    "mlp1_bias": (4, 2, 6),
    "mlp2_weight": (4, 6, 8),
    "mlp2_bias": (4, 8),
}
_STOCK_NAMES = {
    "mlp1_weight": "mlp1_weight_EGD",
    "mlp1_bias": "mlp1_bias_EG",
    "mlp2_weight": "mlp2_weight_EDF",
    "mlp2_bias": "mlp2_bias_ED",
}


def _init(fn, shape, seed):
    torch.manual_seed(seed)
    tensor = torch.empty(shape)
    fn(tensor)
    return tensor


def test_to_nki_moe_config_keeps_stock_router_and_init():
    stock = model_registry("20b").model.layers[3].moe
    nki = to_nki_moe_config(stock)

    assert isinstance(nki, NKIMoE.Config)
    assert nki.num_experts == stock.num_experts
    assert nki.load_balance_coeff == stock.load_balance_coeff
    assert nki.router.top_k == nki.experts.top_k == stock.router.top_k
    assert nki.router.gate == stock.router.gate
    assert nki.experts.swiglu_limit == stock.experts.swiglu_limit

    for name, shape in _PARAM_SHAPES.items():
        stock_init = stock.experts.param_init[_STOCK_NAMES[name]]
        expected = _init(stock_init, shape, seed=7)
        if name == "mlp1_bias":
            # GPT-OSS adds +1 to the up bias (index 1 of the packed gate/up axis).
            expected[:, 1, :] += 1.0
        torch.testing.assert_close(
            _init(nki.experts.param_init[name], shape, seed=7), expected
        )


def test_gpt_oss_nki_recipe_uses_nki_modules():
    config = gpt_oss_20b_nki_tp4_fsdp16()
    model = config.model_spec.model
    model.update_from_config(config=config)

    for layer in model.layers:
        assert isinstance(layer.moe, NKIMoE.Config)
        assert isinstance(
            layer.attention.inner_attention, NKIFlashAttentionWrapper.Config
        )
        assert layer.attention.rope.use_prefix_cache_for_positions is True
        # Placements come from NKIFlashAttentionWrapper.Config.set_sharding_config.
        assert set(
            layer.attention.inner_attention.sharding_config.in_src_shardings
        ) == {
            "q",
            "k",
            "v",
            "sinks",
        }
