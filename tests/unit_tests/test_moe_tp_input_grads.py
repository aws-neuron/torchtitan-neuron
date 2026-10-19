# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

"""With EP off, the routed experts' input gradients must be declared Partial over TP.

Each TP rank computes its experts on a hidden-dim slice, so its x / top-k-score gradients are partial
sums; without the declaration local_map assumes Replicate and never sums them.
"""

from __future__ import annotations

import pytest
import spmd_types as spmd

from torchtitan.models.common.moe import MoE
from torchtitan.models.common.moe_sharding import set_moe_sharding_config, TP

_EXPERT_LAYOUT = {"w1_EFD": spmd.S(1), "w2_EDF": spmd.S(2), "w3_EFD": spmd.S(1)}


def _moe_config() -> MoE.Config:
    from torchtitan.models.common.config_utils import (
        make_experts_config,
        make_moe_config,
        make_router_config,
    )

    init = {"weight": lambda x: x}
    return make_moe_config(
        num_experts=8,
        router=make_router_config(
            dim=16,
            num_experts=8,
            top_k=2,
            gate_param_init=init,
            score_func="softmax",
        ),
        experts=make_experts_config(
            dim=16,
            hidden_dim=32,
            num_experts=8,
            top_k=2,
            param_init=init,
            comm_backend="standard",
        ),
    )


@pytest.mark.parametrize("enable_sp", [True, False])
def test_tp_only_experts_sum_their_input_gradients_over_tp(enable_sp):
    """KILLS: the TP-only MoE handing back ~1/TP of its input and routing-score gradients."""
    cfg = _moe_config()
    set_moe_sharding_config(
        cfg, enable_ep=False, enable_sp=enable_sp, expert_param_layout=_EXPERT_LAYOUT
    )
    grads = cfg.experts.sharding_config.local_map.in_grad_placements
    assert (
        grads is not None
    ), "TP-only experts omit in_grad_placements: partials are never summed"
    x_grad, scores_grad = grads[0], grads[1]
    assert x_grad.axis_types[TP] == spmd.P
    assert scores_grad.axis_types[TP] == spmd.P


def test_the_ep_path_is_unchanged():
    """The EP path already declared its gradients; the fix must not change them."""
    cfg = _moe_config()
    set_moe_sharding_config(
        cfg, enable_ep=True, enable_sp=True, expert_param_layout=_EXPERT_LAYOUT
    )
    grads = cfg.experts.sharding_config.local_map.in_grad_placements
    assert grads is not None
    assert grads[0] == grads[1] == grads[2]
    assert grads[0].axis_types[TP] != spmd.P
