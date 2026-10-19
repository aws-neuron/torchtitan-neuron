# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

# ruff: noqa: E402
"""Qwen3 global-batch load-balancing loss on the NKI MoE."""

import copy
from unittest import mock

import pytest
import torch

pytest.importorskip("nkilib")
pytest.importorskip("spmd_types")
pytest.importorskip("torch_neuronx")

from torch.distributed.tensor import DTensor
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.models.common.nn_modules import Linear

from torchtitan_neuron.models.qwen3 import nki_moe
from torchtitan_neuron.tests.models.qwen3.test_nki_moe import (
    _ROUTER_GRAD_BATCH as _BATCH,
    _ROUTER_GRAD_DIM as _DIM,
    _router_grad_fake_nki_fwd as _fake_nki_fwd,
    _ROUTER_GRAD_HIDDEN_DIM as _HIDDEN_DIM,
    _ROUTER_GRAD_NUM_EXPERTS as _NUM_EXPERTS,
    _ROUTER_GRAD_SEQ as _SEQ,
    _ROUTER_GRAD_TOP_K as _TOP_K,
)


def _nki_moe():
    config = nki_moe.NKIMoE.Config(
        num_experts=_NUM_EXPERTS,
        load_balance_coeff=1.0,
        load_balancing="global_aux_loss",
        router=nki_moe.MaskedScoresTopKRouter.Config(
            gate=Linear.Config(in_features=_DIM, out_features=_NUM_EXPERTS, bias=False),
            top_k=_TOP_K,
            score_func="softmax",
            route_norm=True,
        ),
        experts=nki_moe.NKIGroupedExperts.Config(
            dim=_DIM, hidden_dim=_HIDDEN_DIM, num_experts=_NUM_EXPERTS, top_k=_TOP_K
        ),
    )
    config.set_sharding_config(enable_sp=False, enable_ep=False)
    torch.manual_seed(0)
    moe = config.build()
    with torch.no_grad():
        for param in moe.parameters():
            param.normal_(std=0.5)
    return moe.train()


def _nki_gate_grad(moe, x_BSD, main_weight):
    with mock.patch.object(nki_moe, "nki_fwd", _fake_nki_fwd([])):
        out = moe(x_BSD)
    out = out.full_tensor() if isinstance(out, DTensor) else out
    (main_weight * out.float().pow(2).sum()).backward()
    grad = moe.router.gate.weight.grad
    return grad.full_tensor() if isinstance(grad, DTensor) else grad


class TestNKIUnderTP(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @property
    def device_type(self) -> str:
        return "cpu"

    @with_comms
    def test_gate_gradient_matches_tp1(self):
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=self.world_size,
            pp=1,
            ep=1,
            world_size=self.world_size,
        )
        with mock.patch("torchtitan.distributed.parallel_dims.device_type", "cpu"):
            parallel_dims.build_mesh()
        torch.manual_seed(1)
        x_BSD = torch.randn(_BATCH, _SEQ, _DIM)
        # Aux term alone (zero main loss), then with a main loss: both must equal TP1.
        for main_weight in (0.0, 1.0 / (_BATCH * _SEQ)):
            moe = _nki_moe()
            reference = copy.deepcopy(moe)
            moe.parallelize(parallel_dims)
            want = _nki_gate_grad(reference, x_BSD, main_weight)
            assert float(want.norm()) > 0
            torch.testing.assert_close(_nki_gate_grad(moe, x_BSD, main_weight), want)
