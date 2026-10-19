# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

# ruff: noqa: E402

import contextlib
import copy
from unittest import mock

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

pytest.importorskip("nkilib")
pytest.importorskip("spmd_types")
pytest.importorskip("torch_neuronx")

from torch.distributed.tensor import DTensor, Partial, Replicate
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.models.common.moe import GroupedExperts, MoE, TokenChoiceTopKRouter
from torchtitan.models.common.nn_modules import Linear
from torchtitan.models.common.token_dispatcher import LocalTokenDispatcher

from torchtitan_neuron.models.qwen3 import nki_moe


def _fake_nki_fwd(
    hidden_states,
    masked_scores,
    gate_up_proj_weight,
    down_proj_weight,
    top_k,
    block_size,
):
    del gate_up_proj_weight, down_proj_weight, top_k, block_size
    output = hidden_states * masked_scores.sum(dim=-1, keepdim=True).to(
        hidden_states.dtype
    )
    return (
        masked_scores.reshape(-1, 1),
        torch.empty(0, device=hidden_states.device, dtype=hidden_states.dtype),
        torch.empty(0, device=hidden_states.device, dtype=hidden_states.dtype),
        torch.empty(0, device=hidden_states.device, dtype=torch.int64),
        torch.empty(0, device=hidden_states.device, dtype=torch.int64),
        hidden_states.shape[0],
        masked_scores.shape[1],
        output,
    )


def _tiny_nki_moe_config():
    dim = 8
    num_experts = 4
    top_k = 2
    return nki_moe.NKIMoE.Config(
        num_experts=num_experts,
        load_balance_coeff=None,
        router=nki_moe.MaskedScoresTopKRouter.Config(
            gate=Linear.Config(
                in_features=dim,
                out_features=num_experts,
                bias=False,
            ),
            top_k=top_k,
            score_func="sigmoid",
            route_norm=True,
        ),
        experts=nki_moe.NKIGroupedExperts.Config(
            dim=dim,
            hidden_dim=16,
            num_experts=num_experts,
            top_k=top_k,
        ),
    )


def _stock_moe_config():
    num_experts = 4
    top_k = 2
    return MoE.Config(
        num_experts=num_experts,
        load_balance_coeff=1e-3,
        shared_experts=None,
        router=TokenChoiceTopKRouter.Config(
            num_experts=num_experts,
            gate=Linear.Config(in_features=8, out_features=num_experts, bias=False),
            top_k=top_k,
            score_func="sigmoid",
            route_norm=True,
        ),
        experts=GroupedExperts.Config(
            dim=8,
            hidden_dim=16,
            num_experts=num_experts,
            token_dispatcher=LocalTokenDispatcher.Config(
                num_experts=num_experts,
                top_k=top_k,
            ),
        ),
    )


def test_masked_scores_router_executes_and_keeps_top_k_scores():
    torch.manual_seed(0)
    router = nki_moe.MaskedScoresTopKRouter(
        nki_moe.MaskedScoresTopKRouter.Config(
            gate=Linear.Config(in_features=8, out_features=4, bias=False),
            top_k=2,
            score_func="sigmoid",
            route_norm=True,
        )
    )

    scores = router(torch.randn(3, 8))

    assert scores.shape == (3, 4)
    assert torch.count_nonzero(scores, dim=-1).tolist() == [2, 2, 2]
    torch.testing.assert_close(scores.sum(dim=-1), torch.ones(3))


def test_nki_moe_initializes_compiles_and_executes_with_mocked_nki_op(monkeypatch):
    torch.manual_seed(0)
    monkeypatch.setattr(nki_moe, "nki_fwd", _fake_nki_fwd)
    model = nki_moe.NKIMoE(_tiny_nki_moe_config())

    x = torch.randn(2, 3, 8)
    eager_output = model(x)
    compiled_output = torch.compile(model, backend="eager", fullgraph=True)(x)

    assert eager_output.shape == x.shape
    torch.testing.assert_close(compiled_output, eager_output)


def test_bwmm_grouped_experts_packs_single_expert_weights(monkeypatch):
    captured = {}

    def fake_nki_fwd(
        hidden_states,
        masked_scores,
        gate_up_proj_weight,
        down_proj_weight,
        top_k,
        block_size,
    ):
        captured["masked_scores"] = masked_scores
        captured["gate_up_shape"] = gate_up_proj_weight.shape
        captured["down_shape"] = down_proj_weight.shape
        captured["top_k"] = top_k
        captured["block_size"] = block_size
        gate = hidden_states @ gate_up_proj_weight[0, :, 0, :]
        up = hidden_states @ gate_up_proj_weight[0, :, 1, :]
        output = (torch.nn.functional.silu(gate) * up) @ down_proj_weight[0]
        return (
            masked_scores.reshape(-1, 1),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            hidden_states.shape[0],
            1,
            output,
        )

    monkeypatch.setattr(nki_moe, "nki_fwd", fake_nki_fwd)
    torch.manual_seed(0)
    experts = nki_moe.BWMMGroupedExperts(
        nki_moe.BWMMGroupedExperts.Config(
            dim=8,
            hidden_dim=16,
            num_experts=1,
            token_dispatcher=LocalTokenDispatcher.Config(
                num_experts=1,
                top_k=1,
            ),
        )
    )
    with torch.no_grad():
        experts.w1_EFD.normal_(std=0.02)
        experts.w2_EDF.normal_(std=0.02)
        experts.w3_EFD.normal_(std=0.02)
    x = torch.randn(32, 8, dtype=torch.bfloat16, requires_grad=True)
    output = experts._experts_forward(x, torch.tensor([32], dtype=torch.int32))
    expected = (
        torch.nn.functional.silu(x @ experts.w1_EFD[0].bfloat16().T)
        * (x @ experts.w3_EFD[0].bfloat16().T)
    ) @ experts.w2_EDF[0].bfloat16().T

    assert captured["gate_up_shape"] == (1, 8, 2, 16)
    assert captured["down_shape"] == (1, 16, 8)
    assert captured["top_k"] == 1
    assert captured["block_size"] == nki_moe.EP_BLOCK_SIZE
    torch.testing.assert_close(captured["masked_scores"], torch.ones(32, 1))
    torch.testing.assert_close(output, expected)

    output.float().sum().backward()
    assert x.grad is not None
    assert experts.w1_EFD.grad is not None
    assert experts.w2_EDF.grad is not None
    assert experts.w3_EFD.grad is not None


def test_single_expert_block_metadata_skips_only_padding():
    tokens = nki_moe.BLOCK_SIZE + 7
    num_blocks = 3
    token_ids, expert_ids = nki_moe._single_expert_block_metadata(
        tokens,
        num_blocks,
        nki_moe.EP_BLOCK_SIZE,
        torch.device("cpu"),
    )

    torch.testing.assert_close(
        token_ids[:tokens],
        torch.arange(tokens, dtype=torch.int32),
    )
    torch.testing.assert_close(
        token_ids[tokens:],
        torch.full(
            (num_blocks * nki_moe.EP_BLOCK_SIZE - tokens,),
            -1,
            dtype=torch.int32,
        ),
    )
    torch.testing.assert_close(
        expert_ids,
        torch.zeros((num_blocks, 1), dtype=torch.int32),
    )


def test_qwen3_nki_conversion_does_not_preempt_decoder_sharding():
    stock_moe = _stock_moe_config()

    replacement = nki_moe.to_nki_moe_config(stock_moe)

    assert isinstance(replacement, nki_moe.NKIMoE.Config)
    assert replacement.num_experts == stock_moe.num_experts
    assert replacement.load_balance_coeff == stock_moe.load_balance_coeff
    assert isinstance(replacement.router, nki_moe.MaskedScoresTopKRouter.Config)
    assert isinstance(replacement.experts, nki_moe.NKIGroupedExperts.Config)
    assert replacement.router.top_k == stock_moe.router.top_k
    assert replacement.experts.top_k == stock_moe.router.top_k
    assert replacement.sharding_config is None
    assert replacement.router.gate.sharding_config is None
    assert replacement.experts.sharding_config is None


_ROUTER_GRAD_DIM = 8
_ROUTER_GRAD_HIDDEN_DIM = 16
_ROUTER_GRAD_NUM_EXPERTS = 4
_ROUTER_GRAD_TOP_K = 2
_ROUTER_GRAD_BATCH = 2
_ROUTER_GRAD_SEQ = 8


def _router_grad_fake_nki_fwd(score_grads: list[torch.Tensor]):
    def fake_nki_fwd(
        hidden_states,
        masked_scores,
        gate_up_proj_weight,
        down_proj_weight,
        top_k,
        block_size,
    ):
        del top_k, block_size
        masked_scores.register_hook(score_grads.append)
        gate = torch.einsum(
            "bd,edh->beh",
            hidden_states,
            gate_up_proj_weight[:, :, 0, :],
        )
        up = torch.einsum(
            "bd,edh->beh",
            hidden_states,
            gate_up_proj_weight[:, :, 1, :],
        )
        expert_output = torch.einsum(
            "beh,ehd->bed",
            F.silu(gate) * up,
            down_proj_weight,
        )
        output = torch.einsum("be,bed->bd", masked_scores, expert_output)
        tokens, num_experts = masked_scores.shape
        return (None,) * 5 + (tokens, num_experts, output)

    return fake_nki_fwd


def _partial_gate_output(scores):
    if isinstance(scores, DTensor):
        return scores.to_local(
            grad_placements=tuple(Partial() for _ in scores.placements)
        )
    return scores


def _router_grad_nki_moe(*, enable_sp: bool):
    config = nki_moe.NKIMoE.Config(
        num_experts=_ROUTER_GRAD_NUM_EXPERTS,
        load_balance_coeff=None,
        router=nki_moe.MaskedScoresTopKRouter.Config(
            gate=Linear.Config(
                in_features=_ROUTER_GRAD_DIM,
                out_features=_ROUTER_GRAD_NUM_EXPERTS,
                bias=False,
            ),
            top_k=_ROUTER_GRAD_TOP_K,
            score_func="softmax",
            route_norm=True,
        ),
        experts=nki_moe.NKIGroupedExperts.Config(
            dim=_ROUTER_GRAD_DIM,
            hidden_dim=_ROUTER_GRAD_HIDDEN_DIM,
            num_experts=_ROUTER_GRAD_NUM_EXPERTS,
            top_k=_ROUTER_GRAD_TOP_K,
        ),
    )
    config.set_sharding_config(enable_sp=enable_sp, enable_ep=False)
    torch.manual_seed(0)
    moe = config.build()
    with torch.no_grad():
        for parameter in moe.parameters():
            parameter.normal_(std=0.5)
    return moe


def _router_grad_forward_backward(moe, inputs, output_grad, score_grads):
    with mock.patch.object(
        nki_moe,
        "nki_fwd",
        _router_grad_fake_nki_fwd(score_grads),
    ):
        output = moe(inputs)
    if isinstance(output, DTensor):
        output = output.full_tensor()
    (output * output_grad).sum().backward()
    return moe.router.gate.weight.grad


def _all_gather(tensor):
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor.contiguous())
    return gathered


class TestNKIMoERouterGradUnderTP(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @property
    def device_type(self) -> str:
        return "cpu"

    def _gate_grads(self, *, enable_sp: bool, localize=None):
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=self.world_size,
            pp=1,
            ep=1,
            world_size=self.world_size,
        )
        with mock.patch(
            "torchtitan.distributed.parallel_dims.device_type",
            self.device_type,
        ):
            parallel_dims.build_mesh()

        moe = _router_grad_nki_moe(enable_sp=enable_sp)
        reference = copy.deepcopy(moe)
        moe.parallelize(parallel_dims)

        torch.manual_seed(1)
        inputs = torch.randn(
            _ROUTER_GRAD_BATCH,
            _ROUTER_GRAD_SEQ,
            _ROUTER_GRAD_DIM,
        )
        output_grad = torch.randn(
            _ROUTER_GRAD_BATCH,
            _ROUTER_GRAD_SEQ,
            _ROUTER_GRAD_DIM,
        )
        local_inputs = (
            inputs.chunk(self.world_size, dim=1)[self.rank].contiguous()
            if enable_sp
            else inputs
        )

        reference_score_grads = []
        tp_score_grads = []
        reference_grad = _router_grad_forward_backward(
            reference,
            inputs,
            output_grad,
            reference_score_grads,
        )
        label = (
            mock.patch.object(nki_moe, "_localize_gate_output", localize)
            if localize is not None
            else contextlib.nullcontext()
        )
        with label:
            tp_grad = _router_grad_forward_backward(
                moe,
                local_inputs,
                output_grad,
                tp_score_grads,
            )

        assert len(reference_score_grads) == len(tp_score_grads) == 1
        assert float(reference_grad.norm()) > 0.1
        return (
            tp_grad,
            tp_score_grads[0],
            reference_grad,
            reference_score_grads[0],
        )

    @with_comms
    def test_replicate_label_matches_the_tp1_reference(self):
        for enable_sp in (False, True):
            (
                tp_grad,
                tp_score_grad,
                reference_grad,
                reference_score_grad,
            ) = self._gate_grads(enable_sp=enable_sp)

            shares = _all_gather(tp_score_grad)
            assert not all(torch.equal(share, shares[0]) for share in shares[1:])
            torch.testing.assert_close(
                torch.stack(shares).sum(dim=0),
                reference_score_grad,
            )

            assert isinstance(tp_grad, DTensor)
            assert tuple(tp_grad.placements) == (Replicate(),)
            local_grad = tp_grad.to_local()
            ratio = float(local_grad.norm() / reference_grad.norm())
            torch.testing.assert_close(
                local_grad,
                reference_grad,
                msg=(
                    f"TP{self.world_size} sp={enable_sp}: router gate gradient "
                    f"is {ratio:.6f}x its TP=1 value"
                ),
            )
            for other in _all_gather(local_grad):
                assert torch.equal(other, local_grad)

    @with_comms
    def test_partial_label_scales_the_gradient_by_the_tp_degree(self):
        for enable_sp in (False, True):
            tp_grad, _, reference_grad, _ = self._gate_grads(
                enable_sp=enable_sp,
                localize=_partial_gate_output,
            )
            effective_grad = tp_grad.full_tensor()
            ratio = float(effective_grad.norm() / reference_grad.norm())
            torch.testing.assert_close(
                effective_grad,
                self.world_size * reference_grad,
                msg=(
                    f"TP{self.world_size} sp={enable_sp}: Partial scaled the "
                    f"router gate gradient by {ratio:.6f}x, not "
                    f"{self.world_size}x"
                ),
            )


class TestNKIMoERouterGradUnderTP4(TestNKIMoERouterGradUnderTP):
    @property
    def world_size(self) -> int:
        return 4
