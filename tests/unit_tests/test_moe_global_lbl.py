# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

"""The global-batch load-balancing loss (moe.attach_global_batch_lbl), on CPU."""

import unittest

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Partial
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.models.common.config_utils import (
    make_experts_config,
    make_moe_config,
    make_router_config,
)
from torchtitan.models.common.moe import configure_global_batch_lbl


DIM, NUM_EXPERTS, TOP_K, COEFF = 16, 8, 2, 0.5
_HAS_FX_TRACER = hasattr(torch.compiler, "_patch_engine_backward")


def _moe(load_balancing="global_aux_loss"):
    init = {"weight": lambda x: nn.init.normal_(x, std=0.5)}
    config = make_moe_config(
        num_experts=NUM_EXPERTS,
        router=make_router_config(
            dim=DIM,
            num_experts=NUM_EXPERTS,
            top_k=TOP_K,
            gate_param_init=init,
            score_func="softmax",
        ),
        experts=make_experts_config(
            dim=DIM,
            hidden_dim=2 * DIM,
            num_experts=NUM_EXPERTS,
            top_k=TOP_K,
            param_init=init,
            comm_backend="standard",
        ),
        load_balance_coeff=COEFF,
    )
    config.load_balancing = load_balancing
    torch.manual_seed(0)
    moe = config.build()
    with torch.no_grad():
        for param in moe.parameters():
            nn.init.normal_(param, std=0.5)
    return moe.train()


def _reference_fp64(moe, x_BLD, num_microbatches=1):
    """coeff * E * sum_i f_i P_i from the router alone, in fp64, as a function of the gate."""
    gate = moe.router.gate.weight.detach().double().requires_grad_()
    probs_TE = torch.softmax(x_BLD.reshape(-1, DIM).double() @ gate.t(), dim=-1)
    top_TK = probs_TE.topk(TOP_K, dim=-1).indices
    counts_E = torch.zeros(NUM_EXPERTS, dtype=torch.float64).index_add_(
        0, top_TK.flatten(), torch.ones(top_TK.numel(), dtype=torch.float64)
    )
    f_E = counts_E / counts_E.sum()
    aux = COEFF * NUM_EXPERTS * (f_E * probs_TE.mean(dim=0)).sum() / num_microbatches
    aux.backward()
    return gate.grad


def _aux_gate_grad(moe, x_BLD):
    """Gate gradient of the attached loss alone (a zero main loss)."""
    (moe(x_BLD) * 0).sum().backward()
    return moe.router.gate.weight.grad


def test_matches_the_paper_formula():
    x_BLD = torch.randn(2, 8, DIM)
    moe = _moe()
    torch.testing.assert_close(
        _aux_gate_grad(moe, x_BLD).double(),
        _reference_fp64(moe, x_BLD),
        rtol=1e-5,
        atol=1e-7,
    )


def test_micro_batches_share_the_step_normalization():
    x_BLD = torch.randn(2, 8, DIM)
    moe = _moe()
    moe.lbl_num_microbatches = 2
    torch.testing.assert_close(
        _aux_gate_grad(moe, x_BLD).double(),
        _reference_fp64(moe, x_BLD, num_microbatches=2),
        rtol=1e-5,
        atol=1e-7,
    )


def test_expert_bias_default_unchanged():
    moe = _moe("expert_bias")
    assert moe.expert_bias_E is not None
    grad = _aux_gate_grad(moe, torch.randn(2, 8, DIM))
    assert grad is None or float(grad.abs().max()) == 0.0


def test_activation_checkpointing_unchanged():
    from torch.utils.checkpoint import checkpoint

    x_BLD = torch.randn(2, 8, DIM)
    moe = _moe()
    want = _aux_gate_grad(moe, x_BLD).clone()
    for wrap in (
        lambda m, x: checkpoint(m, x, use_reentrant=False),
        lambda m, x: torch.compile(
            lambda y: checkpoint(m, y, use_reentrant=False), backend="aot_eager"
        )(x),
    ):
        moe = _moe()
        (wrap(moe, x_BLD) * 0).sum().backward()
        torch.testing.assert_close(moe.router.gate.weight.grad, want)


def test_unknown_mechanism_refused():
    with pytest.raises(ValueError, match="unknown load_balancing"):
        _moe("global_aux")


class TestTwoRanksEqualOne(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @property
    def device_type(self) -> str:
        return "cpu"

    @with_comms
    def test_router_grad(self):
        torch.manual_seed(1)
        x_BLD = torch.randn(4, 8, DIM)
        single = _moe()
        want = _aux_gate_grad(single, x_BLD)

        moe = _moe()
        moe.lbl_batch_mesh = init_device_mesh("cpu", (self.world_size,))
        grad = _aux_gate_grad(moe, x_BLD.chunk(self.world_size)[self.rank]).clone()
        # Summed over ranks, as the main loss's gradients are.
        dist.all_reduce(grad)
        assert float(grad.norm()) > 0
        torch.testing.assert_close(grad, want)

    @unittest.skipIf(
        not _HAS_FX_TRACER, "graph_trainer's minimal_fx_tracer needs torch >= 2.13"
    )
    @with_comms
    def test_router_grad_traced_by_aot_fx_trace(self):
        # The counts all-reduce runs inside the make_fx graph (functional collective).
        from torchtitan.experiments.graph_trainer.make_fx_tracer import (
            minimal_fx_tracer,
            run_traced,
        )

        torch.manual_seed(1)
        x_BLD = torch.randn(4, 8, DIM)
        want = _aux_gate_grad(_moe(), x_BLD)
        moe = _moe()
        moe.lbl_batch_mesh = init_device_mesh("cpu", (self.world_size,))

        def fwd_bwd(x):
            loss = (moe(x) * 0).sum()
            return [loss, *torch.autograd.grad(loss, [moe.router.gate.weight])]

        local = x_BLD.chunk(self.world_size)[self.rank]
        traced = minimal_fx_tracer(fwd_bwd, module=moe)(local)
        grad = run_traced(traced, module=moe)(local)[1].clone()
        dist.all_reduce(grad)
        torch.testing.assert_close(grad, want)

    @with_comms
    def test_with_a_dtensor_main_loss(self):
        # Under SimpleFSDP the CE loss is a DTensor (Partial over the batch axis).
        mesh = init_device_mesh("cpu", (self.world_size,))
        torch.manual_seed(1)
        x_BLD = torch.randn(4, 8, DIM)
        tokens = x_BLD.shape[0] * x_BLD.shape[1]
        single = _moe()
        (single(x_BLD).sum() / tokens).backward()

        moe = _moe()
        moe.lbl_batch_mesh = mesh
        out = moe(x_BLD.chunk(self.world_size)[self.rank])
        DTensor.from_local(out.sum() / tokens, mesh, [Partial()]).backward()
        grad = moe.router.gate.weight.grad.clone()
        dist.all_reduce(grad)
        torch.testing.assert_close(grad, single.router.gate.weight.grad)


def test_traced_by_aot_fx_trace():
    """GraphTrainer's aot_fx_trace traces fwd+bwd itself; the attached term must be in that graph."""
    if not _HAS_FX_TRACER:
        pytest.skip("graph_trainer's minimal_fx_tracer needs torch >= 2.13")
    from torchtitan.experiments.graph_trainer.make_fx_tracer import (
        minimal_fx_tracer,
        run_traced,
    )

    x_BLD = torch.randn(2, 8, DIM)
    want = _aux_gate_grad(_moe(), x_BLD)
    moe = _moe()

    def fwd_bwd(x):
        loss = (moe(x) * 0).sum()
        return [loss, *torch.autograd.grad(loss, [moe.router.gate.weight])]

    traced = minimal_fx_tracer(fwd_bwd, module=moe)(x_BLD)
    gate_grad = run_traced(traced, module=moe)(x_BLD)[1]
    assert float(want.norm()) > 0
    torch.testing.assert_close(gate_grad, want)


class _Stage(nn.Module):
    """One pipeline stage: a residual global-batch-LBL MoE block, as in a decoder layer."""

    def __init__(self, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.moe = _moe()

    def forward(self, x_BLD):
        return x_BLD + self.moe(x_BLD)


def _pp_loss(main_weight, global_tokens):
    # Scaled by the step's token count, as the trainer's loss is (scale_grads=False).
    def loss_fn(out_BLD, target_BLD):
        return main_weight * (out_BLD * target_BLD).sum() / global_tokens

    return loss_fn


class TestPipelineParallel(DTensorTestBase):
    """Two PP stages equal one process running the same micro-batches back to back."""

    @property
    def world_size(self) -> int:
        return 2

    @property
    def device_type(self) -> str:
        return "cpu"

    @with_comms
    def test_router_grads_match_the_unpipelined_step(self):
        from torch.distributed.pipelining import (
            PipelineStage,
            Schedule1F1B,
            ScheduleGPipe,
            ScheduleInterleaved1F1B,
            ScheduleInterleavedZeroBubble,
        )

        torch.manual_seed(3)
        x_BLD, target_BLD = torch.randn(4, 8, DIM), torch.randn(4, 8, DIM)
        tokens = x_BLD.shape[0] * x_BLD.shape[1]
        # (schedule, stages per rank); the zero-bubble schedule splits input and weight backward.
        for schedule_cls, n_local in (
            (ScheduleGPipe, 1),
            (Schedule1F1B, 1),
            (ScheduleInterleaved1F1B, 2),
            (ScheduleInterleavedZeroBubble, 2),
        ):
            n_stages = self.world_size * n_local
            stage_ids = [self.rank + i * self.world_size for i in range(n_local)]
            for n_microbatches in (2, 4):
                for main_weight in (0.0, 1.0):
                    loss_fn = _pp_loss(main_weight, tokens)
                    # Reference: every stage in one process, one forward/backward per micro-batch.
                    ref = [_Stage(i) for i in range(n_stages)]
                    configure_global_batch_lbl(
                        ref, batch_mesh=None, num_forwards=n_microbatches
                    )
                    for x, t in zip(
                        x_BLD.chunk(n_microbatches), target_BLD.chunk(n_microbatches)
                    ):
                        for s in ref:
                            x = s(x)
                        loss_fn(x, t).backward()

                    stages = [_Stage(i) for i in stage_ids]
                    configure_global_batch_lbl(
                        stages, batch_mesh=None, num_forwards=n_microbatches
                    )
                    pipe = [
                        PipelineStage(s, i, n_stages, torch.device("cpu"))
                        for s, i in zip(stages, stage_ids)
                    ]
                    schedule = schedule_cls(
                        pipe if n_local > 1 else pipe[0],
                        n_microbatches=n_microbatches,
                        loss_fn=loss_fn,
                        scale_grads=False,
                    )
                    if self.rank == 0:
                        schedule.step(x_BLD)
                    else:
                        schedule.step(target=target_BLD, losses=[])
                    for s, i in zip(stages, stage_ids):
                        want = ref[i].moe.router.gate.weight.grad
                        assert float(want.norm()) > 0
                        torch.testing.assert_close(
                            s.moe.router.gate.weight.grad,
                            want,
                            msg=schedule_cls.__name__,
                        )


if __name__ == "__main__":
    run_tests()
