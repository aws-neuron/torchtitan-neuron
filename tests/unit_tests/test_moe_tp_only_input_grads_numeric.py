# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

"""TP-only MoE input and router gradients at TP=N must match TP=1 (not ~1/N of it).

Real MoE through set_moe_sharding_config + Module.parallelize on a CPU/gloo TP mesh. Tolerance is bf16:
the experts run torch._grouped_mm in bf16, so each rank's partial sum is rounded before the reduce.
"""

import copy
from functools import partial
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor, Shard
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.models.common.config_utils import (
    make_experts_config,
    make_moe_config,
    make_router_config,
)
from torchtitan.models.common.moe import MoE
from torchtitan.models.common.moe_sharding import set_moe_sharding_config
from torchtitan.models.qwen3.sharding import _GROUPED_EXPERTS_PARAM_LAYOUT
from torchtitan.protocols.sharding import LocalMapConfig


_DIM = 16
# CPU torch._grouped_mm needs 16-byte strides, i.e. bf16 sizes that are multiples of 8. Each rank
# holds _HIDDEN_DIM / TP of the hidden dim, so 32 covers TP2 and TP4.
_HIDDEN_DIM = 32
_NUM_EXPERTS = 8
_TOP_K = 2
# Sequence parallelism needs the sequence to split evenly over TP.
_BATCH, _SEQ_LEN = 2, 8
# Bound on max|TP - TP1| / max|TP1|; bf16-limited, see the module docstring.
_TOLERANCE = 1e-2


def _moe_config(*, enable_sp: bool) -> MoE.Config:
    """A small Qwen3-style MoE (softmax router with route_norm, top-2), sharded as Qwen3 shards it."""
    cfg = make_moe_config(
        num_experts=_NUM_EXPERTS,
        router=make_router_config(
            dim=_DIM,
            num_experts=_NUM_EXPERTS,
            top_k=_TOP_K,
            gate_param_init={"weight": partial(nn.init.normal_, std=_DIM**-0.5)},
            score_func="softmax",
            route_norm=True,
        ),
        experts=make_experts_config(
            dim=_DIM,
            hidden_dim=_HIDDEN_DIM,
            num_experts=_NUM_EXPERTS,
            top_k=_TOP_K,
            param_init={
                "w1_EFD": partial(nn.init.normal_, std=_DIM**-0.5),
                "w2_EDF": partial(nn.init.normal_, std=_HIDDEN_DIM**-0.5),
                "w3_EFD": partial(nn.init.normal_, std=_DIM**-0.5),
            },
            # Falls back to local dispatch when EP is off.
            comm_backend="standard",
        ),
    )
    set_moe_sharding_config(
        cfg,
        enable_ep=False,
        enable_sp=enable_sp,
        expert_param_layout=_GROUPED_EXPERTS_PARAM_LAYOUT,
    )
    return cfg


def _error_and_share(got: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    """(max|got - ref| / max|ref|, <got, ref> / <ref, ref>): how far off, and how much of ref it holds."""
    error = (got - ref).abs().max() / ref.abs().max()
    share = (got * ref).sum() / (ref * ref).sum()
    return error.item(), share.item()


class TestTPOnlyMoEInputGrads(DTensorTestBase):
    @property
    def device_type(self) -> str:
        # The bound was measured on CPU, and what is under test is DTensor gradient bookkeeping, not
        # accelerator kernels.
        return "cpu"

    @property
    def world_size(self) -> int:
        # The smallest mesh with a TP partial to lose.
        return 2

    def _parallel_dims(self) -> ParallelDims:
        # TP only: EP, DP, CP and PP are all off.
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=self.world_size,
            pp=1,
            ep=1,
            world_size=self.world_size,
        )
        with patch(
            "torchtitan.distributed.parallel_dims.device_type", self.device_type
        ):
            parallel_dims.build_mesh()
        return parallel_dims

    def _gradients(
        self, cfg: MoE.Config, *, enable_sp: bool
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """TP1 and TP gradients of the MoE input and router gate weight, for one input and upstream grad.

        Under SP the TP module takes and returns this rank's sequence shard, so its input gradient is
        that shard and the TP1 one is sliced to match. Gate gradients are full tensors.
        """
        gen = torch.Generator().manual_seed(0)
        x_BLD = torch.randn(_BATCH, _SEQ_LEN, _DIM, generator=gen)
        grad_out_BLD = torch.randn(_BATCH, _SEQ_LEN, _DIM, generator=gen)

        torch.manual_seed(0)
        reference = cfg.build()
        reference.init_states()
        sharded = copy.deepcopy(reference)

        x_ref = x_BLD.clone().requires_grad_()
        reference(x_ref).backward(grad_out_BLD)

        parallel_dims = self._parallel_dims()
        sharded.parallelize(parallel_dims)
        # The precondition for TP partials: each rank holds a hidden-dim slice of every expert.
        self.assertEqual(
            (
                sharded.experts.w1_EFD.placements,
                sharded.experts.w2_EDF.placements,
                sharded.experts.w3_EFD.placements,
            ),
            ((Shard(1),), (Shard(2),), (Shard(1),)),
        )
        tp_rank = parallel_dims.get_mesh("tp").get_local_rank()

        def local(t: torch.Tensor) -> torch.Tensor:
            return t.chunk(self.world_size, dim=1)[tp_rank] if enable_sp else t

        x_tp = local(x_BLD).clone().requires_grad_()
        out = sharded(x_tp)
        self.assertIsInstance(out, DTensor)
        out.to_local().backward(local(grad_out_BLD))

        gate_grad = sharded.router.gate.weight.grad
        self.assertIsInstance(gate_grad, DTensor)
        reference_grads = {
            "input": local(x_ref.grad),
            "router gate weight": reference.router.gate.weight.grad,
        }
        tp_grads = {"input": x_tp.grad, "router gate weight": gate_grad.full_tensor()}
        return reference_grads, tp_grads

    @with_comms
    def test_input_and_router_gradients_match_tp1(self):
        """KILLS: the TP-only MoE handing back ~1/TP of its input and router gradients.

        The declaration from before the fix, ``in_grad_placements=None``, runs last as the negative
        control: each rank's gradients are then far from TP1, yet they sum over TP to it, i.e. the
        partials were computed and never reduced. Measured share of the TP1 input gradient,
        <g_rank, g_TP1> / <g_TP1, g_TP1>: 0.59/0.41 at TP2 and 0.27/0.32/0.21/0.20 at TP4.
        """
        for enable_sp in (False, True):
            reference, sharded = self._gradients(
                _moe_config(enable_sp=enable_sp), enable_sp=enable_sp
            )
            for name, ref in reference.items():
                error, share = _error_and_share(sharded[name], ref)
                self.assertLess(
                    error,
                    _TOLERANCE,
                    f"{name} gradient at TP{self.world_size} (enable_sp={enable_sp}) is off by "
                    f"{error:.3g} of its magnitude and holds {share:.3f} of the TP1 gradient; "
                    "~1/TP means the routed experts' TP partials were never summed",
                )

        # Negative control, without SP so that each rank's gradient is exactly its own partial.
        cfg = _moe_config(enable_sp=False)
        cfg.experts.sharding_config.local_map = LocalMapConfig(in_grad_placements=None)
        reference, sharded = self._gradients(cfg, enable_sp=False)
        for name, ref in reference.items():
            error, share = _error_and_share(sharded[name], ref)
            # Ten times the tolerance: a missing partial, not rounding.
            self.assertGreater(
                error,
                10 * _TOLERANCE,
                f"{name} gradient matches TP1 (share {share:.3f}) without in_grad_placements; if "
                "local_map now infers Partial, this test no longer proves the fix is needed",
            )
            # TP only, so the default process group is the TP group.
            total = sharded[name].clone()
            dist.all_reduce(total)
            total_error, _ = _error_and_share(total, ref)
            self.assertLess(
                total_error,
                _TOLERANCE,
                f"{name} gradients without in_grad_placements do not sum over TP to TP1",
            )


class TestTPOnlyMoEInputGradsAtFour(TestTPOnlyMoEInputGrads):
    @property
    def world_size(self) -> int:
        # The TP degree of the GPU runs where every ffn_norm gradient came out 0.26-0.28x its TP1 value.
        return 4


if __name__ == "__main__":
    run_tests()
