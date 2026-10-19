# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

"""DeepSeek V3 NKI MoE integration for Neuron."""

from dataclasses import dataclass
from typing import Literal

import nki.language as nl

import spmd_types as spmd
import torch
import torch.nn.functional as F
from nkilib.core.subkernels.find_nonzero_indices import find_nonzero_indices
from nkilib.core.subkernels.indexed_flatten import indexed_flatten
from nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode
from nkilib.experimental.moe.bwd.blockwise_mm_backward import blockwise_mm_bwd
from nkilib.experimental.moe.bwd.moe_bwd_parameters import (
    AffinityOption,
    ShardOption,
    SkipMode as BwdSkipMode,
)
from nkilib.experimental.moe.forward.bwmm_shard_on_H import (
    blockwise_mm_baseline_shard_hidden,
    SkipMode as FwdSkipMode,
)
from torch import nn
from torch.distributed.tensor import DTensor, Partial, Replicate
from torch_neuronx import nki_op
from torch_neuronx.nki_hop import wrap_nki
from torch_neuronx.utils import get_logical_neuron_cores

from torchtitan.models.common import FeedForward, Linear
from torchtitan.models.common.decoder_sharding import (
    dense_activation_placement,
    dense_param_placement,
    dense_sequence_parallel_placement,
)
from torchtitan.models.common.moe import GroupedExperts, MoE, TokenChoiceTopKRouter
from torchtitan.protocols.module import Module
from torchtitan.protocols.sharding import LocalMapConfig, ShardingConfig


BLOCK_SIZE = 512
LNC = (int(get_logical_neuron_cores()),)

wrapped_moe_fwd = wrap_nki(blockwise_mm_baseline_shard_hidden)
wrapped_moe_bwd = wrap_nki(blockwise_mm_bwd)
wrapped_find_nonzero = wrap_nki(find_nonzero_indices)
wrapped_indexed_flatten = wrap_nki(indexed_flatten)


def _debug_force_load_balance_masked_scores(
    scores: torch.Tensor,
    top_k: int,
    num_experts: int,
    route_norm: bool,
    route_scale: float,
) -> torch.Tensor:
    n_tokens = scores.shape[0]
    indices = (
        torch.arange(n_tokens * top_k, device=scores.device, dtype=torch.int64).reshape(
            n_tokens, top_k
        )
        % num_experts
    )
    top_scores = scores.gather(dim=1, index=indices)
    if route_norm:
        top_scores = top_scores / (top_scores.sum(dim=-1, keepdim=True) + 1e-20)
    top_scores = top_scores * route_scale

    masked_scores = torch.zeros_like(scores)
    masked_scores.scatter_(1, indices, top_scores)
    return masked_scores


@nki_op("deepseek_v3_nki_moe::nki_moe_fwd")
def nki_fwd(
    hidden_states: torch.Tensor,
    masked_scores: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    top_k: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    int,
    torch.Tensor,
]:
    tokens = hidden_states.shape[0]
    num_experts = masked_scores.shape[1]
    block_size = BLOCK_SIZE
    num_blocks = (tokens * top_k + block_size - 1) // block_size + num_experts

    if num_experts == 1:
        token_position_to_id = torch.arange(
            num_blocks * block_size,
            device=hidden_states.device,
            dtype=torch.int32,
        )
        token_position_to_id = torch.where(
            token_position_to_id < tokens,
            token_position_to_id,
            -1,
        )
        block_to_expert = torch.zeros(
            (num_blocks, 1),
            device=hidden_states.device,
            dtype=torch.int32,
        )
    else:
        indices, counts = wrapped_find_nonzero[LNC](masked_scores.float())
        blocks_per_expert = ((counts + block_size - 1) // block_size).clamp(min=1)
        slack = num_blocks - blocks_per_expert.sum()
        padded_blocks_per_expert = blocks_per_expert.clone()
        padded_blocks_per_expert[-1] += slack
        row_offsets = (
            torch.cumsum(padded_blocks_per_expert, dim=0) - padded_blocks_per_expert
        )
        expert_ids = torch.arange(
            num_experts, device=hidden_states.device, dtype=torch.int32
        )
        block_to_expert = torch.repeat_interleave(
            expert_ids,
            padded_blocks_per_expert,
            output_size=num_blocks,
        ).unsqueeze(1)
        token_position_to_id = wrapped_indexed_flatten[LNC](
            indices,
            block_size,
            num_blocks * block_size,
            row_offsets,
            padding_val=-1,
        )

    affinities = masked_scores.reshape(-1, 1).clone()
    gate_up_proj_weight = torch.stack(
        (w1.transpose(-2, -1), w3.transpose(-2, -1)),
        dim=2,
    )
    down_proj_weight = w2.transpose(-2, -1)
    hidden_dim_tp = down_proj_weight.shape[1]
    gate_up_activations = torch.empty(
        (num_blocks, 2, hidden_dim_tp, block_size),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    down_activations = None
    output, gate_up_activations = wrapped_moe_fwd[LNC](
        hidden_states,
        affinities,
        gate_up_proj_weight,
        down_proj_weight,
        block_size,
        token_position_to_id,
        block_to_expert,
        gate_up_activations,
        down_activations,
        skip_dma=FwdSkipMode(skip_token=True),
        is_tensor_update_accumulating=True,
        expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
        # The kernel sums the top-k expert outputs in bf16 unless told otherwise.
        accum_dtype=nl.float32,
    )

    return (
        affinities,
        gate_up_activations,
        down_activations,
        token_position_to_id,
        block_to_expert,
        tokens,
        num_experts,
        output,
    )


@nki_op(
    "deepseek_v3_nki_moe::nki_moe_bwd",
    mutates_args={
        "hidden_states_grad_out",
        "expert_affinities_grad_out",
        "gate_up_proj_weight_grad_out",
        "down_proj_weight_grad_out",
    },
)
def nki_bwd(
    hidden_states: torch.Tensor,
    affinities: torch.Tensor,
    gate_up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    gate_up_activations: torch.Tensor,
    down_activations: torch.Tensor,
    token_position_to_id: torch.Tensor,
    block_to_expert: torch.Tensor,
    grad_output: torch.Tensor,
    hidden_states_grad_out: torch.Tensor,
    expert_affinities_grad_out: torch.Tensor,
    gate_up_proj_weight_grad_out: torch.Tensor,
    down_proj_weight_grad_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return wrapped_moe_bwd[LNC](
        hidden_states,
        affinities,
        gate_up_proj_weight,
        down_proj_weight,
        gate_up_activations,
        down_activations,
        token_position_to_id,
        block_to_expert,
        grad_output,
        BLOCK_SIZE,
        skip_dma=BwdSkipMode(skip_token=True),
        skip_grad_initialization=True,
        activation_type=ActFnType.SiLU,
        affinity_option=AffinityOption.AFFINITY_ON_I,
        shard_option=ShardOption.SHARD_ON_HIDDEN,
        bias=False,
        hidden_states_grad_out=hidden_states_grad_out,
        expert_affinities_masked_grad_out=expert_affinities_grad_out,
        gate_up_proj_weight_grad_out=gate_up_proj_weight_grad_out,
        down_proj_weight_grad_out=down_proj_weight_grad_out,
    )


def _setup_context(ctx, inputs, output):
    hidden_states, masked_scores, w1, w2, w3, _ = inputs
    (
        affinities,
        gate_up_activations,
        down_activations,
        token_position_to_id,
        block_to_expert,
        tokens,
        num_experts,
        _,
    ) = output
    ctx.save_for_backward(
        hidden_states,
        affinities,
        w1,
        w2,
        w3,
        gate_up_activations,
        down_activations,
        token_position_to_id,
        block_to_expert,
    )
    ctx.tokens = tokens
    ctx.num_experts = num_experts
    ctx.scores_dtype = masked_scores.dtype


def _backward(ctx, *grad_outputs):
    grad_output = grad_outputs[-1]
    (
        hidden_states,
        affinities,
        w1,
        w2,
        w3,
        gate_up_activations,
        down_activations,
        token_position_to_id,
        block_to_expert,
    ) = ctx.saved_tensors

    gate_up_proj_weight = torch.stack(
        (w1.transpose(-2, -1), w3.transpose(-2, -1)),
        dim=2,
    )
    down_proj_weight = w2.transpose(-2, -1)

    hidden_states_grad_out = torch.zeros_like(hidden_states, dtype=torch.float32)
    expert_affinities_grad_out = torch.zeros_like(affinities)
    gate_up_proj_weight_grad_out = torch.zeros_like(
        gate_up_proj_weight, dtype=torch.float32
    )
    down_proj_weight_grad_out = torch.zeros_like(down_proj_weight, dtype=torch.float32)

    (
        hidden_states_grad,
        affinities_grad,
        gate_up_proj_weight_grad,
        down_proj_weight_grad,
    ) = nki_bwd(
        hidden_states,
        affinities,
        gate_up_proj_weight,
        down_proj_weight,
        gate_up_activations,
        down_activations,
        token_position_to_id,
        block_to_expert,
        grad_output[: ctx.tokens],
        hidden_states_grad_out,
        expert_affinities_grad_out,
        gate_up_proj_weight_grad_out,
        down_proj_weight_grad_out,
    )

    hidden_states_grad = hidden_states_grad.to(hidden_states.dtype)
    gate_up_proj_weight_grad = gate_up_proj_weight_grad.to(gate_up_proj_weight.dtype)
    down_proj_weight_grad = down_proj_weight_grad.to(down_proj_weight.dtype)
    scores_grad = affinities_grad.reshape(-1, ctx.num_experts)[: ctx.tokens].to(
        ctx.scores_dtype
    )
    w1_grad, w3_grad = gate_up_proj_weight_grad.unbind(dim=2)

    return (
        hidden_states_grad[: ctx.tokens],
        scores_grad,
        w1_grad.transpose(-2, -1),
        down_proj_weight_grad.transpose(-2, -1),
        w3_grad.transpose(-2, -1),
        None,
    )


nki_fwd.register_autograd(_backward, setup_context=_setup_context)


def _local(t):
    """Plain local view of a (Replicate) DTensor buffer/param for the local MoE region."""
    return t.to_local() if isinstance(t, DTensor) else t


def _localize_gate_output(scores: torch.Tensor) -> torch.Tensor:
    """Localize the router output with a Replicate local gradient.

    The experts' from_local(Replicate) input wrap already all-reduces the per-rank score
    gradients; a Partial label here would sum them again (gate gradient TP x too large)."""
    if isinstance(scores, DTensor):
        return scores.to_local(
            grad_placements=tuple(Replicate() for _ in scores.placements)
        )
    return scores


class MaskedScoresTopKRouter(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        gate: Linear.Config
        top_k: int
        score_func: Literal["softmax", "sigmoid"]
        route_norm: bool = False
        route_scale: float = 1.0
        _debug_force_load_balance: bool = False

    def __init__(self, config: Config):
        super().__init__()
        self.gate = config.gate.build()
        self.top_k = config.top_k
        self.score_func = config.score_func
        self.route_norm = config.route_norm
        self.route_scale = config.route_scale
        self._debug_force_load_balance = config._debug_force_load_balance

    def forward(
        self,
        x: torch.Tensor,
        expert_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, dtype=torch.float32):
            scores = _localize_gate_output(self.gate(x))
        if self.score_func == "softmax":
            scores = F.softmax(scores, dim=-1)
        elif self.score_func == "sigmoid":
            scores = torch.sigmoid(scores)
        else:
            raise ValueError(f"Unsupported score function: {self.score_func}")

        scores_for_choice = scores if expert_bias is None else scores + expert_bias
        top_indices = torch.topk(
            scores_for_choice,
            k=self.top_k,
            dim=-1,
            sorted=False,
        ).indices
        mask = torch.zeros_like(scores_for_choice, dtype=torch.bool)
        mask.scatter_(-1, top_indices, True)
        masked_scores = scores * mask
        if self.route_norm:
            masked_scores = masked_scores / (
                masked_scores.sum(dim=-1, keepdim=True) + 1e-20
            )
        return masked_scores * self.route_scale


class NKIGroupedExperts(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        hidden_dim: int
        num_experts: int
        top_k: int
        token_dispatcher: object | None = None

    def __init__(self, config: Config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.w1 = nn.Parameter(
            torch.empty(config.num_experts, config.hidden_dim, config.dim)
        )
        self.w2 = nn.Parameter(
            torch.empty(config.num_experts, config.dim, config.hidden_dim)
        )
        self.w3 = nn.Parameter(
            torch.empty(config.num_experts, config.hidden_dim, config.dim)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        masked_scores: torch.Tensor,
    ) -> torch.Tensor:
        w1 = self.w1
        w2 = self.w2
        w3 = self.w3
        if isinstance(w1, DTensor):
            w1 = w1.to_local()
            w2 = w2.to_local()
            w3 = w3.to_local()
        (*_, tokens, _, output) = nki_fwd(
            hidden_states,
            masked_scores,
            w1,
            w2,
            w3,
            self.top_k,
        )
        return output[:tokens]


class NKIMoE(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_experts: int
        router: MaskedScoresTopKRouter.Config
        experts: NKIGroupedExperts.Config
        load_balance_coeff: float | None = 1e-3
        shared_experts: FeedForward.Config | None = None

        def set_sharding_config(self, *, enable_sp: bool, enable_ep: bool) -> None:
            if enable_ep:
                raise NotImplementedError(
                    "DeepSeek V3 NKI MoE supports TP-only experts"
                )

            activation = (
                dense_sequence_parallel_placement()
                if enable_sp
                else dense_activation_placement(tp=spmd.I)
            )
            replicated = dense_activation_placement(tp=spmd.R)
            partial = dense_activation_placement(tp=spmd.P)
            self.sharding_config = ShardingConfig(
                state_shardings={
                    "expert_bias_E": dense_param_placement(tp=spmd.R),
                    "tokens_per_expert_E": dense_param_placement(tp=spmd.R),
                },
                in_src_shardings={"x": activation},
                in_dst_shardings={"x": replicated},
                out_src_shardings=partial,
                out_dst_shardings=activation,
            )
            self.router.gate.sharding_config = ShardingConfig(
                state_shardings={
                    "weight": dense_param_placement(tp=spmd.R),
                    "bias": dense_param_placement(tp=spmd.R),
                },
                in_src_shardings={"input": replicated},
                out_src_shardings=replicated,
            )
            self.experts.sharding_config = ShardingConfig(
                state_shardings={
                    "w1": dense_param_placement(tp=spmd.S(1)),
                    "w2": dense_param_placement(tp=spmd.S(2)),
                    "w3": dense_param_placement(tp=spmd.S(1)),
                },
                in_src_shardings={
                    "hidden_states": replicated,
                    "masked_scores": replicated,
                },
                in_dst_shardings={
                    "hidden_states": replicated,
                    "masked_scores": replicated,
                },
                out_src_shardings=partial,
                out_dst_shardings=partial,
                local_map=LocalMapConfig(in_grad_placements=(partial, partial)),
            )
            if self.shared_experts is not None:
                shared_hidden = dense_activation_placement(tp=spmd.S(1), cp=spmd.R)
                self.shared_experts.sharding_config = ShardingConfig(
                    in_src_shardings={"x": replicated},
                    in_dst_shardings={"x": replicated},
                )
                self.shared_experts.w1.sharding_config = ShardingConfig(
                    state_shardings={
                        "weight": dense_param_placement(tp=spmd.S(0)),
                        "bias": dense_param_placement(tp=spmd.S(0)),
                    },
                    in_src_shardings={"input": replicated},
                    in_dst_shardings={"input": replicated},
                    out_src_shardings=shared_hidden,
                    out_dst_shardings=shared_hidden,
                )
                self.shared_experts.w2.sharding_config = ShardingConfig(
                    state_shardings={
                        "weight": dense_param_placement(tp=spmd.S(1)),
                        "bias": dense_param_placement(tp=spmd.R),
                    },
                    in_src_shardings={"input": shared_hidden},
                    out_src_shardings=partial,
                    out_dst_shardings=partial,
                )
                self.shared_experts.w3.sharding_config = (
                    self.shared_experts.w1.sharding_config
                )

    def __init__(self, config: Config):
        super().__init__()
        self.num_experts = config.num_experts
        self.router = config.router.build()
        self.experts = config.experts.build()
        self.shared_experts = (
            config.shared_experts.build() if config.shared_experts is not None else None
        )
        self.load_balance_coeff = config.load_balance_coeff
        self.moe_enabled = True
        if self.load_balance_coeff is not None:
            self.register_buffer(
                "expert_bias_E",
                torch.zeros(self.num_experts, dtype=torch.float32),
                persistent=True,
            )
        else:
            self.expert_bias_E = None
        self.register_buffer(
            "tokens_per_expert_E",
            torch.zeros(self.num_experts, dtype=torch.float32),
            persistent=False,
        )

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        if buffer_device is None:
            buffer_device = self.tokens_per_expert_E.device

        with torch.device(buffer_device):
            self.tokens_per_expert_E = torch.zeros(
                self.num_experts, dtype=torch.float32
            )
            if self.load_balance_coeff is not None:
                self.expert_bias_E = torch.zeros(self.num_experts, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Run router + experts on plain local tensors. Each sub-module's
        # ShardingConfig re-wraps its plain input (DTensor.from_local on in_src)
        # and that wrap's backward all-reduces its Partial input gradient, so the
        # gradient arriving here is already fully reduced: localize with a plain
        # to_local() (declaring it Partial would sum it a second time).
        spec = None
        if isinstance(x, DTensor):
            spec = (x.device_mesh, x.placements)
            x = x.to_local()
        batch, sequence, dim = x.shape
        flat = x.reshape(-1, dim)
        if self.router._debug_force_load_balance:
            with torch.autocast(device_type=flat.device.type, dtype=torch.float32):
                raw_scores = _localize_gate_output(self.router.gate(flat))
            if self.router.score_func == "sigmoid":
                raw_scores = torch.sigmoid(raw_scores)
            else:
                raw_scores = F.softmax(raw_scores, dim=-1)
            masked_scores = _debug_force_load_balance_masked_scores(
                raw_scores,
                self.router.top_k,
                self.num_experts,
                self.router.route_norm,
                self.router.route_scale,
            )
        else:
            masked_scores = _localize_gate_output(
                self.router(flat, _local(self.expert_bias_E))
            )

        with torch.no_grad():
            _local(self.tokens_per_expert_E).add_(
                (masked_scores != 0).to(torch.float32).sum(dim=0)
            )

        # Both expert paths return Partial DTensors (their ShardingConfig out_src);
        # a plain to_local() labels their gradient Replicate, which is the correct
        # gradient placement for a Partial output.
        routed_output = _local(self.experts(flat, masked_scores))
        if self.shared_experts is not None:
            routed_output = routed_output + _local(self.shared_experts(flat))
        routed_output = routed_output.reshape(batch, sequence, dim)
        if spec is not None:
            routed_output = DTensor.from_local(
                routed_output,
                spec[0],
                tuple(Partial() for _ in spec[1]),
                run_check=False,
            )
        return routed_output


def to_nki_moe_config(moe: MoE.Config) -> NKIMoE.Config:
    if not isinstance(moe.router, TokenChoiceTopKRouter.Config):
        raise TypeError(
            f"Expected TokenChoiceTopKRouter.Config, got {type(moe.router)!r}"
        )
    if not isinstance(moe.experts, GroupedExperts.Config):
        raise TypeError(f"Expected GroupedExperts.Config, got {type(moe.experts)!r}")
    if (
        moe.router.num_expert_groups is not None
        or moe.router.num_limited_groups is not None
    ):
        raise NotImplementedError(
            "DeepSeek V3 NKI MoE does not support group-limited routing"
        )

    param_init = moe.experts.param_init or {}
    nki_param_init = {
        key: param_init[src]
        for key, src in (("w1", "w1_EFD"), ("w2", "w2_EDF"), ("w3", "w3_EFD"))
        if param_init.get(src) is not None
    }
    return NKIMoE.Config(
        num_experts=moe.num_experts,
        load_balance_coeff=moe.load_balance_coeff,
        router=MaskedScoresTopKRouter.Config(
            gate=moe.router.gate,
            top_k=moe.router.top_k,
            score_func=moe.router.score_func,
            route_norm=moe.router.route_norm,
            route_scale=moe.router.route_scale,
            _debug_force_load_balance=moe.router._debug_force_load_balance,
        ),
        experts=NKIGroupedExperts.Config(
            dim=moe.experts.dim,
            hidden_dim=moe.experts.hidden_dim,
            num_experts=moe.experts.num_experts,
            top_k=moe.router.top_k,
            param_init=nki_param_init,
        ),
        shared_experts=moe.shared_experts,
    )
