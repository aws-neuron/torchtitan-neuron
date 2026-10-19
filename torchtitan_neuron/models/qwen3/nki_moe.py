# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

"""Qwen3 NKI MoE implementation for the Neuron extension package."""

from collections.abc import Callable
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
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Partial, Replicate
from torch_neuronx import nki_op
from torch_neuronx.nki_hop import wrap_nki
from torch_neuronx.utils import get_logical_neuron_cores

from torchtitan.models.common import Linear
from torchtitan.models.common.decoder_sharding import (
    dense_activation_placement,
    dense_param_placement,
    dense_sequence_parallel_placement,
)
from torchtitan.models.common.moe import (
    attach_global_batch_lbl,
    GroupedExperts,
    MoE,
    TokenChoiceTopKRouter,
)
from torchtitan.protocols.module import Module
from torchtitan.protocols.sharding import LocalMapConfig, ShardingConfig


BLOCK_SIZE = 512
EP_BLOCK_SIZE = 256
LNC = (int(get_logical_neuron_cores()),)

wrapped_moe_fwd = wrap_nki(blockwise_mm_baseline_shard_hidden)
wrapped_moe_bwd = wrap_nki(blockwise_mm_bwd)
wrapped_find_nonzero = wrap_nki(find_nonzero_indices)
wrapped_indexed_flatten = wrap_nki(indexed_flatten)


def _single_expert_block_metadata(
    tokens: int,
    num_blocks: int,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_position_to_id = torch.arange(
        num_blocks * block_size,
        device=device,
        dtype=torch.int32,
    )
    token_position_to_id = torch.where(
        token_position_to_id < tokens,
        token_position_to_id,
        -1,
    )
    block_to_expert = torch.zeros(
        (num_blocks, 1),
        device=device,
        dtype=torch.int32,
    )
    return token_position_to_id, block_to_expert


@nki_op("qwen3_nki_moe::nki_moe_fwd")
def nki_fwd(
    hidden_states: torch.Tensor,
    masked_scores: torch.Tensor,
    gate_up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    top_k: int,
    block_size: int,
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
    hidden_dim_tp = down_proj_weight.shape[1]
    # At most one affinity can exist per local expert for a token row. For
    # neuron_ep, masked_scores describes the post-dispatch local experts.
    max_affinities_per_row = min(top_k, num_experts)
    num_blocks = (
        tokens * max_affinities_per_row + block_size - 1
    ) // block_size + num_experts

    if num_experts == 1:
        # find_nonzero_indices requires the expert dimension to be divisible
        # across LNC2. Dispatched nle=1 inputs are already expert-major, so the
        # block metadata is an identity mapping and needs no NKI preprocessing.
        token_position_to_id, block_to_expert = _single_expert_block_metadata(
            tokens,
            num_blocks,
            block_size,
            hidden_states.device,
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
        # Repeat expert ids by padded block count. Repeating the counts themselves
        # can create invalid expert ids and drive the NKI gather out of bounds.
        block_to_expert = torch.repeat_interleave(
            expert_ids, padded_blocks_per_expert, output_size=num_blocks
        ).unsqueeze(1)
        token_position_to_id = wrapped_indexed_flatten[LNC](
            indices,
            block_size,
            num_blocks * block_size,
            row_offsets,
            padding_val=-1,
        )

    affinities = masked_scores.reshape(-1, 1).clone()
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

    # Inputs are recovered from setup_context. Returning them would violate
    # the custom-op no-alias contract enforced by PT2.12.
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
    "qwen3_nki_moe::nki_moe_bwd",
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
    block_size: int,
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
        block_size,
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
    (
        hidden_states,
        masked_scores,
        gate_up_proj_weight,
        down_proj_weight,
        _,
        block_size,
    ) = inputs
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
        gate_up_proj_weight,
        down_proj_weight,
        gate_up_activations,
        down_activations,
        token_position_to_id,
        block_to_expert,
    )
    ctx.tokens = tokens
    ctx.num_experts = num_experts
    ctx.scores_dtype = masked_scores.dtype
    ctx.block_size = block_size


def _backward(ctx, *grad_outputs):
    grad_output = grad_outputs[-1]
    (
        hidden_states,
        affinities,
        gate_up_proj_weight,
        down_proj_weight,
        gate_up_activations,
        down_activations,
        token_position_to_id,
        block_to_expert,
    ) = ctx.saved_tensors

    hidden_states_dtype = hidden_states.dtype
    gate_up_proj_weight_dtype = gate_up_proj_weight.dtype
    down_proj_weight_dtype = down_proj_weight.dtype

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
        ctx.block_size,
    )
    hidden_states_grad = hidden_states_grad.to(hidden_states_dtype)
    gate_up_proj_weight_grad = gate_up_proj_weight_grad.to(gate_up_proj_weight_dtype)
    down_proj_weight_grad = down_proj_weight_grad.to(down_proj_weight_dtype)

    scores_grad = affinities_grad.reshape(-1, ctx.num_experts)[: ctx.tokens].to(
        ctx.scores_dtype
    )
    return (
        hidden_states_grad[: ctx.tokens],
        scores_grad,
        gate_up_proj_weight_grad,
        down_proj_weight_grad,
        None,
        None,
    )


nki_fwd.register_autograd(_backward, setup_context=_setup_context)


def _debug_force_load_balance_masked_scores(
    scores: torch.Tensor,
    top_k: int,
    num_experts: int,
    route_norm: bool,
    route_scale: float,
) -> torch.Tensor:
    """Round-robin expert assignment with real gate scores (debug only).

    Same deterministic assignment as the GPU/std-MoE debug helper:
    expert(token i, slot k) = (i * top_k + k) % num_experts.
    """
    # DTensor inputs crash torch.Tensor.gather with a mixed DTensor/plain
    # index. Compute on the local tensor and rewrap with the original
    # (replicated) placements, which is exact for replicated scores.
    _dtensor_spec = None
    if isinstance(scores, DTensor):
        _dtensor_spec = (scores.device_mesh, scores.placements)
        scores = scores.to_local()
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
    if _dtensor_spec is not None:
        masked_scores = DTensor.from_local(
            masked_scores, _dtensor_spec[0], _dtensor_spec[1]
        )
    return masked_scores


class MaskedScoresTopKRouter(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        gate: Linear.Config
        top_k: int = 8
        score_func: Literal["softmax", "sigmoid"] = "softmax"
        route_norm: bool = True
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
        return_scores: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Masked routing scores; with ``return_scores`` also the full unbiased scores."""
        with torch.autocast(device_type=x.device.type, dtype=torch.float32):
            scores = self.gate(x)
        if self.score_func == "softmax":
            scores = F.softmax(scores, dim=-1)
        elif self.score_func == "sigmoid":
            scores = torch.sigmoid(scores)
        else:
            raise ValueError(f"Unsupported score function: {self.score_func}")

        # Auxiliary-loss-free load balancing (https://arxiv.org/abs/2408.15664):
        # the expert bias only steers expert CHOICE; gating values stay the
        # unbiased scores. Same contract as std MoE / DeepSeek NKI MoE.
        scores_for_choice = scores if expert_bias is None else scores + expert_bias
        top_indices = torch.topk(
            scores_for_choice, k=self.top_k, dim=-1, sorted=False
        ).indices
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask.scatter_(-1, top_indices, True)
        masked_scores = scores * mask
        if self.route_norm:
            masked_scores /= masked_scores.sum(dim=-1, keepdim=True) + 1e-20
        masked_scores = masked_scores * self.route_scale
        return (masked_scores, scores) if return_scores else masked_scores


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
        self.gate_up_proj_weight = nn.Parameter(
            torch.empty(config.num_experts, config.dim, 2, config.hidden_dim)
        )
        self.down_proj_weight = nn.Parameter(
            torch.empty(config.num_experts, config.hidden_dim, config.dim)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        masked_scores: torch.Tensor,
    ) -> torch.Tensor:
        gate_up_proj_weight = self.gate_up_proj_weight
        down_proj_weight = self.down_proj_weight
        if isinstance(gate_up_proj_weight, DTensor):
            gate_up_proj_weight = gate_up_proj_weight.to_local()
            down_proj_weight = down_proj_weight.to_local()
        (*_, tokens, _, output) = nki_fwd(
            hidden_states,
            masked_scores,
            gate_up_proj_weight,
            down_proj_weight,
            self.top_k,
            BLOCK_SIZE,
        )
        return output[:tokens]


class BWMMGroupedExperts(GroupedExperts):
    """GroupedExperts using BWMM after the neuron_ep dispatch.

    The dispatcher returns expert-major local blocks. A one-hot local affinity
    matrix routes each block to its corresponding packed expert weights; router
    affinities remain applied by the dispatcher during combine.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(GroupedExperts.Config):
        pass

    def _experts_forward(
        self,
        x_RD: torch.Tensor,
        num_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(self.w1_EFD, DTensor):
            w1_EFD = self.w1_EFD.to_local()
            assert isinstance(self.w2_EDF, DTensor)
            w2_EDF = self.w2_EDF.to_local()
            assert isinstance(self.w3_EFD, DTensor)
            w3_EFD = self.w3_EFD.to_local()
        else:
            w1_EFD = self.w1_EFD
            w2_EDF = self.w2_EDF
            w3_EFD = self.w3_EFD

        local_experts = w1_EFD.shape[0]
        if num_tokens_per_expert_E.numel() != local_experts:
            raise ValueError(
                "BWMMGroupedExperts requires one token count for each local "
                f"expert; got {local_experts} local weight shards and "
                f"{num_tokens_per_expert_E.numel()} token counts."
            )

        # BWMM stores gate/up together as [E, D, 2, H] and down as [E, H, D].
        gate_up_proj_weight = torch.stack(
            (w1_EFD.transpose(-2, -1), w3_EFD.transpose(-2, -1)),
            dim=2,
        ).bfloat16()
        down_proj_weight = w2_EDF.transpose(-2, -1).bfloat16()

        # TODO: Validate the nle>1 neuron_ep + BWMM path on hardware.
        expert_ends_E = num_tokens_per_expert_E.cumsum(dim=0)
        expert_starts_E = expert_ends_E - num_tokens_per_expert_E
        rows_R1 = torch.arange(x_RD.shape[0], device=x_RD.device).unsqueeze(1)
        local_affinities = (
            (rows_R1 >= expert_starts_E.unsqueeze(0))
            & (rows_R1 < expert_ends_E.unsqueeze(0))
        ).to(torch.float32)
        (*_, tokens, _, output) = nki_fwd(
            x_RD.bfloat16(),
            local_affinities,
            gate_up_proj_weight,
            down_proj_weight,
            1,
            EP_BLOCK_SIZE,
        )
        return output[:tokens].type_as(x_RD)


def _local(t):
    """Plain local view of a (Replicate) DTensor buffer for the local MoE region."""
    return t.to_local() if isinstance(t, DTensor) else t


def _localize_gate_output(scores):
    """Localize the router output with a Replicate local gradient.

    The experts' from_local(Replicate) input wrap already all-reduces the per-rank score
    gradients; a Partial label here would sum them again (gate gradient TP x too large)."""
    if isinstance(scores, DTensor):
        return scores.to_local(
            grad_placements=tuple(Replicate() for _ in scores.placements)
        )
    return scores


class NKIMoE(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_experts: int
        router: MaskedScoresTopKRouter.Config
        experts: NKIGroupedExperts.Config
        load_balance_coeff: float | None = 1e-3
        # As MoE.Config.load_balancing (models/common/moe.py).
        load_balancing: Literal["expert_bias", "global_aux_loss"] = "expert_bias"
        shared_experts: None = None

        def set_sharding_config(self, *, enable_sp: bool, enable_ep: bool) -> None:
            if enable_ep:
                raise NotImplementedError("Qwen3 NKI MoE supports TP-only experts")

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
                    "gate_up_proj_weight": dense_param_placement(tp=spmd.S(3)),
                    "down_proj_weight": dense_param_placement(tp=spmd.S(1)),
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
                local_map=LocalMapConfig(
                    in_grad_placements=(partial, partial),
                ),
            )

    def __init__(self, config: Config):
        super().__init__()
        self.num_experts = config.num_experts
        self.router = config.router.build()
        self.experts = config.experts.build()
        self.load_balance_coeff = config.load_balance_coeff
        self.load_balancing = config.load_balancing
        if self.load_balancing not in ("expert_bias", "global_aux_loss"):
            raise ValueError(f"unknown load_balancing {self.load_balancing!r}")
        self.lbl_batch_mesh: DeviceMesh | None = None
        self.lbl_num_microbatches: int = 1
        self.shared_experts = None
        # Auxiliary-loss-free load balancing state. tokens_per_expert_E is
        # accumulated in forward; expert_bias_E is updated by the
        # OptimizersContainer step pre-hook (_update_expert_bias), exactly as for
        # the std MoE the GPU/reference baseline runs and the DeepSeek NKI MoE.
        if self.load_balance_coeff is not None:
            assert self.load_balance_coeff > 0.0
        if self.load_balance_coeff is not None and self.load_balancing == "expert_bias":
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
            if self.expert_bias_E is not None:
                self.expert_bias_E = torch.zeros(self.num_experts, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Run the whole MoE (router + experts) on local tensors. The wrapper's
        # ShardingConfig has already redistributed x to Replicate on TP. The
        # gate and experts sub-modules re-wrap this plain input through their
        # own ShardingConfig (DTensor.from_local on in_src), and that wrap's
        # backward all-reduces their Partial input gradients -- so the gradient
        # arriving here is already fully reduced. Localize with a plain
        # to_local(): declaring it Partial would sum it a second time (4x).
        spec = None
        if isinstance(x, DTensor):
            spec = (x.device_mesh, x.placements)
            x = x.to_local()
        batch, sequence, dim = x.shape
        flat = x.reshape(-1, dim)
        lbl = (
            self.load_balancing == "global_aux_loss"
            and self.load_balance_coeff is not None
            and torch.is_grad_enabled()
        )
        if self.router._debug_force_load_balance:
            with torch.autocast(device_type=flat.device.type, dtype=torch.float32):
                raw_scores = self.router.gate(flat)
            raw_scores = _localize_gate_output(raw_scores)
            if self.router.score_func == "sigmoid":
                raw_scores = torch.sigmoid(raw_scores)
            else:
                raw_scores = F.softmax(raw_scores, dim=1)
            masked_scores = _debug_force_load_balance_masked_scores(
                raw_scores,
                self.router.top_k,
                self.num_experts,
                self.router.route_norm,
                self.router.route_scale,
            )
            if lbl:
                lbl_stats = ((masked_scores != 0).sum(dim=0), raw_scores.sum(dim=0))
        else:
            # Pass expert_bias_E as is: when sharded it is a Replicate DTensor like
            # the router's scores, and DTensor + plain Tensor is rejected.
            masked_scores = self.router(flat, self.expert_bias_E, return_scores=lbl)
            if lbl:
                masked_scores, scores = masked_scores
            masked_scores = _localize_gate_output(masked_scores)
            if lbl:
                # Every TP rank holds the same tokens here, so each computes the same loss term.
                lbl_stats = (
                    (masked_scores != 0).sum(dim=0),
                    # pyrefly: ignore [unbound-name]
                    _localize_gate_output(scores).sum(dim=0),
                )

        # Expert usage for the balancing hook (sign() makes the AC double count harmless).
        with torch.no_grad():
            _local(self.tokens_per_expert_E).add_(
                (masked_scores != 0).to(torch.float32).sum(dim=0)
            )

        output = self.experts(flat, masked_scores)
        if isinstance(output, DTensor):
            output = output.to_local()
        if lbl:
            output = attach_global_batch_lbl(
                output,
                # pyrefly: ignore [unbound-name]
                *lbl_stats,
                self,
                top_k=self.router.top_k,
            )
        output = output.reshape(batch, sequence, dim)
        if spec is not None:
            output = DTensor.from_local(
                output, spec[0], tuple(Partial() for _ in spec[1]), run_check=False
            )
        return output


# NKI packs Qwen3 gate/up/down expert weights under different parameter names.
# Preserve the original per-weight initialization when converting GroupedExperts.
def _default_init(tensor) -> None:
    nn.init.trunc_normal_(tensor, mean=0.0, std=0.02)


def _gate_up_init(
    w1_init: Callable | None,
    w3_init: Callable | None,
) -> Callable:
    def init(tensor) -> None:
        (w1_init or _default_init)(tensor[:, :, 0, :])
        (w3_init or _default_init)(tensor[:, :, 1, :])

    return init


def _down_init(w2_init: Callable | None) -> Callable:
    def init(tensor) -> None:
        (w2_init or _default_init)(tensor)

    return init


def to_nki_moe_config(moe: MoE.Config) -> NKIMoE.Config:
    """Convert a standard Qwen3 MoE config into the Neuron NKI implementation."""
    if moe.shared_experts is not None:
        raise NotImplementedError("Qwen3 NKI MoE does not support shared experts")
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
            "Qwen3 NKI MoE does not support group-limited routing"
        )

    param_init = moe.experts.param_init or {}
    nki_moe = NKIMoE.Config(
        num_experts=moe.num_experts,
        load_balance_coeff=moe.load_balance_coeff,
        load_balancing=moe.load_balancing,
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
            param_init={
                "gate_up_proj_weight": _gate_up_init(
                    param_init.get("w1_EFD"),
                    param_init.get("w3_EFD"),
                ),
                "down_proj_weight": _down_init(param_init.get("w2_EDF")),
            },
        ),
    )
    return nki_moe
