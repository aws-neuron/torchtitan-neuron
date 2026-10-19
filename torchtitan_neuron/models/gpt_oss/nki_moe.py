# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

"""NKI MoE integration for GPT-OSS on Neuron.

GPT-OSS-specific pieces:
- SwiGLU with clamp (gate/up clamped at ±swiglu_limit; default 7.0)
- Fused (gate+up) weight layout: mlp1_weight (E, H, 2, I), mlp2_weight (E, I, H)
- +1 baked into up_bias (GPT-OSS init convention)
- ActFnType.Swish

This file intentionally keeps GPT-OSS NKI helpers local instead of refactoring
Qwen3 NKI or deprecated override paths.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal

import nki.language as nl

import spmd_types as spmd
import torch
import torch.distributed._functional_collectives
import torch.nn.functional as F
from nkilib.core.moe.moe_cte.moe_cte import (
    moe_cte,
    MoECTEImplementation,
    MoECTESpec,
    ShardOnIConfig,
)

from nkilib.core.subkernels.find_nonzero_indices import find_nonzero_indices
from nkilib.core.subkernels.indexed_flatten import indexed_flatten
from nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode
from nkilib.experimental.moe.bwd.blockwise_mm_backward import blockwise_mm_bwd
from nkilib.experimental.moe.bwd.moe_bwd_parameters import (
    AffinityOption,
    ClampLimits,
    ShardOption,
    SkipMode,
)
from torch import nn
from torch.distributed.tensor import DTensor, Partial
from torch_neuronx import nki_op

from torch_neuronx.nki_hop import wrap_nki
from torch_neuronx.utils import get_logical_neuron_cores

from torchtitan.models.common import Linear
from torchtitan.models.common.decoder_sharding import (
    dense_activation_placement,
    dense_param_placement,
    dense_sequence_parallel_placement,
)
from torchtitan.models.common.moe import TokenChoiceTopKRouter
from torchtitan.models.gpt_oss.moe import GptOssGroupedExperts, GptOssMoE
from torchtitan.protocols.module import Module
from torchtitan.protocols.sharding import LocalMapConfig, ShardingConfig


# ---------------------------------------------------------------------------
# GPT-OSS-specific kernel handles & constants
# ---------------------------------------------------------------------------
LNC = (int(get_logical_neuron_cores()),)

wrapped_moe_fwd = wrap_nki(moe_cte)
wrapped_moe_bwd = wrap_nki(blockwise_mm_bwd)
wrapped_find_nonzero = wrap_nki(find_nonzero_indices)
wrapped_indexed_flatten = wrap_nki(indexed_flatten)

NAMESPACE = "gptoss_nki_moe"

# The installed nkilib MoE kernel requires the hidden (H) and intermediate (I)
# dims to be multiples of this. GPT-OSS dims (2880) are not, so the local
# tensors are zero-padded up to the nearest multiple just for the kernel call.
KERNEL_DIM_MULTIPLE = 512


def _round_up(x: int, multiple: int) -> int:
    return -(-x // multiple) * multiple


def _compute_n_max(tokens: int, top_k: int, block_size: int, num_experts: int) -> int:
    return (tokens * top_k + block_size - 1) // block_size + (num_experts - 1)


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


def _local(t):
    """Plain local view of a (Replicate) DTensor buffer/param for the local MoE region."""
    return t.to_local() if isinstance(t, DTensor) else t


class _SumGradOverGroup(torch.autograd.Function):
    """Identity forward; backward all-reduces (sum) the gradient over ``group``."""

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        if ctx.group is not None:
            grad = torch.distributed._functional_collectives.all_reduce(
                grad, "sum", ctx.group
            )
        return grad, None


def _local_gate(gate: nn.Module, x: torch.Tensor, tp_group) -> torch.Tensor:
    """Router gate on local tensors (weight/bias are Replicate on TP).

    Runs outside DTensor so the weight gradient and the input gradient see one
    explicit TP reduction of the incoming scores gradient (each TP rank's experts
    shard returns a per-rank partial). ``tp_group=None`` skips the reduction.
    """
    w = gate.weight.to_local() if isinstance(gate.weight, DTensor) else gate.weight
    b = gate.bias
    if b is not None and isinstance(b, DTensor):
        b = b.to_local()
    # Reduce on the INPUT edge only (what mainline's NoParallel(Partial) on a
    # DTensor plan effectively does): the router-path gradient into x is summed
    # over TP, while the gate weight/bias gradient is taken from the local
    # scores gradient.
    return F.linear(_SumGradOverGroup.apply(x, tp_group), w, b)


class MaskedScoresTopKRouter(Module):
    """GPT-OSS NKI router that emits a dense ``(tokens, experts)`` score mask."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        gate: Linear.Config
        top_k: int
        score_func: Literal["softmax", "sigmoid"] = "softmax"
        route_norm: bool = True
        route_scale: float = 1.0
        _debug_force_load_balance: bool = False

    def __init__(self, config: Config):
        super().__init__()
        self.gate = config.gate.build()
        self.top_k = config.top_k
        self.num_experts = config.gate.out_features
        self.score_func = config.score_func
        self.route_norm = config.route_norm
        self.route_scale = config.route_scale
        self._debug_force_load_balance = config._debug_force_load_balance

    def forward(
        self,
        x: torch.Tensor,
        expert_bias: torch.Tensor | None = None,
        tp_group=None,
    ) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, dtype=torch.float32):
            scores = _local_gate(self.gate, x, tp_group)
        if self.score_func == "softmax":
            scores = F.softmax(scores, dim=-1)
        elif self.score_func == "sigmoid":
            scores = torch.sigmoid(scores)
        else:
            raise ValueError(f"Unsupported score function: {self.score_func}")

        if self._debug_force_load_balance:
            return _debug_force_load_balance_masked_scores(
                scores,
                self.top_k,
                self.num_experts,
                self.route_norm,
                self.route_scale,
            )

        scores_for_choice = scores if expert_bias is None else scores + expert_bias
        top_indices = torch.topk(
            scores_for_choice, k=self.top_k, dim=-1, sorted=False
        ).indices
        mask = torch.zeros_like(scores_for_choice, dtype=torch.bool)
        mask.scatter_(-1, top_indices, True)
        masked_scores = scores * mask
        if self.route_norm:
            masked_scores = masked_scores / (
                masked_scores.sum(dim=-1, keepdim=True) + 1e-20
            )
        return masked_scores * self.route_scale


# Backward-kernel blocking hyperparameters.
# ---------------------------------------------------------------------------
# nki_op-based forward — compile-compatible
# ---------------------------------------------------------------------------
@nki_op(f"{NAMESPACE}::nki_moe_fwd")
def nki_moe_fwd(
    hidden_states: torch.Tensor,  # (T, H)
    masked_scores: torch.Tensor,  # (T, E) float32, top_k nonzero per row
    mlp1_weight: torch.Tensor,  # (E, H, 2, I)
    mlp1_bias: torch.Tensor,  # (E, 2, I)
    mlp2_weight: torch.Tensor,  # (E, I, H)
    mlp2_bias: torch.Tensor,  # (E, H)
    top_k: int,
    tp_degree: int = 1,
    block_size: int = 256,
    swiglu_limit: float = 7.0,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, torch.Tensor
]:
    T, H = hidden_states.shape
    E = masked_scores.shape[1]
    B = block_size
    N_max = _compute_n_max(T, top_k, B, E)

    # --- Index calculation (NKI kernels) ---
    indices, counts = wrapped_find_nonzero[LNC](masked_scores.float())
    blocks_per_expert = (counts + B - 1) // B

    # Tail slack: pad last expert so sum == N_max
    slack = N_max - blocks_per_expert.sum()
    bpe_padded = blocks_per_expert.clone()
    # Use explicit positive index (E-1) instead of [-1] to avoid
    # torch-mlir slice_scatter lowering bug with negative indices
    last_idx = E - 1
    bpe_padded[last_idx] = bpe_padded[last_idx] + slack

    row_offsets = torch.cumsum(bpe_padded, 0) - bpe_padded

    expert_ids = torch.arange(E, device=hidden_states.device, dtype=torch.int32)
    block_to_expert = torch.repeat_interleave(
        expert_ids, bpe_padded, output_size=N_max
    ).unsqueeze(1)

    # padding_val=-1: indexed_flatten's LNC2 merge uses signed nl.maximum; sentinel
    # must be < every real value (real token ids are in [0, T)).
    token_position_to_id = wrapped_indexed_flatten[LNC](
        indices,
        B,
        N_max * B,
        row_offsets,
        padding_val=-1,
    )

    # With skip_dma=SkipMode(skip_token=True), the MoE kernel uses oob.skip
    # on token-indirect DMAs — sentinel index -1 is OOB and the DMA is
    # suppressed. No padding needed.
    affinities_fwd = masked_scores.to(hidden_states.dtype).reshape(-1, 1)
    # clone: a custom-op return must not alias an input (masked_scores); reshape
    # returns a view, which the eager torch._library aliasing check rejects.
    affinities = masked_scores.reshape(-1, 1).clone()

    if tp_degree > 1:
        mlp2_bias = mlp2_bias / tp_degree

    # --- Forward kernel ---
    spec = MoECTESpec(
        implementation=MoECTEImplementation.shard_on_i,
        shard_on_I=ShardOnIConfig(
            checkpoint_activation=True,
            expert_affinity_multiply_on_I=True,
        ),
    )

    # With expert_affinity_multiply_on_I=True, the kernel returns
    # (output, gate_up_activations_T) — NO down_activations.
    output, gup_act_T = wrapped_moe_fwd[LNC](
        hidden_states,
        affinities_fwd,  # bf16 to fwd kernel; fp32 is saved for bwd via the autograd output tuple
        mlp1_weight,
        mlp2_weight,
        token_position_to_id,
        block_to_expert,
        B,
        spec,
        gate_and_up_proj_bias=mlp1_bias,
        down_proj_bias=mlp2_bias,
        activation_function=ActFnType.Swish,
        expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
        skip_dma=SkipMode(skip_token=True),
        is_tensor_update_accumulating=True,
        gate_clamp_upper_limit=swiglu_limit,
        # +1 is baked into the up-bias, which the kernel clamps BEFORE the
        # swiglu multiply. GPT-OSS clamps the raw up then adds +1, so shift
        # the up-clamp window by +1: clamp(up+1, -L+1, L+1) == clamp(up, -L, L) + 1.
        up_clamp_lower_limit=-swiglu_limit + 1.0,
        up_clamp_upper_limit=swiglu_limit + 1.0,
        accumulation_dtype=nl.float32,
    )

    return (affinities, gup_act_T, token_position_to_id, block_to_expert, T, E, output)


# ---------------------------------------------------------------------------
# nki_op-based backward — compile-compatible
# ---------------------------------------------------------------------------
@nki_op(f"{NAMESPACE}::nki_moe_bwd")
def nki_moe_bwd(
    hidden_states: torch.Tensor,
    affinities: torch.Tensor,
    mlp1_weight: torch.Tensor,
    mlp2_weight: torch.Tensor,
    gup_act_T: torch.Tensor,
    token_position_to_id: torch.Tensor,
    block_to_expert: torch.Tensor,
    grad_out: torch.Tensor,
    B: int,
    swiglu_limit: float = 7.0,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    clamp_limits = ClampLimits(
        non_linear_clamp_upper_limit=swiglu_limit,
        linear_clamp_lower_limit=-swiglu_limit + 1.0,
        linear_clamp_upper_limit=swiglu_limit + 1.0,
    )

    return wrapped_moe_bwd[LNC](
        hidden_states,
        affinities,
        mlp1_weight,
        mlp2_weight,
        gup_act_T,
        None,  # down_proj_act_checkpoint — not used with AFFINITY_ON_I
        token_position_to_id,
        block_to_expert,
        grad_out,
        B,
        activation_type=ActFnType.Swish,
        affinity_option=AffinityOption.AFFINITY_ON_I,
        shard_option=ShardOption.SHARD_ON_HIDDEN,
        clamp_limits=clamp_limits,
        bias=True,
        skip_dma=SkipMode(skip_token=True),
        accumulation_dtype=nl.float32,
    )


# ---------------------------------------------------------------------------
# Autograd wiring via register_autograd
# ---------------------------------------------------------------------------
def _nki_moe_fwd_setup_context(ctx, inputs, output):
    (
        hidden_states,
        masked_scores,
        mlp1_weight,
        mlp1_bias,
        mlp2_weight,
        mlp2_bias,
        top_k,
        tp_degree,
        block_size,
        swiglu_limit,
    ) = inputs
    (
        affinities,
        gup_act_T,
        token_position_to_id,
        block_to_expert,
        T,
        E,
        _output,
    ) = output
    ctx.save_for_backward(
        hidden_states,
        affinities,
        mlp1_weight,
        mlp2_weight,
        gup_act_T,
        token_position_to_id,
        block_to_expert,
        mlp2_bias,
    )
    ctx.T = T
    ctx.E = E
    ctx.block_size = block_size
    ctx.swiglu_limit = swiglu_limit
    ctx.scores_dtype = masked_scores.dtype
    ctx.tp_degree = tp_degree


def _nki_moe_fwd_backward(ctx, *grad_outputs):
    # grad_outputs has one grad per forward output (10 total); only the last (output) matters
    grad_out_full = grad_outputs[-1]
    (
        hidden_states,
        affinities,
        mlp1_weight,
        mlp2_weight,
        gup_act_T,
        token_position_to_id,
        block_to_expert,
        mlp2_bias,
    ) = ctx.saved_tensors
    T, E = ctx.T, ctx.E
    B = ctx.block_size

    grad_out = grad_out_full[:T]

    (
        hidden_grad,
        affinities_grad,
        mlp1_weight_grad,
        mlp2_weight_grad,
        mlp1_bias_grad,
        mlp2_bias_grad,
    ) = nki_moe_bwd(
        hidden_states,
        affinities,
        mlp1_weight,
        mlp2_weight,
        gup_act_T,
        token_position_to_id,
        block_to_expert,
        grad_out,
        B,
        ctx.swiglu_limit,
    )

    scores_grad = affinities_grad.reshape(-1, E)[:T]

    # Kernel's AFFINITY_ON_I affinity grad omits the down-bias term, but the
    # forward scales the full down output (incl. bias) by affinity. Add it back.
    eff_bias = mlp2_bias.to(torch.float32)
    if ctx.tp_degree > 1:
        eff_bias = eff_bias / ctx.tp_degree
    bias_corr = grad_out.to(torch.float32) @ eff_bias.t()  # (T, E)
    routed_mask = (affinities.reshape(-1, E)[:T] != 0).to(bias_corr.dtype)
    scores_grad = (scores_grad.to(torch.float32) + bias_corr * routed_mask).to(
        ctx.scores_dtype
    )

    return (
        hidden_grad[:T],  # hidden_states
        scores_grad,  # masked_scores
        mlp1_weight_grad,  # mlp1_weight
        mlp1_bias_grad,  # mlp1_bias
        mlp2_weight_grad,  # mlp2_weight
        mlp2_bias_grad,  # mlp2_bias
        None,  # top_k
        None,  # tp_degree
        None,  # block_size
        None,  # swiglu_limit
    )


nki_moe_fwd.register_autograd(
    _nki_moe_fwd_backward, setup_context=_nki_moe_fwd_setup_context
)


# ---------------------------------------------------------------------------
# Module classes — GPT-OSS subclasses of common base classes
# ---------------------------------------------------------------------------
class NKIGroupedExperts(Module):
    """GPT-OSS specific blockwise experts.

    Weight layout (fused gate+up):
        mlp1_weight: (E, H, 2, I)
        mlp1_bias:   (E, 2, I)
        mlp2_weight: (E, I, H)
        mlp2_bias:   (E, H)

    Quirks:
    - SwiGLU clamp ±7.0 (handled by the kernel via ``gate_clamp_*`` kwargs)
    - ``+1`` baked into the up-bias at init
    - 2880→3072 hidden-dim padding so the installed nkilib kernel accepts H
    - Calls the GPT-OSS-specific ``nki_moe_fwd`` op
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        hidden_dim: int
        num_experts: int
        top_k: int
        block_size: int = 256
        swiglu_limit: float = 7.0
        tp_degree: int = 1

    def __init__(self, config: Config):
        super().__init__()
        self.dim = config.dim
        self.hidden_dim = config.hidden_dim
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.block_size = config.block_size
        self.swiglu_limit = config.swiglu_limit
        self._tp_degree = config.tp_degree

        E = self.num_experts
        H = self.dim
        I = self.hidden_dim
        self.mlp1_weight = nn.Parameter(torch.empty(E, H, 2, I))
        self.mlp1_bias = nn.Parameter(torch.empty(E, 2, I))
        self.mlp2_weight = nn.Parameter(torch.empty(E, I, H))
        self.mlp2_bias = nn.Parameter(torch.empty(E, H))

    def forward(self, hidden_states, masked_scores):
        """Triggers FSDP all-gather. Runs the GPT-OSS NKI MoE forward op."""
        if isinstance(self.mlp1_weight, DTensor):
            mlp1_w = self.mlp1_weight.to_local()
            mlp1_b = self.mlp1_bias.to_local()
            mlp2_w = self.mlp2_weight.to_local()
            mlp2_b = self.mlp2_bias.to_local()
        else:
            mlp1_w = self.mlp1_weight
            mlp1_b = self.mlp1_bias
            mlp2_w = self.mlp2_weight
            mlp2_b = self.mlp2_bias

        tp_degree = self._tp_degree
        if isinstance(self.mlp1_weight, DTensor):
            mesh_dim_names = self.mlp1_weight.device_mesh.mesh_dim_names
            if "tp" in mesh_dim_names:
                tp_dim_idx = mesh_dim_names.index("tp")
                tp_degree = self.mlp1_weight.device_mesh.size(tp_dim_idx)

        # Zero-pad H and I up to a kernel-legal multiple of KERNEL_DIM_MULTIPLE.
        H_orig = hidden_states.shape[-1]
        I_orig = mlp1_w.shape[-1]  # I per TP rank
        tp = tp_degree if tp_degree > 0 else 1
        H_target = _round_up(H_orig, KERNEL_DIM_MULTIPLE)
        I_target = _round_up(I_orig * tp, KERNEL_DIM_MULTIPLE) // tp

        needs_pad = (H_orig != H_target) or (I_orig != I_target)
        if needs_pad:
            H_pad = H_target - H_orig
            I_pad = I_target - I_orig
            hidden_states = torch.nn.functional.pad(hidden_states, (0, H_pad))
            mlp1_w = torch.nn.functional.pad(mlp1_w, (0, I_pad, 0, 0, 0, H_pad))
            mlp1_b = torch.nn.functional.pad(mlp1_b, (0, I_pad))
            mlp2_w = torch.nn.functional.pad(mlp2_w, (0, H_pad, 0, I_pad))
            mlp2_b = torch.nn.functional.pad(mlp2_b, (0, H_pad))

        (*_saved, T, _E, output) = nki_moe_fwd(
            hidden_states,
            masked_scores,
            mlp1_w,
            mlp1_b,
            mlp2_w,
            mlp2_b,
            self.top_k,
            tp_degree,
            self.block_size,
            self.swiglu_limit,
        )
        return output[:T, :H_orig]


class NKIMoE(Module):
    """GPT-OSS NKI MoE using shared masked-score routing helpers."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_experts: int
        router: MaskedScoresTopKRouter.Config
        experts: NKIGroupedExperts.Config
        load_balance_coeff: float | None = 1e-3
        shared_experts: None = None

        def set_sharding_config(self, *, enable_sp: bool, enable_ep: bool) -> None:
            if enable_ep:
                raise NotImplementedError("GPT-OSS NKI MoE supports TP-only experts")

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
                    "mlp1_weight": dense_param_placement(tp=spmd.S(3)),
                    "mlp1_bias": dense_param_placement(tp=spmd.S(2)),
                    "mlp2_weight": dense_param_placement(tp=spmd.S(1)),
                    "mlp2_bias": dense_param_placement(tp=spmd.R),
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
        self.top_k = config.router.top_k
        self.load_balance_coeff = config.load_balance_coeff
        self.shared_experts = None
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Run router + experts on plain local tensors. Each sub-module's
        # ShardingConfig re-wraps its plain input (DTensor.from_local on in_src)
        # and that wrap's backward all-reduces its Partial input gradient, so the
        # gradient arriving here is already fully reduced: localize with a plain
        # to_local(). Declaring it Partial sums it a second time (4x ffn_norm grad).
        spec = None
        tp_group = None
        if isinstance(x, DTensor):
            assert x.device_mesh.ndim == 1
            assert x.device_mesh.mesh_dim_names == ("tp",)
            spec = (x.device_mesh, x.placements)
            tp_group = x.device_mesh.get_group("tp")
            x = x.to_local()

        batch, sequence, dim = x.shape
        flat = x.reshape(-1, dim)
        # router.forward already localizes the gate output (Partial grad label);
        # its return value is a plain local tensor here.
        masked_scores = self.router(flat, _local(self.expert_bias_E), tp_group)

        with torch.no_grad():
            _local(self.tokens_per_expert_E).add_(
                (masked_scores != 0).to(torch.float32).sum(dim=0)
            )

        output = self.experts(flat, masked_scores)
        if isinstance(output, DTensor):
            output = output.to_local()
        output = output.reshape(batch, sequence, dim)
        if spec is not None:
            output = DTensor.from_local(
                output, spec[0], tuple(Partial() for _ in spec[1]), run_check=False
            )
        return output

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        if buffer_device is None:
            buffer_device = self.tokens_per_expert_E.device
        with torch.device(buffer_device):
            self.tokens_per_expert_E = torch.zeros(
                self.num_experts, dtype=torch.float32
            )
            if self.load_balance_coeff is not None:
                self.expert_bias_E = torch.zeros(self.num_experts, dtype=torch.float32)


def _mlp1_bias_init(stock_init: Callable) -> Callable:
    """Stock GPT-OSS gate/up bias init, plus GPT-OSS's +1 on the up bias.

    The NKI kernel packs the gate/up bias as ``(E, 2, I)``; index 1 is up.
    """

    def init(tensor) -> None:
        stock_init(tensor)
        tensor[:, 1, :] += 1.0

    return init


def to_nki_moe_config(moe: GptOssMoE.Config) -> NKIMoE.Config:
    """Convert a stock GPT-OSS MoE config into the Neuron NKI implementation.

    Keeps the stock router gate and per-parameter initialization, so a model
    built from either config starts from the same distribution.
    """
    if not isinstance(moe, GptOssMoE.Config):
        raise TypeError(f"Expected GptOssMoE.Config, got {type(moe)!r}")
    if not isinstance(moe.router, TokenChoiceTopKRouter.Config):
        raise TypeError(
            f"Expected TokenChoiceTopKRouter.Config, got {type(moe.router)!r}"
        )
    if not isinstance(moe.experts, GptOssGroupedExperts.Config):
        raise TypeError(
            f"Expected GptOssGroupedExperts.Config, got {type(moe.experts)!r}"
        )
    stock_init = moe.experts.param_init
    return NKIMoE.Config(
        num_experts=moe.num_experts,
        load_balance_coeff=moe.load_balance_coeff,
        router=MaskedScoresTopKRouter.Config(
            gate=replace(moe.router.gate),
            top_k=moe.router.top_k,
            score_func=moe.router.score_func,
            route_norm=moe.router.route_norm,
        ),
        experts=NKIGroupedExperts.Config(
            dim=moe.experts.dim,
            hidden_dim=moe.experts.hidden_dim,
            num_experts=moe.num_experts,
            top_k=moe.router.top_k,
            block_size=256,
            swiglu_limit=moe.experts.swiglu_limit,
            param_init={
                "mlp1_weight": stock_init["mlp1_weight_EGD"],
                "mlp1_bias": _mlp1_bias_init(stock_init["mlp1_bias_EG"]),
                "mlp2_weight": stock_init["mlp2_weight_EDF"],
                "mlp2_bias": stock_init["mlp2_bias_ED"],
            },
        ),
    )
