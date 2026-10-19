# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

"""NKI Flash Attention wrapper for GPT-OSS on Neuron.

Calls nkilib attention_cte (forward) and attention_bwd (backward) directly,
with native support for attention sinks, sliding window, causal masking, and GQA.

Uses nki_op + register_autograd for torch.compile compatibility.
"""

from dataclasses import dataclass

import spmd_types as spmd
import torch
from nkilib.core.attention.attention_bwd import attention_bwd
from nkilib.core.attention.attention_cte import attention_cte
from torch.distributed.tensor import DTensor
from torch_neuronx import nki_op

from torch_neuronx.nki_hop import wrap_nki
from torch_neuronx.utils import get_logical_neuron_cores
from torchtitan.models.common.decoder_sharding import (
    dense_activation_placement,
    dense_param_placement,
)
from torchtitan.protocols.module import Module
from torchtitan.protocols.sharding import LocalMapConfig, ShardingConfig

wrapped_flash_fwd = wrap_nki(attention_cte)
wrapped_flash_bwd = wrap_nki(attention_bwd)

LNC = (int(get_logical_neuron_cores()),)
NAMESPACE = "gptoss_nki_attn"


def _nki_attn_fwd_impl(
    q: torch.Tensor,  # (B, S, H, D)
    k: torch.Tensor,  # (B, S, H_kv, D)
    v: torch.Tensor,  # (B, S, H_kv, D)
    sinks: torch.Tensor,  # (H,)
    scale: float,
    sw: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, S, H, D = q.shape
    H_kv = k.shape[2]

    q_nki = q.transpose(1, 2).contiguous().reshape(B * H, S, D)
    k_nki = k.transpose(1, 2).contiguous().reshape(B * H_kv, S, D)
    v_nki = v.transpose(1, 2).contiguous().reshape(B * H_kv, S, D)

    # When using sliding window, NKI requires scale=1.0; pre-scale Q instead.
    if sw is not None:
        q_nki = q_nki * scale
        kernel_scale = 1.0
    else:
        kernel_scale = scale

    sink_nki = sinks.unsqueeze(0).expand(B, -1).reshape(B * H, 1)

    output, neg_max, sum_recip = wrapped_flash_fwd[LNC](
        q_nki,
        k_nki,
        v_nki,
        scale=kernel_scale,
        causal_mask=True,
        sink=sink_nki,
        sliding_window=sw,
        tp_q=True,
        tp_k=True,
        cache_softmax=True,
    )

    lse = -(neg_max + torch.log(sum_recip))
    lse = lse.to(torch.float32).reshape(B, H, *lse.shape[-2:])
    # Output back to (B, S, H, D)
    output = output.reshape(B, H, S, D).transpose(1, 2).contiguous()

    return output, lse


# ---------------------------------------------------------------------------
# nki_op-based forward
# ---------------------------------------------------------------------------
@nki_op(f"{NAMESPACE}::nki_attn_fwd_full")
def nki_attn_fwd_full(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sinks: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _nki_attn_fwd_impl(q, k, v, sinks, scale, None)


@nki_op(f"{NAMESPACE}::nki_attn_fwd_window_128")
def nki_attn_fwd_window_128(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sinks: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _nki_attn_fwd_impl(q, k, v, sinks, scale, 128)


def _nki_attn_bwd_impl(
    q: torch.Tensor,  # (B, S, H, D)
    k: torch.Tensor,  # (B, S, H_kv, D)
    v: torch.Tensor,  # (B, S, H_kv, D)
    out: torch.Tensor,  # (B, S, H, D)
    grad_out: torch.Tensor,  # (B, S, H, D)
    lse: torch.Tensor,
    sinks: torch.Tensor,
    scale: float,
    sw: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, S, H, D = q.shape

    # (B, S, H, D) → (B, H, D, S) for the kernel's expected bwd layout.
    q_t = q.permute(0, 2, 3, 1).contiguous()
    k_t = k.permute(0, 2, 3, 1).contiguous()
    v_t = v.permute(0, 2, 3, 1).contiguous()
    out_t = out.permute(0, 2, 3, 1).contiguous()
    grad_out_t = grad_out.permute(0, 2, 3, 1).contiguous()
    sink_bwd = sinks.unsqueeze(0).expand(B, -1)

    grad_q_t, grad_k_t, grad_v_t, grad_sinks = wrapped_flash_bwd[LNC](
        q_t,
        k_t,
        v_t,
        out_t,
        grad_out_t,
        lse,
        sinks_ref=sink_bwd,
        use_causal_mask=True,
        mixed_precision=True,
        softmax_scale=scale,
        sliding_window=sw,
    )

    # Permute grads back to (B, S, H, D) so they match the forward's input
    return (
        grad_q_t.permute(0, 3, 1, 2).contiguous(),
        grad_k_t.permute(0, 3, 1, 2).contiguous(),
        grad_v_t.permute(0, 3, 1, 2).contiguous(),
        grad_sinks.sum(dim=0),
    )


# ---------------------------------------------------------------------------
# nki_op-based backward
# ---------------------------------------------------------------------------
@nki_op(f"{NAMESPACE}::nki_attn_bwd")
def nki_attn_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    grad_out: torch.Tensor,
    lse: torch.Tensor,
    sinks: torch.Tensor,
    scale: float,
    sliding_window: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sw = sliding_window if sliding_window != 0 else None
    return _nki_attn_bwd_impl(q, k, v, out, grad_out, lse, sinks, scale, sw)


# ---------------------------------------------------------------------------
# Autograd wiring
# ---------------------------------------------------------------------------
def _nki_attn_fwd_setup_context(ctx, inputs, output):
    q, k, v, sinks, scale = inputs
    out, lse = output
    ctx.save_for_backward(q, k, v, sinks, out, lse)
    ctx.scale = scale


def _nki_attn_fwd_full_backward(ctx, *grad_outputs):
    # grad_outputs: one per forward output (2); output is at index 0
    grad_out = grad_outputs[0]
    q, k, v, sinks, out, lse = ctx.saved_tensors

    grad_q, grad_k, grad_v, grad_sinks = nki_attn_bwd(
        q,
        k,
        v,
        out,
        grad_out,
        lse,
        sinks,
        ctx.scale,
        0,
    )

    return grad_q, grad_k, grad_v, grad_sinks, None


def _nki_attn_fwd_window_128_backward(ctx, *grad_outputs):
    # grad_outputs: one per forward output (2); output is at index 0
    grad_out = grad_outputs[0]
    q, k, v, sinks, out, lse = ctx.saved_tensors

    grad_q, grad_k, grad_v, grad_sinks = nki_attn_bwd(
        q,
        k,
        v,
        out,
        grad_out,
        lse,
        sinks,
        ctx.scale,
        128,
    )

    return grad_q, grad_k, grad_v, grad_sinks, None


nki_attn_fwd_full.register_autograd(
    _nki_attn_fwd_full_backward,
    setup_context=_nki_attn_fwd_setup_context,
)
nki_attn_fwd_window_128.register_autograd(
    _nki_attn_fwd_window_128_backward,
    setup_context=_nki_attn_fwd_setup_context,
)


# ---------------------------------------------------------------------------
# Module wrapper (unchanged interface)
# ---------------------------------------------------------------------------
class NKIFlashAttentionWrapper(Module):
    """Drop-in inner_attention replacement for GPT-OSS Attention.

    Accepts q/k/v in (B, S, H, D) layout (the natural output of view/cat from
    the linear projections). The transpose to head-major (B*H, S, D) layout
    expected by the NKI kernel happens inside the @nki_op so dynamo never
    sees non-contiguous strided tensors in the compiled HLO graph.

    Handles DTensor unwrap/wrap for TP compatibility.
    Sinks, sliding window, and causal masking are handled natively by the NKI kernel.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        softmax_scale: float
        sliding_window: int | None = None

        def set_sharding_config(self) -> None:
            """Set the TP-sharded q/k/v/sinks local region for the NKI kernel.

            Heads are sharded on TP (``Shard(2)`` of ``(B, S, H, D)``). Under CP,
            k/v are gathered to Replicate and their gradients return Partial.
            """
            q_placements = dense_activation_placement(tp=spmd.S(2))
            kv_src_placements = dense_activation_placement(tp=spmd.S(2))
            kv_dst_placements = dense_activation_placement(tp=spmd.S(2), cp=spmd.R)
            kv_grad_placements = dense_activation_placement(tp=spmd.S(2), cp=spmd.P)
            sinks_placements = dense_param_placement(tp=spmd.S(0))
            self.sharding_config = ShardingConfig(
                in_src_shardings={
                    "q": q_placements,
                    "k": kv_src_placements,
                    "v": kv_src_placements,
                    "sinks": sinks_placements,
                },
                in_dst_shardings={
                    "q": q_placements,
                    "k": kv_dst_placements,
                    "v": kv_dst_placements,
                    "sinks": sinks_placements,
                },
                out_src_shardings=q_placements,
                local_map=LocalMapConfig(
                    in_grad_placements=(
                        q_placements,
                        kv_grad_placements,
                        kv_grad_placements,
                        sinks_placements,
                    ),
                ),
            )

    uses_attention_sinks_natively = True

    def __init__(self, config: Config):
        super().__init__()
        if config.sliding_window not in (None, 128):
            raise ValueError(
                "GPT-OSS NKI attention only supports sliding_window=None or 128, "
                f"got {config.sliding_window!r}"
            )
        self.softmax_scale = config.softmax_scale
        # Use 0 to represent None (nki_op needs concrete int, not Optional)
        self.sliding_window_int = (
            config.sliding_window if config.sliding_window is not None else 0
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sinks: torch.Tensor,
    ) -> torch.Tensor:
        mesh = None
        placements = None
        if isinstance(q, DTensor):
            mesh = q.device_mesh
            placements = q.placements
            q = q.to_local()
            k = k.to_local()
            v = v.to_local()
        if isinstance(sinks, DTensor):
            sinks = sinks.to_local()

        if self.sliding_window_int == 128:
            (*_saved, out, _lse) = nki_attn_fwd_window_128(
                q,
                k,
                v,
                sinks,
                self.softmax_scale,
            )
        else:
            (*_saved, out, _lse) = nki_attn_fwd_full(
                q,
                k,
                v,
                sinks,
                self.softmax_scale,
            )

        if mesh is not None:
            out = DTensor.from_local(out, mesh, placements)
        return out
