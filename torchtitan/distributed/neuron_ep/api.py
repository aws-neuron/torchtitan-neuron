# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

"""Neuron NKI all-to-all-v primitives for the ``neuron_ep`` token dispatcher.

Wraps the ``permute_a2av`` (dispatch) / ``unpermute_a2av`` (combine) NKI kernels
from ``nkilib.experimental.collectives.a2av_train`` into autograd-aware ops, and
builds the ``nki.collectives.ReplicaGroup`` from an EP ``DeviceMesh``.

The packed kernels are mathematical transposes: the adjoint of
``permute_a2av`` is ``unpermute_a2av`` and vice-versa. We wrap ONLY the kernel
calls in an ``autograd.Function`` so the surrounding score-reduction stays in
the autograd-tracked graph and the kernel-boundary backward is just the sibling
kernel.

NKI / nkilib / torch_neuronx imports are deferred to call time so this module
imports on a CPU-only host (the ``NeuronEPTokenDispatcher`` class is importable
everywhere; only running dispatch/combine requires the Neuron toolchain).

Tensor shape suffixes used here:
    M  routed rows on the local rank (== N = T*top_k for neuron_ep)
    H  hidden dimension
    E  EP degree (number of expert-parallel ranks)
    e  local experts per rank (num_experts // E)
"""

from __future__ import annotations

import torch


def gmm_align() -> int:
    """Row alignment the Neuron grouped-mm requires for its token dim and offsets.

    Must match the value ``validate_offs`` checks in
    ``torch_neuronx.python_ops.grouped_mm`` (default 128, overridable via
    ``TORCH_NEURONX_ATEN_GMM_ALIGN``). Falls back to the env/default off-device so
    the dispatcher pads consistently even where torch_neuronx is unavailable.
    """
    try:
        from torch_neuronx.utils import get_gmm_align

        return get_gmm_align()
    except ImportError:
        import os

        return int(os.environ.get("TORCH_NEURONX_ATEN_GMM_ALIGN", "128"))


_LNC: int | None = None


def _resolve_lnc() -> int:
    """Return the cached LNC, resolving it once before graph capture."""
    global _LNC
    if _LNC is None:
        from torch_neuronx.utils import get_logical_neuron_cores

        _LNC = int(get_logical_neuron_cores())
    return _LNC


def build_ep_replica_group_ranks(ep_mesh) -> list[list[int]]:
    """Return the full SPMD replica-group rank lists for an EP ``DeviceMesh``.

    ``nki.collectives.ReplicaGroup`` is constructed from ``list[list[int]]`` --
    one rank list per replica group. Under SPMD every rank must see the SAME
    full set of groups, so we read them from the Neuron mesh registry (populated
    by the DeviceMesh init hook at ``torch_neuronx`` import) rather than from the
    local process group, which only knows the current rank's group.
    """
    from torch_neuronx.distributed import mesh_registry

    _resolve_lnc()
    pg_name = ep_mesh.get_group().group_name
    # get_all_replica_groups falls back to the current rank's group only when the
    # PG is not in the registry -- wrong for multi-node SPMD. Require the entry.
    if pg_name not in mesh_registry._MESH_REGISTRY:
        raise ValueError(
            f"EP process group '{pg_name}' is not in the Neuron mesh registry. "
            "neuron_ep needs the full SPMD replica groups. Ensure the EP mesh was "
            "created via DeviceMesh so the registry hook captured it, and that "
            "TORCH_NEURONX_SPMD_DISABLE is unset."
        )
    ep_ranks = mesh_registry.get_all_replica_groups(pg_name)
    _validate_packed_ep_ranks(ep_ranks)
    return ep_ranks


def _validate_packed_ep_ranks(ep_ranks: list[list[int]]) -> None:
    """Validate the Trn3 device-group topology required by packed A2AV."""
    if not ep_ranks:
        raise ValueError("packed A2AV requires at least one EP replica group.")
    for ranks in ep_ranks:
        ep_size = len(ranks)
        if ep_size < 8 or ep_size % 4 != 0:
            raise ValueError(
                "packed A2AV requires EP >= 8 in complete four-rank Trn3 "
                f"device groups. Got replica group of size {ep_size}."
            )
        expected = list(range(ranks[0], ranks[0] + ep_size))
        if ranks != expected or ranks[0] % 4 != 0:
            raise ValueError(
                "packed A2AV requires sequential, four-rank-device-aligned "
                f"replica groups. Got {ranks}."
            )


def _packed_lnc2_grid(logical_neuron_cores: int) -> tuple[int]:
    if logical_neuron_cores != 2:
        raise RuntimeError(
            "packed A2AV requires NEURON_LOGICAL_NC_CONFIG=2; "
            f"got {logical_neuron_cores} logical NeuronCore(s)."
        )
    return (logical_neuron_cores,)


def _packed_permute_a2av_kernel(
    hidden_states_NH,
    packed_source_indices_N,
    send_counts_1E,
    recv_capacity,
    ep_ranks,
):
    """Call packed dispatch, returning ``[C,H]`` data and A2AV metadata."""
    from nki.collectives import ReplicaGroup
    from nkilib.experimental.collectives.a2av_train import permute_a2av
    from torch_neuronx.nki_hop import wrap_nki

    grid = _packed_lnc2_grid(_resolve_lnc())
    return wrap_nki(permute_a2av)[grid](
        hidden_states=hidden_states_NH,
        send_counts=send_counts_1E,
        recv_capacity=int(recv_capacity),
        replica_group=ReplicaGroup(ep_ranks),
        packed_source_indices=packed_source_indices_N,
    )


def build_packed_source_indices(expert_ids_N, send_counts_1E):
    """Build stable peer segments using tile-local NKI prefix scans."""
    from nkilib.experimental.collectives.a2av_train.build_packed_routing import (
        build_packed_routing,
    )
    from torch_neuronx.nki_hop import wrap_nki

    grid = _packed_lnc2_grid(_resolve_lnc())
    return wrap_nki(build_packed_routing)[grid](
        expert_ids_N.to(torch.int32), send_counts_1E
    )


def _packed_unpermute_a2av_kernel(
    output_CH,
    packed_source_indices_N,
    send_counts_1E,
    recv_counts_1E,
    ep_ranks,
):
    """Call packed combine, returning rows in original ``[N,H]`` order."""
    from nki.collectives import ReplicaGroup
    from nkilib.experimental.collectives.a2av_train import unpermute_a2av
    from torch_neuronx.nki_hop import wrap_nki

    grid = _packed_lnc2_grid(_resolve_lnc())
    return wrap_nki(unpermute_a2av)[grid](
        output=output_CH,
        send_counts=send_counts_1E,
        recv_counts=recv_counts_1E,
        replica_group=ReplicaGroup(ep_ranks),
        packed_source_indices=packed_source_indices_N,
    )


class _PackedPermuteA2AV(torch.autograd.Function):
    """Capacity-bounded packed dispatch with packed combine as its adjoint."""

    @staticmethod
    def forward(
        ctx,
        hidden_states_NH,
        packed_source_indices_N,
        send_counts_1E,
        recv_capacity,
        ep_ranks,
    ):
        recv_data_CH, metadata_4E = _packed_permute_a2av_kernel(
            hidden_states_NH,
            packed_source_indices_N,
            send_counts_1E,
            recv_capacity,
            ep_ranks,
        )
        h = hidden_states_NH.shape[1]
        # Metadata is uint32 elements. PyTorch does not implement floor division
        # for uint32, and narrowing first can wrap above INT32_MAX.
        recv_counts_1E = (metadata_4E[2:3, :].to(torch.int64) // h).to(torch.int32)
        ctx.save_for_backward(packed_source_indices_N, send_counts_1E, recv_counts_1E)
        ctx.ep_ranks = ep_ranks
        return recv_data_CH, recv_counts_1E

    @staticmethod
    def backward(ctx, grad_recv_data_CH, _grad_recv_counts):
        packed_source_indices_N, send_counts_1E, recv_counts_1E = ctx.saved_tensors
        grad_hidden_NH = _packed_unpermute_a2av_kernel(
            grad_recv_data_CH.contiguous(),
            packed_source_indices_N,
            send_counts_1E,
            recv_counts_1E,
            ctx.ep_ranks,
        )
        return grad_hidden_NH, None, None, None, None


class _PackedUnpermuteA2AV(torch.autograd.Function):
    """Packed combine with capacity-bounded packed dispatch as its adjoint."""

    @staticmethod
    def forward(
        ctx,
        output_CH,
        packed_source_indices_N,
        send_counts_1E,
        recv_counts_1E,
        ep_ranks,
    ):
        ctx.save_for_backward(packed_source_indices_N, send_counts_1E)
        ctx.recv_capacity = output_CH.shape[0]
        ctx.ep_ranks = ep_ranks
        return _packed_unpermute_a2av_kernel(
            output_CH,
            packed_source_indices_N,
            send_counts_1E,
            recv_counts_1E,
            ep_ranks,
        )

    @staticmethod
    def backward(ctx, grad_combined_NH):
        packed_source_indices_N, send_counts_1E = ctx.saved_tensors
        grad_output_CH, _metadata = _packed_permute_a2av_kernel(
            grad_combined_NH.contiguous(),
            packed_source_indices_N,
            send_counts_1E,
            ctx.recv_capacity,
            ctx.ep_ranks,
        )
        return grad_output_CH, None, None, None, None


def permute_a2av_dispatch_compact(
    hidden_states_NH,
    packed_source_indices_N,
    send_counts_1E,
    recv_capacity,
    ep_ranks,
):
    """Packed dispatch: ``[N,H]`` -> ``[C,H]`` plus source receive counts."""
    return _PackedPermuteA2AV.apply(
        hidden_states_NH,
        packed_source_indices_N,
        send_counts_1E,
        recv_capacity,
        ep_ranks,
    )


def unpermute_a2av_combine_packed(
    output_CH,
    packed_source_indices_N,
    send_counts_1E,
    recv_counts_1E,
    ep_ranks,
):
    """Packed combine: capacity buffer ``[C,H]`` -> original rows ``[N,H]``."""
    return _PackedUnpermuteA2AV.apply(
        output_CH,
        packed_source_indices_N,
        send_counts_1E,
        recv_counts_1E,
        ep_ranks,
    )
