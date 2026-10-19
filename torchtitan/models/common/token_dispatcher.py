# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Modifications Copyright (c) Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the BSD-3-Clause license.

from dataclasses import dataclass
from typing import cast, ClassVar

import spmd_types as spmd
import torch
from torch.distributed._functional_collectives import all_to_all_single
from torch.distributed.tensor import DeviceMesh

from torchtitan.config import Configurable
from torchtitan.distributed.minimal_async_ep import (
    combine_op as minimal_async_ep_combine_op,
    dispatch_op as minimal_async_ep_dispatch_op,
    init_buffer as minimal_async_ep_init_buffer,
    MinimalAsyncEPDispatchMetadata,
)
from torchtitan.distributed.neuron_ep import (
    build_ep_replica_group_ranks,
    build_packed_source_indices,
    gmm_align,
    permute_a2av_dispatch_compact,
    unpermute_a2av_combine_packed,
)
from torchtitan.distributed.spmd_types import current_spmd_mesh, maybe_set_sparse_mesh
from torchtitan.distributed.utils import get_spmd_backend
from torchtitan.ops.scatter_add import deterministic_scatter_add
from torchtitan.tools.utils import device_module, device_type


@dataclass(frozen=True, kw_only=True)
class LocalDispatchMetadata:
    """Metadata returned by LocalTokenDispatcher.dispatch() for use in combine()."""

    token_indices_experts_sorted_N: torch.Tensor  # noqa: N815
    topk_scores_experts_sorted_N: torch.Tensor  # noqa: N815


@dataclass(frozen=True, kw_only=True)
class AllToAllDispatchMetadata(LocalDispatchMetadata):
    """Metadata returned by AllToAllTokenDispatcher.dispatch() for use in combine()."""

    input_shape: tuple  # for _unpermute
    permuted_indices: torch.Tensor  # for _unpermute
    input_splits: list[int]
    output_splits: list[int]


@dataclass(frozen=True, kw_only=True)
class NeuronEPDispatchMetadata(LocalDispatchMetadata):
    """Metadata returned by NeuronEPTokenDispatcher.dispatch() for use in combine().

    ``token_indices_experts_sorted_N`` / ``topk_scores_experts_sorted_N`` are the
    pre-expanded (N = T*top_k) per-row token-id and gate-score maps, index-aligned
    with the kernel's ``routed_input_ND`` rows so combine can apply scores and
    scatter-add back to T tokens. They are TOKEN-MAJOR (``arange(N)//top_k`` and
    ``topk_scores_TK.reshape(-1)``): the dispatcher leaves the expert-sort to the
    kernel's packed-index permute, so combine relies only on 1:1 alignment with
    the kernel row order, not on the ordering being expert-sorted. (The field names
    keep the ``_experts_sorted_N`` suffix for historical/dataclass compatibility.)
    """

    send_counts_1E: torch.Tensor  # noqa: N815  # int32 [1, EP], rows sent per dest
    recv_counts_1E: torch.Tensor  # noqa: N815  # int32 [1, EP], rows received per source
    compact_capacity: int  # row count of the packed [C,H] expert buffer
    packed_source_indices_N: torch.Tensor  # noqa: N815


class LocalTokenDispatcher(Configurable):
    """Token dispatcher for EP=1. Handles local token reordering only.

    Not an nn.Module — dispatchers have no learnable parameters or buffers.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        num_experts: int
        top_k: int

    def __init__(self, config: Config):
        self.num_experts = config.num_experts
        self.top_k = config.top_k

    def wire_meshes(
        self,
        *,
        ep_mesh: DeviceMesh | None,
        tp_mesh: DeviceMesh | None,
    ) -> None:
        """No-op for the EP=1 dispatcher. Subclasses override."""
        del ep_mesh, tp_mesh

    def _local_reorder(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reorder tokens by expert assignment for local expert computation.

        Groups tokens by expert index via argsort. Routing scores are applied
        to the expert outputs in ``combine``, after expert computation.

        Args:
            x_TD: ``(T, D)`` input tokens
            topk_scores_TK: ``(T, K)`` routing scores
            topk_expert_ids_TK: ``(T, K)`` expert indices

        Returns:
            routed_input_ND: ``(N, D)`` where N = T*K. Tokens in expert-sorted
                order.
            token_indices_experts_sorted_N: ``(N,)`` token-to-original mapping
            topk_scores_experts_sorted_N: ``(N,)`` scores in expert-sorted order
        """
        # Reorder the token indices to match the order of the experts where N = T*K
        token_indices_experts_sorted_N = torch.argsort(
            topk_expert_ids_TK.view(-1), stable=True
        )
        topk_scores_experts_sorted_N = topk_scores_TK.view(-1)[
            token_indices_experts_sorted_N
        ]
        token_indices_experts_sorted_N = token_indices_experts_sorted_N // self.top_k
        routed_input_ND = x_TD[token_indices_experts_sorted_N]

        return (
            routed_input_ND,
            token_indices_experts_sorted_N,
            topk_scores_experts_sorted_N,
        )

    def dispatch(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, LocalDispatchMetadata]:
        """Reorder tokens by expert assignment for local expert computation.

        Args:
            x_TD: ``(T, D)`` all input tokens
            topk_scores_TK: ``(T, K)`` routing scores
            topk_expert_ids_TK: ``(T, K)`` expert indices per token
            num_local_tokens_per_expert_E: ``(E,)`` token counts per expert

        Returns:
            routed_input_RD: ``[R = sum(num_local_tokens_per_expert_E), input_dim(D)]``.
                Tokens sorted by expert index.
            num_local_tokens_per_expert_E: ``(E,)`` token counts per expert
            metadata: LocalDispatchMetadata for combine()
        """
        # R = N (no EP all-to-all)
        (
            routed_input_RD,
            token_indices_experts_sorted_N,
            topk_scores_experts_sorted_N,
        ) = self._local_reorder(x_TD, topk_scores_TK, topk_expert_ids_TK)

        metadata = LocalDispatchMetadata(
            token_indices_experts_sorted_N=token_indices_experts_sorted_N,
            topk_scores_experts_sorted_N=topk_scores_experts_sorted_N,
        )
        return routed_input_RD, num_local_tokens_per_expert_E, metadata

    def combine(
        self,
        routed_output_RD: torch.Tensor,
        metadata: LocalDispatchMetadata,
        x_TD: torch.Tensor,
        *,
        num_local_tokens_after_padding: int,
        local_seq_len_after_padding: int,
    ) -> torch.Tensor:
        """Score and scatter_add routed expert outputs.

        Args:
            routed_output_RD: ``(R, D)`` expert outputs
            metadata: LocalDispatchMetadata from dispatch()
            x_TD: ``(T, D)`` original input tokens
            num_local_tokens_after_padding: Unused for local dispatch; kept
                for a shared dispatcher combine signature.
            local_seq_len_after_padding: Unused for local dispatch; kept for
                a shared dispatcher combine signature.

        Returns:
            out_TD: ``(T, D)`` combined output.
        """
        del num_local_tokens_after_padding, local_seq_len_after_padding
        out_TD = torch.zeros_like(x_TD)

        routed_output_RD = (
            routed_output_RD.to(torch.float32)
            * metadata.topk_scores_experts_sorted_N.reshape(-1, 1)
        ).to(routed_output_RD.dtype)

        dim = x_TD.shape[-1]
        out_TD = deterministic_scatter_add(
            out_TD,
            metadata.token_indices_experts_sorted_N.reshape(-1, 1).expand(-1, dim),
            routed_output_RD,
        )
        return out_TD


class BaseEPTokenDispatcher(LocalTokenDispatcher):
    """Base class for EP token dispatchers.

    Owns EP mesh wiring and SP coordinate helpers shared by EP implementations.
    LocalTokenDispatcher intentionally does not know about SP: local dispatch is
    used when EP is off, and expert activations are replicated for expert TP.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(LocalTokenDispatcher.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        self.ep_mesh: DeviceMesh | None = None
        # Sequence-parallel split coordinates derived from tp_mesh.
        # ``sp_rank`` uses ``DeviceMesh._sym_get_coordinate`` so it is a
        # ``SymInt`` under CooR precompile, keeping the FX graph
        # rank-agnostic. Defaults are the TP=1 values.
        self.sp_size: int = 1
        self.sp_rank: int | torch.SymInt = 0

    def wire_meshes(
        self,
        *,
        ep_mesh: DeviceMesh | None,
        tp_mesh: DeviceMesh | None,
    ) -> None:
        """Install the EP mesh and SP coordinates used by dispatch / combine."""
        self.ep_mesh = ep_mesh
        if tp_mesh is not None:
            self.sp_size = tp_mesh.size()
            self.sp_rank = tp_mesh._sym_get_coordinate(0)

    def _sp_global_token_indices(
        self,
        local_indices: torch.Tensor,
        local_seq_len: int,
    ) -> torch.Tensor:
        if self.sp_size == 1:
            return local_indices

        local_pos = local_indices % local_seq_len
        batch_idx = local_indices // local_seq_len
        global_seq_len = local_seq_len * self.sp_size
        global_indices = batch_idx * global_seq_len + local_pos
        return torch.add(  # pyrefly: ignore [no-matching-overload]
            global_indices, self.sp_rank * local_seq_len
        )

    def dispatch(self, *args, **kwargs):
        raise NotImplementedError("BaseEPTokenDispatcher does not implement dispatch")

    def combine(self, *args, **kwargs):
        raise NotImplementedError("BaseEPTokenDispatcher does not implement combine")


class AllToAllTokenDispatcher(BaseEPTokenDispatcher):
    """Token dispatcher for EP>1. Handles token reorder + all-to-all dispatch/combine.

    Handles the full token routing lifecycle:
    dispatch (reorder + EP all-to-all) and combine (reverse).

    ``ep_mesh`` and the ``sp_size`` / ``sp_rank`` SP coordinates are wired
    by the owning ``GroupedExperts.parallelize`` override via
    ``wire_meshes``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseEPTokenDispatcher.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)

    def _token_count_exchange(
        self,
        num_local_tokens_per_expert_E: torch.Tensor,
        pg,
        ep_size: int,
    ) -> torch.Tensor:
        """Exchange per-rank expert token counts before the data all-to-all.

        This method is separate from ``dispatch`` so graph passes can annotate
        the count exchange independently from the true token-exchange
        scheduling markers.
        """
        assert self.ep_mesh is not None
        if (
            torch.compiler.is_compiling() or torch.compiler._is_non_strict_tracing()
        ) and get_spmd_backend() != "spmd_types":
            return all_to_all_single(
                num_local_tokens_per_expert_E.view(ep_size, -1),
                None,
                None,
                group=self.ep_mesh,
            )

        return spmd.all_to_all(
            num_local_tokens_per_expert_E.view(ep_size, -1),
            pg,
            src=spmd.V,
            dst=spmd.V,
        )

    def _sync_token_count_exchange(
        self,
        num_local_tokens_per_expert_E: torch.Tensor,
        num_global_tokens_per_local_expert_EP_e: torch.Tensor,
        ep_size: int,
    ) -> tuple[torch.Tensor, list[int], list[int]]:
        """Wait for token counts and materialize CPU split lists.

        Local input splits can copy to CPU non-blocking; remote output splits
        must be ready before launching the variable-size data all-to-all.
        """
        # Need to wait explicitly because it is used by a triton kernel later
        # which doesn't realize that AsyncCollectiveTensor needs unwrapping
        num_global_tokens_per_local_expert_EP_e = (
            torch.ops._c10d_functional.wait_tensor(
                num_global_tokens_per_local_expert_EP_e
            )
        )
        num_global_tokens_per_local_expert_E = (
            num_global_tokens_per_local_expert_EP_e.reshape(-1)
        )
        input_splits = (
            num_local_tokens_per_expert_E.view(ep_size, -1)
            .sum(dim=1)
            .to(torch.device("cpu"), non_blocking=True)
        )
        # NOTE: this would incur a device-to-host sync
        output_splits = (
            num_global_tokens_per_local_expert_E.view(ep_size, -1)
            .sum(dim=1)
            .to(torch.device("cpu"), non_blocking=False)
        )
        input_splits_list = input_splits.tolist()
        output_splits_list = output_splits.tolist()

        return (
            num_global_tokens_per_local_expert_E,
            input_splits_list,
            output_splits_list,
        )

    def _dispatch_token_exchange(
        self,
        routed_input_ND: torch.Tensor,
        pg,
        output_splits: list[int],
        input_splits: list[int],
    ) -> torch.Tensor:
        """Launch the dispatch all-to-all that moves routed tokens to experts."""
        assert self.ep_mesh is not None
        if (
            torch.compiler.is_compiling() or torch.compiler._is_non_strict_tracing()
        ) and get_spmd_backend() != "spmd_types":
            return all_to_all_single(
                routed_input_ND,
                output_splits,
                input_splits,
                self.ep_mesh,
            )

        return spmd.all_to_all(
            routed_input_ND,
            pg,
            src=spmd.V,
            dst=spmd.V,
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
        )

    def _combine_token_exchange(
        self,
        routed_output_RD: torch.Tensor,
        pg,
        input_splits: list[int],
        output_splits: list[int],
    ) -> torch.Tensor:
        """Launch the combine all-to-all that returns expert outputs to tokens."""
        assert self.ep_mesh is not None
        if (
            torch.compiler.is_compiling() or torch.compiler._is_non_strict_tracing()
        ) and get_spmd_backend() != "spmd_types":
            return all_to_all_single(
                routed_output_RD,
                input_splits,
                output_splits,
                self.ep_mesh,
            )

        return spmd.all_to_all(
            routed_output_RD,
            pg,
            src=spmd.V,
            dst=spmd.V,
            output_split_sizes=input_splits,
            input_split_sizes=output_splits,
        )

    # pyrefly: ignore [bad-override]
    def dispatch(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> tuple[
        torch.Tensor, torch.Tensor, AllToAllDispatchMetadata | LocalDispatchMetadata
    ]:
        """Reorder tokens, then all-to-all dispatch to expert-parallel ranks.

        When ep_mesh is None (EP=1), falls back to local dispatch — no
        all-to-all communication, just local token reordering with padding.

        With SP, x_TD/topk_scores_TK/topk_expert_ids_TK are already
        the local SP shard (from DTensor Shard to_local via LocalMapConfig).

        Args:
            x_TD: ``(T, D)`` local token shard
            topk_scores_TK: ``(T, K)`` routing scores
            topk_expert_ids_TK: ``(T, K)`` expert indices
            num_local_tokens_per_expert_E: ``(E,)`` token counts for this local
                token shard

        Returns:
            routed_input_RD: ``[R = sum(num_tokens_per_local_expert_e), input_dim(D)]``.
                Tokens in expert-major order for local experts.
            num_tokens_per_local_expert_e: ``(num_local_experts,)`` token counts
            metadata: dispatch metadata for combine()
        """
        # EP=1: fall back to local dispatch (no all-to-all needed)
        if self.ep_mesh is None:
            return LocalTokenDispatcher.dispatch(
                self,
                x_TD,
                topk_scores_TK,
                topk_expert_ids_TK,
                num_local_tokens_per_expert_E,
            )

        ep_size = self.ep_mesh.size()
        # _local_reorder returns (N, D) where N = T*K.
        # EP all-to-all below produces (R, D) where R != N.
        (
            routed_input_ND,
            token_indices_experts_sorted_N,
            topk_scores_experts_sorted_N,
        ) = self._local_reorder(x_TD, topk_scores_TK, topk_expert_ids_TK)

        if (
            get_spmd_backend() == "spmd_types" and spmd.is_type_checking()
        ):  # sparse mesh reinterpret
            for axis in ["dp", "cp", "tp"]:
                spmd.mutate_type(
                    num_local_tokens_per_expert_E, axis, src=spmd.P, dst=spmd.V
                )

        # generate the input splits and output splits for all-to-all
        with maybe_set_sparse_mesh():
            pg = (
                current_spmd_mesh().get_group(  # pyrefly: ignore [missing-attribute]
                    "ep"
                )
                if get_spmd_backend() == "spmd_types"
                else self.ep_mesh.get_group()
            )
            if get_spmd_backend() == "spmd_types" and spmd.is_type_checking():
                num_local_tokens_per_expert_E = spmd.reinterpret_mesh(
                    num_local_tokens_per_expert_E, spmd.current_mesh()
                )
                routed_input_ND = spmd.reinterpret_mesh(
                    routed_input_ND, spmd.current_mesh()
                )

            with torch.no_grad():
                num_global_tokens_per_local_expert_EP_e = self._token_count_exchange(
                    num_local_tokens_per_expert_E,
                    pg,
                    ep_size,
                )
                (
                    num_global_tokens_per_local_expert_E,
                    input_splits_list,
                    output_splits_list,
                ) = self._sync_token_count_exchange(
                    num_local_tokens_per_expert_E,
                    num_global_tokens_per_local_expert_EP_e,
                    ep_size,
                )

            routed_input_RD = self._dispatch_token_exchange(
                routed_input_ND,
                pg,
                output_splits_list,
                input_splits_list,
            )
            # Reorder from rank-major to expert-major via _permute.
            #
            # num_global_tokens_per_local_expert_E layout after all-to-all
            # (e = local experts, EP = EP ranks):
            #   (e0,r0), (e1,r0), ..., (e0,r1), (e1,r1), ...  (rank-major)
            # _permute reshuffles to:
            #   (e0,r0), (e0,r1), ..., (e1,r0), (e1,r1), ...  (expert-major)
            # TODO: Consider using num_global_tokens_per_local_expert_e as the
            # expert_bias_e update buffer, then all-gather on EP ranks. This
            # is blocked by clarification on HybridEP token dropping.
            (
                input_shape,
                routed_input_RD,
                permuted_indices,
                num_global_tokens_per_local_expert_e,
            ) = self._permute(
                routed_input_RD,
                num_global_tokens_per_local_expert_E,
            )

        metadata = AllToAllDispatchMetadata(
            token_indices_experts_sorted_N=token_indices_experts_sorted_N,
            topk_scores_experts_sorted_N=topk_scores_experts_sorted_N,
            input_shape=input_shape,
            permuted_indices=permuted_indices,
            input_splits=input_splits_list,
            output_splits=output_splits_list,
        )
        return routed_input_RD, num_global_tokens_per_local_expert_e, metadata

    def _permute(
        self,
        routed_input_RD,
        num_global_tokens_per_local_expert_E,
    ):
        """Reorder tokens from rank-major to expert-major layout.

        Input layout:  (e0,r0), (e1,r0), ..., (e0,r1), (e1,r1), ...  (rank-major)
        Output layout: (e0,r0), (e0,r1), ..., (e1,r0), (e1,r1), ...  (expert-major)

        Collapses token count matrix ``t_mat`` from ``(EP, e)`` to
        ``num_global_tokens_per_local_expert_e`` ``(e,)`` by summing across ranks.
        """
        # pyrefly: ignore [missing-attribute]
        ep_size = self.ep_mesh.size()
        e = num_global_tokens_per_local_expert_E.shape[0] // ep_size
        device = num_global_tokens_per_local_expert_E.device
        total = routed_input_RD.shape[0]

        # (EP, e) matrix of token counts per (rank, local_expert)
        t_mat = num_global_tokens_per_local_expert_E.view(ep_size, e)

        # Where each (r, e) segment starts in the input (rank-major order)
        input_starts = (
            num_global_tokens_per_local_expert_E.cumsum(0)
            - num_global_tokens_per_local_expert_E
        ).view(ep_size, e)

        # Transpose to expert-major (e, EP) and flatten
        segment_lens = t_mat.t().reshape(-1)
        input_starts = input_starts.t().reshape(-1)

        # For each output position, find its input position:
        #   output[p] = input[input_starts[seg] + (p - output_starts[seg])]
        seg_ids = torch.arange(segment_lens.shape[0], device=device).repeat_interleave(
            segment_lens, output_size=total
        )
        output_starts = segment_lens.cumsum(0) - segment_lens
        # seg_ids.shape[0] == segment_lens.sum(), but reuses the unbacked symint
        # already created by repeat_interleave above.
        permuted_indices = (
            input_starts[seg_ids]
            + torch.arange(seg_ids.shape[0], device=device)
            - output_starts[seg_ids]
        )

        num_global_tokens_per_local_expert_e = t_mat.sum(0)
        return (
            routed_input_RD.shape,
            routed_input_RD[permuted_indices, :],
            permuted_indices,
            num_global_tokens_per_local_expert_e,
        )

    def _unpermute(self, routed_output_RD, input_shape, permuted_indices):
        """Reverse expert-major reordering."""
        out_unpermuted_RD = routed_output_RD.new_empty(input_shape)
        out_unpermuted_RD[permuted_indices, :] = routed_output_RD
        return out_unpermuted_RD

    # pyrefly: ignore [bad-override]
    def combine(
        self,
        routed_output_RD: torch.Tensor,
        metadata: AllToAllDispatchMetadata,
        x_TD: torch.Tensor,
        *,
        num_local_tokens_after_padding: int,
        local_seq_len_after_padding: int,
    ) -> torch.Tensor:
        """Reverse the dispatch: unpermute + all-to-all + score + scatter_add.

        When sp_size > 1, dispatch uses local token indices.
        Combine offsets them to global positions so scatter_add
        into full x_TD is correct.

        Args:
            routed_output_RD: ``(R, D)`` expert outputs in expert-major order
            metadata: AllToAllDispatchMetadata from dispatch()
            x_TD: ``(T, D)`` original input tokens
            num_local_tokens_after_padding: Local token count to use for the
                combined SP view after logical padding. MoE padding passes this
                count without materializing pad rows.
            local_seq_len_after_padding: Per-batch local sequence length after
                logical padding, used to map local token indices to global SP
                positions.

        Returns:
            out_TD: Combined output. With SP, shape is
                ``(num_local_tokens_after_padding * sp_size, D)``.
        """
        # EP=1: fall back to local combine (no all-to-all needed)
        if self.ep_mesh is None:
            return LocalTokenDispatcher.combine(
                self,
                routed_output_RD,
                metadata,
                x_TD,
                num_local_tokens_after_padding=num_local_tokens_after_padding,
                local_seq_len_after_padding=local_seq_len_after_padding,
            )

        with maybe_set_sparse_mesh():
            pg = (
                current_spmd_mesh().get_group(  # pyrefly: ignore [missing-attribute]
                    "ep"
                )
                if get_spmd_backend() == "spmd_types"
                else self.ep_mesh.get_group()
            )
            # Reverse expert-major reordering
            routed_output_RD = self._unpermute(
                routed_output_RD, metadata.input_shape, metadata.permuted_indices
            )
            # All-to-all combine: returns AsyncCollectiveTensor — the a2a runs
            # on the NCCL stream and won't block until the tensor is accessed.
            routed_output_RD = self._combine_token_exchange(
                routed_output_RD,
                pg,
                metadata.input_splits,
                metadata.output_splits,
            )

        if get_spmd_backend() == "spmd_types":
            if spmd.is_type_checking():  # dense mesh reinterpret
                routed_output_RD = spmd.reinterpret_mesh(
                    routed_output_RD, spmd.current_mesh()
                )

        # With SP, create a full-size buffer for scatter_add so routed results
        # from all SP ranks can be placed at global positions.
        out_TD = torch.zeros(
            num_local_tokens_after_padding * self.sp_size,
            x_TD.shape[-1],
            device=x_TD.device,
            dtype=x_TD.dtype,
        )

        routed_output_RD = (
            routed_output_RD.to(torch.float32)
            * metadata.topk_scores_experts_sorted_N.reshape(-1, 1)
        ).to(routed_output_RD.dtype)

        token_indices_experts_sorted_N = self._sp_global_token_indices(
            metadata.token_indices_experts_sorted_N,
            local_seq_len_after_padding,
        )

        assert isinstance(token_indices_experts_sorted_N, torch.Tensor)
        out_TD = deterministic_scatter_add(
            out_TD,
            token_indices_experts_sorted_N.reshape(-1, 1).expand(-1, out_TD.shape[-1]),
            routed_output_RD,
        )
        return out_TD


class TorchAOTokenDispatcher(AllToAllTokenDispatcher):
    """Token dispatcher with token group padding for quantized grouped GEMMs.

    Uses torchao's ``permute_and_pad`` instead of the standard ``_permute`` to
    reorder tokens into expert-major order and pad each expert's token group to
    a multiple of ``pad_multiple``. This alignment is required by FP8/MXFP8
    quantized grouped GEMM kernels (e.g. 16 for FP8, 32 for MXFP8).

    Requires EP to be enabled (ep_mesh must be set). Raises ValueError
    if ep_mesh is None, since quantized grouped GEMMs need padded token
    groups which are only produced by the EP permute_and_pad path.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(AllToAllTokenDispatcher.Config):
        pad_multiple: int

    def __init__(self, config: Config):
        super().__init__(config)
        self.pad_multiple = config.pad_multiple

    def dispatch(
        self,
        x_TD,
        topk_scores_TK,
        topk_expert_ids_TK,
        num_local_tokens_per_expert_E,
    ):
        if self.ep_mesh is None:
            raise ValueError(
                "TorchAOTokenDispatcher requires expert parallelism (ep_mesh must be set). "
                "Quantized grouped GEMMs need padded token groups, which requires EP>1. "
            )
        return super().dispatch(
            x_TD, topk_scores_TK, topk_expert_ids_TK, num_local_tokens_per_expert_E
        )

    def _permute(
        self,
        routed_input_RD,
        num_global_tokens_per_local_expert_E,
    ):
        # FP8/MXFP8 require groups to be permuted to expert major order AND
        # padded to nearest multiple of 16.
        # It also does padding to make sure the number of tokens each expert
        # gets locally is a multiple of `self.pad_multiple`.
        # Note that this will create side effects when wrapping the for-loop
        # implementation of GroupedExperts, as it does not need padding.
        from torchao.prototype.moe_training.ep.permute import permute_and_pad

        # pyrefly: ignore [missing-attribute]
        ep_size = self.ep_mesh.size()
        e = num_global_tokens_per_local_expert_E.shape[0] // ep_size

        (
            input_shape,
            routed_input_RD,
            permuted_indices,
            num_global_tokens_per_local_expert_padded_e,
            _group_offsets,
        ) = permute_and_pad(
            routed_input_RD,
            num_global_tokens_per_local_expert_E,
            ep_size,
            e,
            self.pad_multiple,
        )
        return (
            input_shape,
            routed_input_RD,
            permuted_indices,
            num_global_tokens_per_local_expert_padded_e,
        )

    def _unpermute(self, routed_output_RD, input_shape, permuted_indices):
        # Strip the padding sentinel row added by permute_and_pad
        out_unpermuted_RD = routed_output_RD.new_empty(input_shape)
        out_unpermuted_RD[permuted_indices, :] = routed_output_RD
        return out_unpermuted_RD[:-1]


class NeuronEPTokenDispatcher(BaseEPTokenDispatcher):
    """EP token dispatcher backed by Neuron NKI all-to-all-v kernels.

    Design:

    1. Packed capacity. The supported EP path requires
       ``num_local_experts == 1`` and sends real per-destination counts into a
       packed ``[C,H]`` receive buffer. The aligned capacity
       ``C = FSDP * local_batch * global_sequence_length`` is the unique-token
       count represented by the EP group. With multiple local experts, capacity
       scales by ``min(top_k, num_local_experts)`` once that path is enabled.
       The kernel output is ``[C,H]``, so its element count is ``C * H``. Its
       NKI kernel overlaps the count exchange with local payload packing.

    2. Token-major kernel input. Feeds the NKI ``permute_a2av`` kernel a TOKEN-MAJOR
       ``routed_input_ND`` (row ``r = t*K + k`` = token ``t`` replicated for its k-th
       expert choice) and lets the kernel's packed vector do the routing sort
       in a single pass -- no glue-side argsort + gather.

    The published kernels currently require Trn3, LNC=2, EP >= 8, and complete
    four-rank device groups. Multi-local-expert packed regrouping is a separate
    milestone because source-major receive counts do not identify local experts.

    Scope: EP with ``num_experts % ep_size == 0``. TP/SP (``sp_size > 1``) is supported
    -- routing is unaffected by the sequence shard (the packed vector keys off expert ids
    only, and the a2a-v round-trip preserves per-row identity), so SP only offsets the
    final scatter in ``combine`` to global token positions. CP is not supported.

    Tensor shape suffixes (module-local): ``N`` routed rows (T*top_k), ``D``/``H`` hidden,
    ``E`` EP degree, ``e`` local experts per rank, and ``C`` packed receive capacity.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseEPTokenDispatcher.Config):
        # Optional per-source capacity override. The packed buffer has
        # EP*min(top_k, num_local_experts)*capacity_per_expert rows when set.
        capacity_per_expert: int | None = None

    def __init__(self, config: Config):
        super().__init__(config)
        self._cap_override = config.capacity_per_expert
        # Full SPMD replica-group rank lists for nki.collectives.ReplicaGroup,
        # set in wire_meshes once the EP mesh is known.
        self._ep_ranks: list[list[int]] | None = None

    def wire_meshes(self, *, ep_mesh, tp_mesh) -> None:
        super().wire_meshes(ep_mesh=ep_mesh, tp_mesh=tp_mesh)
        if ep_mesh is not None:
            if self.num_experts % ep_mesh.size() != 0:
                raise ValueError(
                    f"neuron_ep requires num_experts ({self.num_experts}) divisible by "
                    f"ep_size ({ep_mesh.size()})."
                )
            self._ep_ranks = build_ep_replica_group_ranks(ep_mesh)

    def _packed_capacity(
        self,
        local_token_count: int,
        ep_size: int,
        num_local_experts: int,
    ) -> int:
        """Return the grouped-mm-aligned dropless receive capacity.

        ``local_token_count * ep_size`` equals
        ``FSDP * local_batch * global_sequence_length`` for the EP/TP mesh.
        A token can target at most ``min(top_k, num_local_experts)`` experts on
        one destination rank because its top-k expert IDs are distinct.
        """
        alignment = gmm_align()
        expert_multiplier = min(self.top_k, num_local_experts)
        model_capacity = local_token_count * ep_size * expert_multiplier
        capacity = model_capacity
        if self._cap_override is not None:
            capacity = ep_size * expert_multiplier * self._cap_override
            if capacity < model_capacity:
                raise ValueError(
                    "compact neuron_ep requires "
                    "EP*min(top_k,num_local_experts)*capacity_per_expert "
                    ">= model capacity "
                    f"({ep_size}*{expert_multiplier}*{self._cap_override}="
                    f"{capacity} < {model_capacity})."
                )
        return ((capacity + alignment - 1) // alignment) * alignment

    def _build_packed_routing(
        self,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ):
        """Build stable destination-major routing for one local expert per rank.

        Returns ``(counts[1, EP], source_indices[N])``. Peer segments preserve
        token-major arrival order, with every routed row appearing exactly once.
        """
        # The router already counts local tokens for each expert. With one expert
        # per rank and distinct top-k choices, these are exactly the peer counts.
        send_counts_1E = num_local_tokens_per_expert_E.to(torch.int32).view(1, -1)
        packed_source_indices_N = build_packed_source_indices(
            topk_expert_ids_TK.reshape(-1), send_counts_1E
        )
        return send_counts_1E, packed_source_indices_N

    def _dispatch_compact(
        self,
        routed_input_ND: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        token_indices_per_row_N: torch.Tensor,
        topk_scores_per_row_N: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
        recv_capacity: int,
    ) -> tuple[torch.Tensor, torch.Tensor, NeuronEPDispatchMetadata]:
        """num_local_experts == 1 compact dispatch: send only real rows.

        Builds a contiguous destination-major source-index vector. The packed kernel issues the
        count exchange before local packing and delays synchronization until the
        received counts are consumed, permitting the compiler to overlap them. It
        receives directly into the static ``[recv_capacity, H]`` expert buffer.
        """
        device = routed_input_ND.device
        send_counts_1E, packed_source_indices_N = self._build_packed_routing(
            topk_expert_ids_TK, num_local_tokens_per_expert_E
        )
        # recv_counts is a kernel OUTPUT (variable-count a2a-v derives it internally).
        routed_input_RD, recv_counts_1E = permute_a2av_dispatch_compact(
            routed_input_ND,
            packed_source_indices_N,
            send_counts_1E,
            recv_capacity,
            self._ep_ranks,
        )
        padded_counts_e = torch.full(
            (1,), recv_capacity, dtype=torch.int32, device=device
        )
        metadata = NeuronEPDispatchMetadata(
            token_indices_experts_sorted_N=token_indices_per_row_N,
            topk_scores_experts_sorted_N=topk_scores_per_row_N,
            send_counts_1E=send_counts_1E,
            recv_counts_1E=recv_counts_1E,
            compact_capacity=recv_capacity,
            packed_source_indices_N=packed_source_indices_N,
        )
        return routed_input_RD, padded_counts_e, metadata

    # pyrefly: ignore [bad-override]
    def dispatch(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, NeuronEPDispatchMetadata]:
        """Packed dispatch through the published NKILib A2AV API.

        The glue does not expert-sort on the send side. We expand ``x_TD`` to
        token-major ``routed_input_ND`` (row r = t*K + k = token t replicated for
        its k-th expert) and let the kernel gather rows from the packed vector.
        """
        if self.ep_mesh is None:
            # Local (EP=1) fallback: unchanged. LocalTokenDispatcher.dispatch still
            # does expert-sort on the send side and consumes num_local_tokens_per_expert_E
            # verbatim; only the EP path benefits from the token-major kernel-permute layout.
            return LocalTokenDispatcher.dispatch(
                self,
                x_TD,
                topk_scores_TK,
                topk_expert_ids_TK,
                num_local_tokens_per_expert_E,
            )
        assert self._ep_ranks is not None
        ep_size = self.ep_mesh.size()
        num_local_experts = self.num_experts // ep_size
        t = x_TD.shape[0]
        k = topk_expert_ids_TK.shape[1]
        if topk_expert_ids_TK.shape != topk_scores_TK.shape:
            raise ValueError(
                "neuron_ep requires matching top-k score and expert-id shapes; "
                f"got {tuple(topk_scores_TK.shape)} and "
                f"{tuple(topk_expert_ids_TK.shape)}."
            )
        expected_routing_shape = (t, self.top_k)
        if topk_expert_ids_TK.shape != expected_routing_shape:
            raise ValueError(
                "neuron_ep requires routing shape "
                f"{expected_routing_shape}. Got {tuple(topk_expert_ids_TK.shape)}."
            )
        if ep_size < 8 or ep_size % 4 != 0:
            raise ValueError(
                "packed neuron_ep requires ep_size >= 8 in complete four-rank "
                f"Trn3 device groups. Got {ep_size}."
            )
        n = t * k
        if num_local_experts != 1:
            raise ValueError(
                "packed neuron_ep currently requires num_local_experts == 1; "
                f"got num_experts={self.num_experts}, ep_size={ep_size}, "
                f"num_local_experts={num_local_experts}."
            )

        # Token-major expansion: row r = t*K + k is token t's embedding, repeated K
        # times contiguously. Only routing indices are sorted; the kernel gathers
        # the hidden-state rows via packed source indices.
        routed_input_ND = x_TD.repeat_interleave(k, dim=0)
        # Per-row metadata index-aligned with routed_input_ND rows:
        #   token_indices_per_row_N[r] = t   = origin token id (for combine scatter_add)
        #   topk_scores_per_row_N[r]   = s_tk = gate score for that (token, expert) pair
        # Kept in the NeuronEPDispatchMetadata fields named ``*_experts_sorted_N`` (a
        # historical name): the data is token-major, not expert-sorted, but combine
        # only needs 1:1 alignment with the kernel row order.
        arange_N = torch.arange(n, device=x_TD.device)
        token_indices_per_row_N = arange_N // k
        topk_scores_per_row_N = topk_scores_TK.reshape(-1)

        # The kernel's per-source recv_counts are also per-(source, expert) when
        # num_local_experts == 1.
        return self._dispatch_compact(
            routed_input_ND,
            topk_expert_ids_TK,
            token_indices_per_row_N,
            topk_scores_per_row_N,
            num_local_tokens_per_expert_E,
            self._packed_capacity(t, ep_size, num_local_experts),
        )

    # pyrefly: ignore [bad-override]
    def combine(
        self,
        routed_output_RD: torch.Tensor,
        metadata: NeuronEPDispatchMetadata,
        x_TD: torch.Tensor,
        *,
        num_local_tokens_after_padding: int,
        local_seq_len_after_padding: int,
    ) -> torch.Tensor:
        """Run packed NKI combine, apply scores, and reduce routed rows N -> tokens T.

        The EP path consumes the packed ``[C,H]`` expert output directly.
        Metadata's ``*_experts_sorted_N`` fields carry token-major (per-row)
        values -- see ``dispatch`` -- but combine treats them as opaque
        row-aligned indices/scores.

        With SP (``sp_size > 1``) each rank holds only a ``1/sp_size`` slice of the
        sequence, so ``token_indices_experts_sorted_N`` are LOCAL token ids. They are
        offset to GLOBAL positions here and scattered into a global-length buffer, so
        every SP rank writes a DISJOINT slice (zeros elsewhere). The MoE forward output
        is declared ``Partial(tp) -> Shard(seq)`` in ``set_moe_sharding_config``, and
        that DTensor redistribution reduce-scatters these disjoint slices back into a
        per-rank sequence shard -- which is why the buffer must be global-length and why
        no TP collective belongs here. This mirrors ``AllToAllTokenDispatcher.combine``
        (the SP remap is combine-only in both: dispatch keeps indices local because it
        indexes local tensors).
        """
        if self.ep_mesh is None:
            return LocalTokenDispatcher.combine(
                self,
                routed_output_RD,
                metadata,
                x_TD,
                num_local_tokens_after_padding=num_local_tokens_after_padding,
                local_seq_len_after_padding=local_seq_len_after_padding,
            )
        assert self._ep_ranks is not None

        if routed_output_RD.shape[0] != metadata.compact_capacity:
            raise ValueError(
                "neuron_ep compact combine expects expert output with "
                f"capacity ({metadata.compact_capacity}) rows. "
                f"got {routed_output_RD.shape[0]}."
            )
        combined_ND = unpermute_a2av_combine_packed(
            routed_output_RD,
            metadata.packed_source_indices_N,
            metadata.send_counts_1E,
            metadata.recv_counts_1E,
            self._ep_ranks,
        )

        combined_ND = (
            combined_ND.to(torch.float32)
            * metadata.topk_scores_experts_sorted_N.reshape(-1, 1)
        ).to(x_TD.dtype)
        dim = x_TD.shape[-1]
        if self.sp_size > 1:
            # Global-length buffer: this rank fills only its own disjoint slice (via the
            # sp_rank offset below); the downstream Partial -> Shard(seq) redistribution
            # reassembles all SP ranks' slices. moe.py pads the sequence to a multiple of
            # sp_size before dispatch, which keeps the per-rank token count uniform --
            # the packed receive layout relies on that uniformity.
            out_TD = torch.zeros(
                num_local_tokens_after_padding * self.sp_size,
                dim,
                device=x_TD.device,
                dtype=x_TD.dtype,
            )
            token_indices_N = self._sp_global_token_indices(
                metadata.token_indices_experts_sorted_N,
                local_seq_len_after_padding,
            )
        else:
            out_TD = torch.zeros_like(x_TD)
            token_indices_N = metadata.token_indices_experts_sorted_N
        # TODO: Replace this with a fixed-order NKI token-reduction kernel
        # that provides deterministic scatter-add semantics on Neuron.
        # Native Neuron scatter-add is compilable but does not guarantee reduction order.
        # Functionally equivalent to Tensor.scatter_add in eager mode. Use the
        # explicit ATen overload so AOT capture preserves a Torch-MLIR-recognized
        # operator target.
        out_TD = torch.ops.aten.scatter_add.default(
            out_TD,
            0,
            token_indices_N.reshape(-1, 1).expand(-1, dim),
            combined_ND,
        )
        return out_TD


@dataclass(frozen=True, kw_only=True)
class DeepEPDispatchMetadata:
    """Metadata for DeepEP, HybridEP, and MinimalAsyncEP token dispatch."""

    state: object  # Backend-specific dispatch state.


class DeepEPTokenDispatcher(BaseEPTokenDispatcher):
    """Token dispatcher using DeepEP v2's unified ``ElasticBuffer`` dispatch/combine.

    DeepEP v2 (>= 2.0.0) collapses the v1 high-throughput (HT) and low-latency (LL)
    paths into a single ``buffer.dispatch``/``combine``. The compact, expert-grouped
    layout feeds the grouped-GEMM expert path directly (no permute). Combine is
    asynchronous -- callers must call sync_combine() before using the result.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseEPTokenDispatcher.Config):
        # Select the dispatch layout. False (default, also forced under autograd): compact,
        # host-synced, backward-able path for training. True: static, no-host-sync expand
        # layout so the MoE forward is cudagraph-capturable -- inference only (covers BOTH
        # prefill and decode, since both run under no_grad), no backward. The deepep
        # primitives gate on grad context, so a True spec falls back to compact in training.
        cudagraphable: bool = False
        # EXPAND (cudagraphable=True, inference) ONLY: the per-rank dispatch CAPACITY. It
        # fixes the static output-slab shape; tokens a rank sends beyond it are DROPPED
        # (masked layout), so set it >= the largest per-rank token count for droplessness
        # (e.g. max_num_batched_tokens / sp). None = unset: REQUIRED for the expand path, but IGNORED in compact
        # (cudagraphable=False) mode. training auto-sizes from the per-rank token count at
        # dispatch (always dropless), so it is left None there.
        num_max_tokens_per_rank: int | None = None
        # Model hidden dim, threaded by the builder so the expand buffer can be created
        # eagerly at wire_meshes (before any cudagraph capture). None until the builder sets it.
        hidden_dim: int | None = None

    def __init__(self, config: Config):
        super().__init__(config)
        self.num_max_tokens_per_rank = config.num_max_tokens_per_rank
        self.hidden_dim = config.hidden_dim
        self.cudagraphable = config.cudagraphable

        # Import to register custom ops so SAC saves communication outputs
        # instead of recomputing them. This must happen before apply_ac.
        from torchtitan.distributed.deepep import deepep  # noqa: F401

    def wire_meshes(self, *, ep_mesh=None, tp_mesh=None) -> None:
        """Wire EP/SP meshes. For the cudagraph (inference) path, EAGERLY create the
        ElasticBuffer so its construction-time barrier runs at parallelize time, never
        inside a CUDA graph capture. The compact (training) path skips this: it sizes the
        buffer from the actual per-rank token count at first dispatch (no capture, so the
        one-time construction barrier there is fine), which frees the user from setting
        num_max_tokens_per_rank for training.
        """
        super().wire_meshes(ep_mesh=ep_mesh, tp_mesh=tp_mesh)
        # TODO(unify-ep-buffers): move this eager buffer creation into an init_buffer() like
        # MinimalAsyncEPTokenDispatcher, and unify DeepEP / HybridEP / MinimalAsyncEP buffer setup.
        if self.cudagraphable and ep_mesh is not None:
            # Inference (expand) path: num_max_tokens_per_rank fixes the static dispatch slab,
            # so it must be set here; the compact/training path auto-sizes and leaves it None.
            if self.num_max_tokens_per_rank is None:
                raise ValueError(
                    "DeepEP cudagraphable (expand) dispatch requires num_max_tokens_per_rank "
                    " but it is None."
                )
            if self.hidden_dim is None:
                raise ValueError(
                    "DeepEP cudagraphable (expand) dispatch requires hidden_dim, but it is "
                    "None; the builder must thread it through the dispatcher config."
                )
            from torchtitan.distributed.deepep.deepep import get_buffer

            get_buffer(
                ep_mesh.get_group(),
                hidden=self.hidden_dim,
                num_max_tokens_per_rank=self.num_max_tokens_per_rank,
                num_topk=self.top_k,
            )

    # pyrefly: ignore [bad-override]
    def dispatch(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, DeepEPDispatchMetadata]:
        # Ignore input num_local_tokens_per_expert_E. DeepEP returns the number
        # of global routed tokens for every local expert using other inputs.
        del num_local_tokens_per_expert_E
        assert self.ep_mesh is not None, (
            "ep_mesh must be set before dispatch. "
            "ExpertParallel._partition_fn() should set it."
        )
        ep_group = self.ep_mesh.get_group()
        num_local_experts = self.num_experts // ep_group.size()

        from torchtitan.distributed.deepep.deepep import dispatch_tokens

        hidden_states_RD, num_global_tokens_per_local_expert_e, state = dispatch_tokens(
            x_TD,
            topk_expert_ids_TK,
            topk_scores_TK,
            num_local_experts,
            self.num_experts,
            ep_group,
            num_max_tokens_per_rank=self.num_max_tokens_per_rank,
            cudagraphable=self.cudagraphable,
        )

        metadata = DeepEPDispatchMetadata(state=state)
        return hidden_states_RD, num_global_tokens_per_local_expert_e, metadata

    # pyrefly: ignore [bad-override]
    def combine(
        self,
        routed_output_RD: torch.Tensor,
        metadata: DeepEPDispatchMetadata,
        x_TD: torch.Tensor,
        *,
        num_local_tokens_after_padding: int,
        local_seq_len_after_padding: int,
    ) -> torch.Tensor:
        """Combine tokens via DeepEP.

        When sp_size == 1, combine is async — sync_combine() is deferred
        to MoE.forward, enabling overlap with shared_experts.
        When sp_size > 1, there is no overlap: sync is forced here because
        the SP expansion must read the combine result before returning.
        """
        from torchtitan.distributed.deepep.deepep import combine_tokens, sync_combine

        # pyrefly: ignore [bad-argument-type]
        combined_TD = combine_tokens(routed_output_RD, metadata.state)

        if self.sp_size > 1:
            sync_combine()
            out_TD = torch.zeros(
                num_local_tokens_after_padding * self.sp_size,
                combined_TD.shape[-1],
                device=combined_TD.device,
                dtype=combined_TD.dtype,
            )
            local_indices = torch.arange(
                combined_TD.shape[0], device=combined_TD.device
            )
            global_indices = self._sp_global_token_indices(
                local_indices,
                local_seq_len_after_padding,
            )
            out_TD[global_indices] = combined_TD
            return out_TD

        return combined_TD


class HybridEPTokenDispatcher(BaseEPTokenDispatcher):
    """Token dispatcher using HybridEP for efficient token dispatch/combine.

    Uses HybridEP library kernels (GB200/NVLink72) instead of standard
    all-to-all collectives.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseEPTokenDispatcher.Config):
        """Config for HybridEP token dispatcher.

        Args:
            non_blocking_capacity_factor: Enable non-blocking HybridEP dispatch
                with a given capacity factor.

                Setting this to a float in (0, 1] enables CPU-free non-blocking
                dispatch and controls num_permuted_tokens — the fused-permute
                output capacity, estimated as:
                num_tokens × ep_size × min(num_local_experts, top_k) × cf,
                aligned for MXFP8.  Tokens whose permuted offset exceeds this
                limit are silently dropped (overflow_flag is set on GPU).

                - None = blocking mode (default).  HybridEP calls
                  cudaStreamSynchronize after dispatch, copies
                  tokens_per_expert to pinned CPU memory, and computes the
                  exact num_permuted_tokens on the host.  No token dropping.
                - 1.0 = non-blocking, worst-case sizing: every token can reach
                  every local expert, no drops, highest memory.
                - < 1.0 = non-blocking, reduced memory; controls the
                  fused-permute output tensor size (num_permuted_tokens).
                  Safe in practice when forced load balancing (e.g. aux-loss /
                  round-robin) keeps distribution roughly uniform.

                Note: this factor has no lasting effect on the all-to-all
                communication buffer.  HybridEP's dispatch_with_permute
                internally passes the actual num_tokens to
                update_template_config, which auto-grows the buffer to the
                full token count on the first dispatch regardless of this
                setting.
        """

        non_blocking_capacity_factor: float | None = None
        pad_multiple: int | None = None

    def __init__(self, config: Config):
        super().__init__(config)
        self.non_blocking_capacity_factor = config.non_blocking_capacity_factor
        self.pad_multiple = config.pad_multiple

        # Import to register custom ops so SAC saves communication outputs
        # instead of recomputing them. This must happen before apply_ac.
        from torchtitan.distributed.deepep import hybridep  # noqa: F401

    # pyrefly: ignore [bad-override]
    def dispatch(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, DeepEPDispatchMetadata]:
        # Ignore input num_local_tokens_per_expert_E. HybridEP returns the
        # number of global routed tokens for every local expert using other inputs.
        del num_local_tokens_per_expert_E
        assert self.ep_mesh is not None, (
            "ep_mesh must be set before dispatch. "
            "ExpertParallel._partition_fn() should set it."
        )
        ep_group = self.ep_mesh.get_group()
        num_local_experts = self.num_experts // ep_group.size()

        from torchtitan.distributed.deepep.hybridep import dispatch_tokens

        hidden_states_RD, num_global_tokens_per_local_expert_e, state = dispatch_tokens(
            x_TD,
            topk_expert_ids_TK,
            topk_scores_TK,
            num_local_experts,
            self.num_experts,
            ep_group,
            non_blocking_expert_capacity_factor=self.non_blocking_capacity_factor,
            pad_multiple=self.pad_multiple,
        )

        metadata = DeepEPDispatchMetadata(state=state)
        return hidden_states_RD, num_global_tokens_per_local_expert_e, metadata

    # pyrefly: ignore [bad-override]
    def combine(
        self,
        routed_output_RD: torch.Tensor,
        metadata: DeepEPDispatchMetadata,
        x_TD: torch.Tensor,
        *,
        num_local_tokens_after_padding: int,
        local_seq_len_after_padding: int,
    ) -> torch.Tensor:
        """Combine tokens via HybridEP."""
        from torchtitan.distributed.deepep import hybridep

        combined_TD = hybridep.combine_tokens(
            routed_output_RD,
            metadata.state,  # pyrefly: ignore [bad-argument-type]
            pad_multiple=self.pad_multiple,
        )

        if self.sp_size > 1:
            out_TD = torch.zeros(
                num_local_tokens_after_padding * self.sp_size,
                combined_TD.shape[-1],
                device=combined_TD.device,
                dtype=combined_TD.dtype,
            )
            local_indices = torch.arange(
                combined_TD.shape[0], device=combined_TD.device
            )
            global_indices = self._sp_global_token_indices(
                local_indices,
                local_seq_len_after_padding,
            )
            out_TD[global_indices] = combined_TD
            return out_TD

        return combined_TD


class MinimalAsyncEPTokenDispatcher(LocalTokenDispatcher):
    """Token dispatcher using MinimalAsyncEP for constrained EP communication.

    This first integration supports EP with ``sp_size == 1`` only. TP/SP, CP,
    PP, padding, and async combine overlap are intentionally out of scope.
    """

    ep_mesh: DeviceMesh | None
    sp_size: int
    hidden_dim: int | None
    tokens_per_rank: int | None
    dtype: torch.dtype | None
    buffer_device: torch.device

    @dataclass(kw_only=True, slots=True)
    class Config(LocalTokenDispatcher.Config):
        hidden_dim: int | None = None
        tokens_per_rank: int | None = None
        dtype: torch.dtype | None = None
        device: torch.device | None = None

    def __init__(self, config: Config):
        super().__init__(config)
        self.ep_mesh: DeviceMesh | None = None
        self.sp_size: int = 1
        self.hidden_dim = config.hidden_dim
        self.tokens_per_rank = config.tokens_per_rank
        self.dtype = config.dtype
        if config.device is None:
            buffer_device = torch.device(device_type, device_module.current_device())
        else:
            buffer_device = config.device
        # pyrefly: ignore [read-only]
        self.buffer_device = buffer_device

    # MinimalAsyncEP has one process-global buffer: the first dispatcher
    # initializes it, same-configuration dispatchers reuse it, and differing
    # metadata is invalid because the buffer layout would not match.
    _global_buffer_key: ClassVar[
        tuple[object, int, int, int, int, torch.dtype, torch.device] | None
    ] = None

    def wire_meshes(
        self,
        *,
        ep_mesh: DeviceMesh | None,
        tp_mesh: DeviceMesh | None,
    ) -> None:
        """Install the EP mesh used by MinimalAsyncEP dispatch / combine."""
        if ep_mesh is None:
            raise ValueError(
                "MinimalAsyncEPTokenDispatcher requires expert parallelism "
                "(ep_mesh must be set)."
            )
        del tp_mesh
        self.ep_mesh = ep_mesh
        self.sp_size = 1
        self.init_buffer()

    def init_buffer(self) -> None:
        """Initialize MinimalAsyncEP's process-local symmetric-memory buffer."""
        if self.ep_mesh is None:
            raise ValueError("MinimalAsyncEPTokenDispatcher requires an EP mesh.")
        missing_fields = [
            field
            for field, value in (
                ("hidden_dim", self.hidden_dim),
                ("tokens_per_rank", self.tokens_per_rank),
                ("dtype", self.dtype),
                ("device", self.buffer_device),
            )
            if value is None
        ]
        if missing_fields:
            raise ValueError(
                "MinimalAsyncEPTokenDispatcher.Config is missing "
                f"{', '.join(missing_fields)}. Call update_from_config() with a "
                "Trainer.Config before building the model."
            )

        assert self.hidden_dim is not None
        assert self.tokens_per_rank is not None
        assert self.dtype is not None
        assert self.buffer_device is not None

        ep_size = self.ep_mesh.size()
        ep_group = self.ep_mesh.get_group()

        num_local_experts = self.num_experts // ep_size
        buffer_key = (
            ep_group,
            self.hidden_dim,
            self.tokens_per_rank,
            num_local_experts,
            self.top_k,
            self.dtype,
            self.buffer_device,
        )
        if MinimalAsyncEPTokenDispatcher._global_buffer_key is not None:
            if MinimalAsyncEPTokenDispatcher._global_buffer_key != buffer_key:
                raise ValueError(
                    "MinimalAsyncEP buffer was already initialized with a "
                    "different configuration."
                )
            return

        minimal_async_ep_init_buffer(
            group=ep_group,
            hidden_dim=self.hidden_dim,
            tokens_per_rank=self.tokens_per_rank,
            num_local_experts=num_local_experts,
            top_k=self.top_k,
            dtype=self.dtype,
            device=self.buffer_device,
        )
        MinimalAsyncEPTokenDispatcher._global_buffer_key = buffer_key

    # pyrefly: ignore [bad-override]
    def dispatch(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, DeepEPDispatchMetadata]:
        """Dispatch tokens to expert ranks with MinimalAsyncEP.

        Args:
            x_TD, topk_scores_TK, topk_expert_ids_TK,
                num_local_tokens_per_expert_E: standard ``GroupedExperts``
                dispatch inputs; see ``torchtitan.models.common.moe`` for shape
                suffix definitions.

        Returns:
            routed_input_RD: local-expert rows for grouped-mm.
            num_tokens_per_local_expert_e: ``(num_local_experts,)`` token counts
            metadata: dispatch metadata for combine()
        """
        assert self.ep_mesh is not None, "ep_mesh must be set before dispatch"
        ep_group = self.ep_mesh.get_group()

        top_k = self.top_k
        routed_scores_N = topk_scores_TK.view(-1)

        ep_size = ep_group.size()
        num_tokens = x_TD.shape[0]
        num_local_experts = num_local_tokens_per_expert_E.numel() // ep_size
        # TODO(xmfan): make this capacity configurable by user
        num_receive_rows_per_source_rank = num_tokens * min(top_k, num_local_experts)
        receive_capacity = ep_size * num_receive_rows_per_source_rank

        (
            hidden_states_RD,
            dispatch_dst_ranks,  # local E-major row -> destination EP rank
            dispatch_dst_rows,  # local E-major row -> destination receive row
            combine_dst_ranks,  # received row -> origin EP rank; length R_max
            combine_dst_rows,  # received row -> origin E-major row; length R_max
            combine_num_valid_rows,  # actual received rows, device-side scalar
            E_row_to_T_row_N,  # local E-major row -> local T-major row
            T_row_to_E_row_N,  # local T-major row -> local E-major row
            num_tokens_per_local_expert_e,  # local expert -> active row count
        ) = minimal_async_ep_dispatch_op(
            x_TD,
            topk_expert_ids_TK,
            num_local_tokens_per_expert_E,
            receive_capacity,
            ep_size,
        )

        state = MinimalAsyncEPDispatchMetadata(
            dispatch_dst_ranks=dispatch_dst_ranks,
            dispatch_dst_rows=dispatch_dst_rows,
            combine_dst_ranks=combine_dst_ranks,
            combine_dst_rows=combine_dst_rows,
            combine_num_valid_rows=combine_num_valid_rows,
            E_row_to_T_row=E_row_to_T_row_N,
            T_row_to_E_row=T_row_to_E_row_N,
            routed_scores=routed_scores_N,
            num_tokens=num_tokens,
            top_k=top_k,
        )

        metadata = DeepEPDispatchMetadata(state=state)
        return hidden_states_RD, num_tokens_per_local_expert_e, metadata

    # pyrefly: ignore [bad-override]
    def combine(
        self,
        routed_output_RD: torch.Tensor,
        metadata: DeepEPDispatchMetadata,
        x_TD: torch.Tensor,
        *,
        num_local_tokens_after_padding: int,
        local_seq_len_after_padding: int,
    ) -> torch.Tensor:
        """Combine tokens via MinimalAsyncEP."""
        del num_local_tokens_after_padding, local_seq_len_after_padding
        state = cast(MinimalAsyncEPDispatchMetadata, metadata.state)
        combined_TD, _routed_output_ND = minimal_async_ep_combine_op(  # noqa: N806
            routed_output_RD,
            state.dispatch_dst_ranks,
            state.dispatch_dst_rows,
            state.combine_dst_ranks,
            state.combine_dst_rows,
            state.combine_num_valid_rows,
            state.T_row_to_E_row,
            state.E_row_to_T_row,
            state.routed_scores,
            state.num_tokens,
            state.top_k,
        )
        return combined_TD
