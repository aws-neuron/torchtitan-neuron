# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

"""Unit tests for the neuron_ep token dispatcher layout logic.

These cover the CPU-computable, non-kernel parts of NeuronEPTokenDispatcher:
packed routing-vector forwarding, capacity validation, mesh-wiring validation, data
provenance across ranks, the EP=64 metadata contract, and the EP=1 local-dispatch
fallback. The NKI ``permute_a2av`` / ``unpermute_a2av`` kernels themselves are
exercised on Trainium hardware; they are not invoked here.
"""

import unittest
import unittest.mock

import torch

from torchtitan.distributed.neuron_ep import gmm_align
from torchtitan.models.common.token_dispatcher import (
    NeuronEPDispatchMetadata,
    NeuronEPTokenDispatcher,
)


def _make_dispatcher(num_experts: int, top_k: int, ep_size: int | None):
    cfg = NeuronEPTokenDispatcher.Config(num_experts=num_experts, top_k=top_k)
    dispatcher = NeuronEPTokenDispatcher(cfg)
    if ep_size is None:
        dispatcher.ep_mesh = None
    else:
        mock_mesh = unittest.mock.MagicMock()
        mock_mesh.size.return_value = ep_size
        dispatcher.ep_mesh = mock_mesh
        dispatcher._ep_ranks = [list(range(ep_size))]
    return dispatcher


def _counts_from_ids(topk_ids_TK: torch.Tensor, num_experts: int) -> torch.Tensor:
    counts = torch.zeros(num_experts, dtype=torch.int64)
    for e in topk_ids_TK.reshape(-1):
        counts[int(e)] += 1
    return counts


def setUpModule():  # noqa: N802
    # The dispatcher tests use an independent CPU oracle at the NKI boundary.
    # Tile-prefix construction itself is covered by on-device kernel tests.
    def packed_indices(ids, counts):
        expected_counts = torch.bincount(ids.long(), minlength=counts.numel())
        torch.testing.assert_close(counts.flatten().long(), expected_counts)
        return torch.argsort(ids, stable=True).to(torch.int32)

    patcher = unittest.mock.patch(
        "torchtitan.models.common.token_dispatcher.build_packed_source_indices",
        side_effect=packed_indices,
    )
    patcher.start()
    unittest.addModuleCleanup(patcher.stop)


class TestWireMeshesValidation(unittest.TestCase):
    def test_accepts_sp(self):
        """TP/SP is supported: wire_meshes records the SP coordinates."""
        disp = _make_dispatcher(128, 8, ep_size=None)
        tp_mesh = unittest.mock.MagicMock()
        tp_mesh.size.return_value = 4
        tp_mesh._sym_get_coordinate.return_value = 2
        disp.wire_meshes(ep_mesh=None, tp_mesh=tp_mesh)
        self.assertEqual(disp.sp_size, 4)
        self.assertEqual(disp.sp_rank, 2)

    def test_rejects_indivisible_experts(self):
        disp = _make_dispatcher(num_experts=127, top_k=8, ep_size=None)
        ep_mesh = unittest.mock.MagicMock()
        ep_mesh.size.return_value = 4
        with self.assertRaisesRegex(ValueError, "divisible"):
            disp.wire_meshes(ep_mesh=ep_mesh, tp_mesh=None)


class TestEP1Fallback(unittest.TestCase):
    """With ep_mesh None, dispatch/combine reduce to LocalTokenDispatcher."""

    def test_dispatch_combine_roundtrip(self):
        torch.manual_seed(0)
        num_experts, top_k = 8, 2
        T, D = 6, 16
        disp = _make_dispatcher(num_experts, top_k, ep_size=None)

        x_TD = torch.randn(T, D)
        scores_TK = torch.rand(T, top_k)
        ids_TK = torch.randint(0, num_experts, (T, top_k))
        counts_E = _counts_from_ids(ids_TK, num_experts)

        routed_RD, counts_e, md = disp.dispatch(x_TD, scores_TK, ids_TK, counts_E)
        # No all-to-all: R == N == T*top_k.
        self.assertEqual(routed_RD.shape[0], T * top_k)

        # Identity experts -> combine reproduces the score-weighted top-k sum.
        out_TD = disp.combine(
            routed_RD,
            md,
            x_TD,
            num_local_tokens_after_padding=0,
            local_seq_len_after_padding=0,
        )
        self.assertEqual(out_TD.shape, x_TD.shape)
        expected = (x_TD.unsqueeze(1) * scores_TK.unsqueeze(-1)).sum(dim=1)
        torch.testing.assert_close(out_TD, expected, rtol=1e-4, atol=1e-4)


def _make_compact_dispatcher(
    num_experts,
    top_k,
    ep_size,
    *,
    capacity_per_expert=None,
):
    """Dispatcher configured for the supported packed path."""
    cfg = NeuronEPTokenDispatcher.Config(
        num_experts=num_experts,
        top_k=top_k,
        capacity_per_expert=capacity_per_expert,
    )
    disp = NeuronEPTokenDispatcher(cfg)
    mock_mesh = unittest.mock.MagicMock()
    mock_mesh.size.return_value = ep_size
    disp.ep_mesh = mock_mesh
    disp._ep_ranks = [list(range(ep_size))]
    return disp


class TestOptimizedCompactTransmission(unittest.TestCase):
    """Compact (variable-count) dispatch for num_local_experts == 1.

    Sends only real per-destination rows and receives them at cumulative source
    offsets in an FSDP/global-token-bound capacity buffer.

    Requires nle == 1 (EP == num_experts): only then are the kernel's per-source
    recv_counts also per-(source, expert), which the expert buffer needs.
    """

    def test_packed_capacity_matches_fsdp_batch_sequence_bound(self):
        batch_size = 2
        local_seq_len = 512
        sp_size = 4
        fsdp_size = 2
        ep_size = fsdp_size * sp_size
        disp = _make_compact_dispatcher(ep_size, 4, ep_size)
        capacity = disp._packed_capacity(batch_size * local_seq_len, ep_size, 1)
        expected = fsdp_size * batch_size * (local_seq_len * sp_size)
        self.assertEqual(capacity, expected)

    def test_packed_capacity_rounds_model_bound_for_gmm(self):
        ep_size, top_k, num_local_tokens = 8, 4, 129
        disp = _make_compact_dispatcher(ep_size, top_k, ep_size)
        capacity = disp._packed_capacity(num_local_tokens, ep_size, 1)
        unaligned_bound = num_local_tokens * ep_size
        self.assertGreaterEqual(capacity, unaligned_bound)
        self.assertLess(capacity - unaligned_bound, gmm_align())
        self.assertEqual(capacity % gmm_align(), 0)

    def test_packed_capacity_is_independent_of_top_k(self):
        num_local_tokens = 1024
        ep_size = 64
        capacities = [
            _make_compact_dispatcher(ep_size, top_k, ep_size)._packed_capacity(
                num_local_tokens, ep_size, 1
            )
            for top_k in (1, 2, 4, 8)
        ]
        self.assertEqual(capacities, [num_local_tokens * ep_size] * 4)

    def test_capacity_override_retains_per_source_semantics(self):
        ep_size = 8
        disp = _make_compact_dispatcher(
            ep_size,
            4,
            ep_size,
            capacity_per_expert=1024,
        )
        self.assertEqual(disp._packed_capacity(1024, ep_size, 1), ep_size * 1024)

    def test_capacity_override_rejects_truncating_bound(self):
        ep_size = 8
        disp = _make_compact_dispatcher(
            ep_size,
            4,
            ep_size,
            capacity_per_expert=127,
        )
        with self.assertRaisesRegex(ValueError, "model capacity"):
            disp._packed_capacity(1024, ep_size, 1)

    def test_dispatch_rejects_mismatched_routing_shapes(self):
        disp = _make_compact_dispatcher(4, 2, 4)
        x = torch.randn(3, 8)
        ids = torch.tensor([[0, 1], [1, 2], [2, 3]])
        scores = torch.randn(3, 1)
        with self.assertRaisesRegex(ValueError, "matching top-k"):
            disp.dispatch(x, scores, ids, torch.zeros(4, dtype=torch.int32))

    def test_dispatch_rejects_routing_width_that_differs_from_config(self):
        disp = _make_compact_dispatcher(4, 2, 4)
        x = torch.randn(3, 8)
        ids = torch.tensor([[0, 1, 2], [1, 2, 3], [0, 2, 3]])
        scores = torch.randn(3, 3)
        with self.assertRaisesRegex(ValueError, "routing shape \\(3, 2\\)"):
            disp.dispatch(x, scores, ids, torch.zeros(4, dtype=torch.int32))

    def test_dispatch_rejects_routing_token_count_that_differs_from_input(self):
        disp = _make_compact_dispatcher(4, 2, 4)
        x = torch.randn(3, 8)
        ids = torch.tensor([[0, 1], [1, 2]])
        scores = torch.randn(2, 2)
        with self.assertRaisesRegex(ValueError, "routing shape \\(3, 2\\)"):
            disp.dispatch(x, scores, ids, torch.zeros(4, dtype=torch.int32))

    def test_combine_rejects_output_that_does_not_match_compact_capacity(self):
        ep_size, capacity, hidden = 4, 3, 8
        disp = _make_compact_dispatcher(
            num_experts=ep_size,
            top_k=2,
            ep_size=ep_size,
        )
        metadata = NeuronEPDispatchMetadata(
            token_indices_experts_sorted_N=torch.empty(0, dtype=torch.int64),
            topk_scores_experts_sorted_N=torch.empty(0),
            packed_source_indices_N=torch.empty(0, dtype=torch.int32),
            send_counts_1E=torch.zeros(1, ep_size, dtype=torch.int32),
            recv_counts_1E=torch.zeros(1, ep_size, dtype=torch.int32),
            compact_capacity=capacity,
        )
        routed_output = torch.empty(capacity - 1, hidden)

        with self.assertRaisesRegex(ValueError, "capacity"):
            disp.combine(
                routed_output,
                metadata,
                torch.empty(1, hidden),
                num_local_tokens_after_padding=1,
                local_seq_len_after_padding=1,
            )

    def _packed_permute_model(
        self, routed_ND, send_vector, send_cnt, ep_size, n, capacity, D
    ):
        """CPU model of explicit-displacement packed dispatch."""
        # send_rows[(s, d)] = the real rows source s sends to dest d.
        send_rows = {}
        for s in range(ep_size):
            for d in range(ep_size):
                cnt = int(send_cnt[s][0, d])
                rows = []
                offset = int(send_cnt[s][0, :d].sum())
                for slot in range(cnt):
                    src_row = int(send_vector[s][offset + slot])
                    self.assertLess(src_row, n)
                    rows.append(routed_ND[s][src_row])
                send_rows[(s, d)] = rows
        recv = {}
        for d in range(ep_size):
            recv_counts = [len(send_rows[(s, d)]) for s in range(ep_size)]
            total = sum(recv_counts)
            self.assertLessEqual(total, capacity)
            buf = torch.zeros(capacity, D)
            offset = 0
            for s in range(ep_size):
                for j, row in enumerate(send_rows[(s, d)]):
                    buf[offset + j] = row
                offset += recv_counts[s]
            recv[d] = (
                buf,
                torch.tensor(recv_counts, dtype=torch.int32).view(1, ep_size),
            )
        return recv

    def test_packed_routing_and_receive_offsets(self):
        torch.manual_seed(0)
        # nle == 1: num_experts == ep_size.
        num_experts = ep_size = 8
        top_k = 4
        T, D = 12, 16
        n = T * top_k
        capacity = 2 * n

        disp = [
            _make_compact_dispatcher(num_experts, top_k, ep_size)
            for _ in range(ep_size)
        ]
        x = [torch.randn(T, D) for _ in range(ep_size)]
        # Distinct top_k experts per token (precondition) so each expert gets <= T rows.
        ids = [
            torch.stack([torch.randperm(num_experts)[:top_k] for _ in range(T)])
            for _ in range(ep_size)
        ]
        routed_ND = [x[r].repeat_interleave(top_k, dim=0) for r in range(ep_size)]

        # Peer counts partition the vector, preserving each destination's arrival order.
        send_vector, send_cnt = [], []
        for r in range(ep_size):
            sc, si = disp[r]._build_packed_routing(
                ids[r], _counts_from_ids(ids[r], ep_size)
            )
            send_vector.append(si)
            send_cnt.append(sc)
            self.assertEqual(int(sc.sum()), n)
            flat_ids = ids[r].reshape(-1)
            for d in range(ep_size):
                cnt = int(sc[0, d])
                self.assertEqual(cnt, int((flat_ids == d).sum()))  # dest == expert
                offset = int(sc[0, :d].sum())
                expected = torch.where(flat_ids == d)[0].to(torch.int32)
                torch.testing.assert_close(si[offset : offset + cnt], expected)
            torch.testing.assert_close(
                si.sort().values, torch.arange(n, dtype=torch.int32)
            )

        # 2. Incoming source chunks are adjacent at cumulative receive offsets.
        recv = self._packed_permute_model(
            routed_ND, send_vector, send_cnt, ep_size, n, capacity, D
        )
        for d in range(ep_size):
            recv_buf, recv_counts_1E = recv[d]
            rc = recv_counts_1E.reshape(-1)
            offset = 0
            for s in range(ep_size):
                cs = int(rc[s])
                for j in range(cs):
                    source_offset = int(send_cnt[s][0, :d].sum())
                    expected = routed_ND[s][int(send_vector[s][source_offset + j])]
                    torch.testing.assert_close(recv_buf[offset + j], expected)
                offset += cs
            self.assertTrue(bool((recv_buf[offset:] == 0).all()))

    def test_compact_end_to_end_matches_reference(self):
        # Full dispatch->expert->combine round-trip under the packed receive model,
        # with distinct per-expert weights so a misroute changes the value.
        torch.manual_seed(3)
        num_experts = ep_size = 8
        top_k = 4
        T, D = 12, 16
        n = T * top_k
        capacity = 2 * n
        E = num_experts

        def expert_fn(row, g):
            return row * (1.0 + g / E)

        disp = [
            _make_compact_dispatcher(num_experts, top_k, ep_size)
            for _ in range(ep_size)
        ]
        x = [torch.randn(T, D) for _ in range(ep_size)]
        ids = [
            torch.stack([torch.randperm(num_experts)[:top_k] for _ in range(T)])
            for _ in range(ep_size)
        ]
        scores = [torch.rand(T, top_k) for _ in range(ep_size)]
        routed_ND = [x[r].repeat_interleave(top_k, dim=0) for r in range(ep_size)]
        tok_idx = [torch.arange(n) // top_k for _ in range(ep_size)]
        tok_scores = [scores[r].reshape(-1) for r in range(ep_size)]

        send_vector, send_cnt = [], []
        for r in range(ep_size):
            sc, si = disp[r]._build_packed_routing(
                ids[r], _counts_from_ids(ids[r], ep_size)
            )
            send_vector.append(si)
            send_cnt.append(sc)

        recv = self._packed_permute_model(
            routed_ND, send_vector, send_cnt, ep_size, n, capacity, D
        )

        expert_out_packed = {}
        for d in range(ep_size):
            recv_buf, recv_counts_1E = recv[d]
            expert_out_packed[d] = (expert_fn(recv_buf, d), recv_counts_1E)

        # Reverse A2AV reads this origin's chunk at its cumulative source offset.
        out = [torch.zeros(T, D) for _ in range(ep_size)]
        for o in range(ep_size):
            combined_ND = torch.zeros(n, D)
            for d in range(ep_size):
                out_packed, recv_counts_1E = expert_out_packed[d]
                cnt = int(send_cnt[o][0, d])
                offset = int(recv_counts_1E[0, :o].sum())
                for slot in range(cnt):
                    source_offset = int(send_cnt[o][0, :d].sum())
                    src_row = int(send_vector[o][source_offset + slot])
                    combined_ND[src_row] = out_packed[offset + slot]
            combined_ND = combined_ND * tok_scores[o].reshape(-1, 1)
            out[o] = torch.zeros(T, D).index_add(0, tok_idx[o], combined_ND)

        for o in range(ep_size):
            expected = torch.zeros(T, D)
            for t in range(T):
                for kk in range(top_k):
                    g = int(ids[o][t, kk])
                    expected[t] += scores[o][t, kk] * expert_fn(x[o][t], g)
            torch.testing.assert_close(out[o], expected, rtol=1e-4, atol=1e-4)

    def test_compact_rejects_nle_gt_1(self):
        disp = _make_compact_dispatcher(num_experts=128, top_k=8, ep_size=64)  # nle=2
        T = 16
        x = torch.randn(T, 8)
        ids = torch.stack([torch.randperm(128)[:8] for _ in range(T)])
        scores = torch.rand(T, 8)
        counts = _counts_from_ids(ids, 128)
        with self.assertRaisesRegex(ValueError, "num_local_experts == 1"):
            disp.dispatch(x, scores, ids, counts)

    def test_ep_below_kernel_minimum_is_rejected(self):
        ep_size = 4
        disp = _make_compact_dispatcher(ep_size, 2, ep_size)
        T = 4
        x = torch.randn(T, 8)
        ids = torch.stack([torch.randperm(ep_size)[:2] for _ in range(T)])
        scores = torch.rand(T, 2)
        counts = _counts_from_ids(ids, ep_size)
        with self.assertRaisesRegex(ValueError, "ep_size >= 8"):
            disp.dispatch(x, scores, ids, counts)

    def test_ep_without_complete_four_rank_devices_is_rejected(self):
        ep_size = 10
        disp = _make_compact_dispatcher(ep_size, 2, ep_size)
        T = 4
        x = torch.randn(T, 8)
        ids = torch.stack([torch.randperm(ep_size)[:2] for _ in range(T)])
        scores = torch.rand(T, 2)
        counts = _counts_from_ids(ids, ep_size)
        with self.assertRaisesRegex(ValueError, "complete four-rank"):
            disp.dispatch(x, scores, ids, counts)


class TestSequenceParallelCombine(unittest.TestCase):
    """combine() under TP/SP (sp_size > 1) scatters to GLOBAL token positions.

    With SP each rank holds a 1/sp_size slice of the sequence, so combine must write
    into a global-length buffer at the offsets given by ``_sp_global_token_indices``
    (blocked layout: local position p of batch row b -> b*global_seq + sp_rank*L + p),
    leaving every other rank's slice zero. The downstream Partial -> Shard(seq)
    redistribution reassembles the slices, so a rank writing outside its own slice (or
    a local-length buffer) would silently corrupt the sequence.

    NOTE on coverage: the ``_run`` helper below drives the EP=1 (``ep_mesh=None``) path,
    which returns from ``combine`` via ``LocalTokenDispatcher`` BEFORE the sp_size > 1
    branch -- so it pins the delegation contract only, NOT the SP scatter.
    ``TestSequenceParallelCombineEPPath`` covers the real branch.
    """

    def _run(self, *, sp_size, sp_rank, B, local_L, D=8, num_experts=8, top_k=2):
        torch.manual_seed(0)
        disp = _make_dispatcher(num_experts, top_k, ep_size=None)
        tp_mesh = unittest.mock.MagicMock()
        tp_mesh.size.return_value = sp_size
        tp_mesh._sym_get_coordinate.return_value = sp_rank
        disp.wire_meshes(ep_mesh=None, tp_mesh=tp_mesh)
        self.assertEqual(disp.sp_size, sp_size)

        t_local = B * local_L
        x_TD = torch.randn(t_local, D)
        scores_TK = torch.rand(t_local, top_k)
        ids_TK = torch.randint(0, num_experts, (t_local, top_k))
        counts_E = _counts_from_ids(ids_TK, num_experts)

        routed_RD, _counts_e, md = disp.dispatch(x_TD, scores_TK, ids_TK, counts_E)
        out = disp.combine(
            routed_RD,
            md,
            x_TD,
            num_local_tokens_after_padding=t_local,
            local_seq_len_after_padding=local_L,
        )
        expected_local = (x_TD.unsqueeze(1) * scores_TK.unsqueeze(-1)).sum(dim=1)
        return out, expected_local

    def test_ep1_sp_falls_back_to_local_shape(self):
        """EP=1 delegates to LocalTokenDispatcher: local-length, no SP offset.

        Guards the delegation contract -- the SP-aware global scatter lives in the EP
        path; the EP=1 fallback stays local-length (its own scatter target is x_TD).
        """
        out, expected = self._run(sp_size=4, sp_rank=1, B=2, local_L=3)
        self.assertEqual(out.shape, expected.shape)
        torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)

    def test_global_scatter_positions_blocked_layout(self):
        """The SP remap places local ids at the expected global rows, disjointly."""
        sp_size, sp_rank, B, local_L = 4, 2, 2, 3
        disp = _make_dispatcher(8, 2, ep_size=None)
        tp_mesh = unittest.mock.MagicMock()
        tp_mesh.size.return_value = sp_size
        tp_mesh._sym_get_coordinate.return_value = sp_rank
        disp.wire_meshes(ep_mesh=None, tp_mesh=tp_mesh)

        local_ids = torch.arange(B * local_L)
        global_ids = disp._sp_global_token_indices(local_ids, local_L)
        global_seq = local_L * sp_size
        expected = torch.tensor(
            [
                (i // local_L) * global_seq + sp_rank * local_L + (i % local_L)
                for i in range(B * local_L)
            ]
        )
        torch.testing.assert_close(global_ids, expected)
        # Every index lands inside this rank's own slice of each batch row, and within
        # the global buffer -- i.e. ranks write disjoint regions.
        self.assertTrue(int(global_ids.max()) < B * global_seq)
        for i, g in enumerate(global_ids.tolist()):
            pos_in_row = g % global_seq
            self.assertGreaterEqual(pos_in_row, sp_rank * local_L)
            self.assertLess(pos_in_row, (sp_rank + 1) * local_L)
            self.assertEqual(g // global_seq, i // local_L)  # batch row preserved

    def test_sp_rank0_matches_no_sp_prefix(self):
        """At sp_rank=0 the global ids of batch row 0 are the identity prefix."""
        disp = _make_dispatcher(8, 2, ep_size=None)
        tp_mesh = unittest.mock.MagicMock()
        tp_mesh.size.return_value = 2
        tp_mesh._sym_get_coordinate.return_value = 0
        disp.wire_meshes(ep_mesh=None, tp_mesh=tp_mesh)
        ids = torch.arange(4)  # B=1, local_L=4
        torch.testing.assert_close(disp._sp_global_token_indices(ids, 4), ids)


class TestSequenceParallelCombineEPPath(unittest.TestCase):
    """combine() SP branch on the REAL EP path (ep_mesh set), with kernels stubbed.

    This is the test that actually executes the sp_size > 1 code: the global-length
    output buffer and the ``_sp_global_token_indices`` remap of the final scatter. The
    EP=1 tests above return early via LocalTokenDispatcher and never reach it, so a
    regression to ``torch.zeros_like(x_TD)`` + raw local indices would not be caught
    without this.

    The packed NKI dispatch and combine kernels are replaced with CPU stubs:
    dispatch packs this rank's rows by destination, and combine restores their
    original order. That leaves the index/scatter logic under test without
    needing Trainium or ``torch_neuronx``. ``ep_mesh`` / ``_ep_ranks`` are set
    directly rather than through ``wire_meshes``, which would call
    ``build_ep_replica_group_ranks`` (needs
    ``torch_neuronx.distributed.mesh_registry``).
    """

    def _make(self, *, num_experts, top_k, ep_size, sp_size, sp_rank):
        disp = _make_compact_dispatcher(num_experts, top_k, ep_size)
        disp.sp_size = sp_size
        disp.sp_rank = sp_rank
        return disp

    def _run_ep_sp(self, *, sp_size, sp_rank, B, local_L, D=8, num_experts=8, top_k=2):
        """Drive dispatch+combine through the EP path with stubbed kernels."""
        torch.manual_seed(0)
        ep_size = num_experts  # nle == 1 -> compact path
        disp = self._make(
            num_experts=num_experts,
            top_k=top_k,
            ep_size=ep_size,
            sp_size=sp_size,
            sp_rank=sp_rank,
        )
        t_local = B * local_L
        n = t_local * top_k
        x_TD = torch.randn(t_local, D)
        scores_TK = torch.rand(t_local, top_k)
        ids_TK = torch.stack(
            [torch.randperm(num_experts)[:top_k] for _ in range(t_local)]
        )
        counts_E = _counts_from_ids(ids_TK, num_experts)

        mod = "torchtitan.models.common.token_dispatcher"

        def fake_dispatch_compact(
            routed_ND,
            packed_source_indices_N,
            send_cnt_1E,
            recv_capacity,
            _ranks,
        ):
            recv = routed_ND.new_zeros((recv_capacity, routed_ND.shape[-1]))
            counts = send_cnt_1E.clone()
            offset = 0
            for d in range(ep_size):
                for j in range(int(send_cnt_1E[0, d])):
                    recv[offset + j] = routed_ND[
                        int(packed_source_indices_N[offset + j])
                    ]
                offset += int(send_cnt_1E[0, d])
            return recv, counts

        def fake_unpermute(
            output_RD,
            packed_source_indices_N,
            send_cnt_1E,
            _rc,
            _ranks,
        ):
            assert packed_source_indices_N is not None
            out = output_RD.new_zeros((n, output_RD.shape[-1]))
            offset = 0
            for d in range(ep_size):
                for j in range(int(send_cnt_1E[0, d])):
                    out[int(packed_source_indices_N[offset + j])] = output_RD[
                        offset + j
                    ]
                offset += int(send_cnt_1E[0, d])
            return out

        with (
            unittest.mock.patch(
                f"{mod}.permute_a2av_dispatch_compact", fake_dispatch_compact
            ),
            unittest.mock.patch(f"{mod}.unpermute_a2av_combine_packed", fake_unpermute),
        ):
            routed_RD, _counts_e, md = disp.dispatch(x_TD, scores_TK, ids_TK, counts_E)
            out = disp.combine(
                routed_RD,
                md,
                x_TD,
                num_local_tokens_after_padding=t_local,
                local_seq_len_after_padding=local_L,
            )
        # Identity experts -> each token's contribution is its score-weighted sum.
        expected_rows = (x_TD.unsqueeze(1) * scores_TK.unsqueeze(-1)).sum(dim=1)
        return disp, out, expected_rows, t_local

    def test_output_is_global_length(self):
        """The SP buffer spans the WHOLE sequence, not just this rank's shard."""
        sp_size, local_L, B = 4, 3, 2
        _disp, out, _exp, t_local = self._run_ep_sp(
            sp_size=sp_size, sp_rank=2, B=B, local_L=local_L
        )
        self.assertEqual(out.shape[0], t_local * sp_size)

    def test_rows_land_at_global_offsets_and_elsewhere_zero(self):
        """Each local token lands at b*global_seq + sp_rank*L + p; rest is zero."""
        sp_size, sp_rank, B, local_L = 4, 2, 2, 3
        disp, out, expected_rows, t_local = self._run_ep_sp(
            sp_size=sp_size, sp_rank=sp_rank, B=B, local_L=local_L
        )
        global_seq = local_L * sp_size
        written = set()
        for i in range(t_local):
            g = (i // local_L) * global_seq + sp_rank * local_L + (i % local_L)
            written.add(g)
            torch.testing.assert_close(out[g], expected_rows[i], rtol=1e-4, atol=1e-4)
        # Every row outside this rank's slice must be untouched (zero), so the
        # downstream Partial -> Shard(seq) reduce cannot double-count.
        for g in range(out.shape[0]):
            if g not in written:
                self.assertTrue(
                    torch.equal(out[g], torch.zeros_like(out[g])),
                    f"row {g} outside this rank's slice was written",
                )

    def test_sp1_is_local_length_and_unchanged(self):
        """sp_size == 1 keeps the validated pure-EP behaviour: local-length output."""
        _disp, out, expected_rows, t_local = self._run_ep_sp(
            sp_size=1, sp_rank=0, B=1, local_L=6
        )
        self.assertEqual(out.shape[0], t_local)
        torch.testing.assert_close(out, expected_rows, rtol=1e-4, atol=1e-4)

    def test_distinct_sp_ranks_write_disjoint_slices(self):
        """Two SP ranks of the same sequence never write the same global row."""
        sp_size, B, local_L = 4, 1, 4
        touched = []
        for sp_rank in (0, 3):
            _d, out, _e, t_local = self._run_ep_sp(
                sp_size=sp_size, sp_rank=sp_rank, B=B, local_L=local_L
            )
            touched.append({g for g in range(out.shape[0]) if out[g].abs().sum() > 0})
        self.assertTrue(touched[0], "rank 0 wrote nothing")
        self.assertTrue(touched[1], "rank 3 wrote nothing")
        self.assertEqual(touched[0] & touched[1], set(), "SP slices overlap")


class TestPackedSourceIndices(unittest.TestCase):
    def test_full_size_routing_avoids_sparse_tables(self):
        generator = torch.Generator().manual_seed(42)
        for ep in (64, 128):
            with self.subTest(ep=ep):
                dispatcher = _make_dispatcher(ep, 8, ep)
                ids = torch.stack(
                    [torch.randperm(ep, generator=generator)[:8] for _ in range(1024)]
                )
                flat_ids = ids.flatten()
                with unittest.mock.patch(
                    "torch.nn.functional.one_hot",
                    side_effect=AssertionError(
                        "Packed routing must not build a sparse table"
                    ),
                ):
                    counts, vector = dispatcher._build_packed_routing(
                        ids, _counts_from_ids(ids, ep)
                    )
                expected = torch.cat(
                    [torch.where(flat_ids == peer)[0] for peer in range(ep)]
                )
                torch.testing.assert_close(vector.long(), expected, rtol=0, atol=0)
                torch.testing.assert_close(
                    counts.flatten().long(),
                    torch.bincount(flat_ids, minlength=ep),
                    rtol=0,
                    atol=0,
                )

    def test_ep64_dispatch_metadata_contract(self):
        ep_size, top_k, tokens, hidden = 64, 8, 17, 4
        dispatcher = _make_dispatcher(ep_size, top_k, ep_size)
        n = tokens * top_k
        inputs = torch.arange(tokens * hidden, dtype=torch.float32).reshape(
            tokens, hidden
        )
        ids = (torch.arange(n) % ep_size).reshape(tokens, top_k)
        scores = torch.arange(n, dtype=torch.float32).reshape(tokens, top_k)
        expected_counts = torch.bincount(ids.reshape(-1), minlength=ep_size).to(
            torch.int32
        )
        expected_recv_counts = expected_counts.clone().reshape(1, ep_size)
        expected_recv_counts[0, 1] += 1
        expected_recv_counts[0, 2] -= 1
        expected_vector = torch.tensor(
            sorted(range(n), key=lambda row: int(ids.reshape(-1)[row])),
            dtype=torch.int32,
        )
        expected_capacity = dispatcher._packed_capacity(tokens, ep_size, 1)

        def fake_permute(routed, vector, counts, capacity, ranks):
            torch.testing.assert_close(routed, inputs.repeat_interleave(top_k, dim=0))
            torch.testing.assert_close(counts.reshape(-1), expected_counts)
            torch.testing.assert_close(vector, expected_vector)
            self.assertEqual(capacity, expected_capacity)
            self.assertEqual(ranks, [list(range(ep_size))])
            return routed.new_zeros((capacity, hidden)), expected_recv_counts

        with unittest.mock.patch(
            "torchtitan.models.common.token_dispatcher.permute_a2av_dispatch_compact",
            fake_permute,
        ):
            routed, local_counts, metadata = dispatcher.dispatch(
                inputs, scores, ids, _counts_from_ids(ids, ep_size)
            )

        self.assertEqual(routed.shape, (expected_capacity, hidden))
        torch.testing.assert_close(
            local_counts, torch.tensor([expected_capacity], dtype=torch.int32)
        )
        self.assertEqual(metadata.compact_capacity, expected_capacity)
        torch.testing.assert_close(metadata.packed_source_indices_N, expected_vector)
        torch.testing.assert_close(metadata.send_counts_1E.reshape(-1), expected_counts)
        torch.testing.assert_close(metadata.recv_counts_1E, expected_recv_counts)
        torch.testing.assert_close(
            metadata.token_indices_experts_sorted_N, torch.arange(n) // top_k
        )
        torch.testing.assert_close(
            metadata.topk_scores_experts_sorted_N, scores.reshape(-1)
        )

    def test_stable_destination_order_with_skew_and_zero_peers(self):
        dispatcher = _make_dispatcher(8, 2, 8)
        for ids in (
            torch.tensor([[7, 0], [7, 1], [0, 1], [7, 0]]),
            torch.zeros((65, 2), dtype=torch.int64),
            torch.arange(8).repeat(17).reshape(-1, 2),
        ):
            n = ids.numel()
            counts, vector = dispatcher._build_packed_routing(
                ids, _counts_from_ids(ids, 8)
            )
            flat_ids = ids.reshape(-1)
            expected = torch.cat(
                [torch.where(flat_ids == peer)[0].to(torch.int32) for peer in range(8)]
            )
            torch.testing.assert_close(
                counts, _counts_from_ids(ids, 8).to(torch.int32).reshape(1, 8)
            )
            self.assertTrue(torch.equal(vector, expected))
            self.assertEqual(int(counts.sum()), n)
            self.assertTrue(
                torch.equal(vector.sort().values, torch.arange(n, dtype=torch.int32))
            )

    def test_distinct_topk_preserves_stable_order_and_peer_bounds(self):
        tokens = 17
        generator = torch.Generator().manual_seed(918)
        for ep, top_k in ((8, 1), (8, 2), (8, 8), (64, 8)):
            dispatcher = _make_dispatcher(ep, top_k, ep)
            patterns = {
                "skew": torch.arange(ep - top_k, ep).repeat(tokens, 1),
                "rotating": (
                    torch.arange(tokens)[:, None] + torch.arange(top_k)[None, :]
                )
                % ep,
                "random": torch.stack(
                    [
                        torch.randperm(ep, generator=generator)[:top_k]
                        for _ in range(tokens)
                    ]
                ),
            }
            for pattern, ids in patterns.items():
                with self.subTest(ep=ep, top_k=top_k, pattern=pattern):
                    n = ids.numel()
                    counts, vector = dispatcher._build_packed_routing(
                        ids, _counts_from_ids(ids, ep)
                    )
                    flat_ids = ids.reshape(-1).tolist()
                    expected = torch.tensor(
                        sorted(range(n), key=lambda row: flat_ids[row]),
                        dtype=torch.int32,
                    )
                    torch.testing.assert_close(vector, expected, rtol=0, atol=0)
                    self.assertEqual(int(counts.sum()), n)
                    self.assertGreaterEqual(int(counts.min()), 0)
                    self.assertLessEqual(int(counts.max()), tokens)
                    torch.testing.assert_close(
                        vector.sort().values,
                        torch.arange(n, dtype=torch.int32),
                        rtol=0,
                        atol=0,
                    )
                    expected_counts = torch.tensor(
                        [[flat_ids.count(peer) for peer in range(ep)]],
                        dtype=torch.int32,
                    )
                    torch.testing.assert_close(counts, expected_counts, rtol=0, atol=0)

    def test_compile_dispatch_combine_with_nonuniform_gate_scores(self):
        self._check_dispatch_combine_with_nonuniform_gate_scores(compiling=True)

    def test_eager_dispatch_combine_with_nonuniform_gate_scores(self):
        self._check_dispatch_combine_with_nonuniform_gate_scores(compiling=False)

    def _check_dispatch_combine_with_nonuniform_gate_scores(self, compiling):
        from torchtitan.distributed.neuron_ep import api

        ep, top_k, tokens, hidden = 8, 3, 5, 4
        dispatcher = _make_dispatcher(ep, top_k, ep)
        ids = torch.tensor([[7, 1, 4], [0, 7, 1], [4, 6, 0], [2, 5, 7], [6, 2, 1]])
        inputs = (
            torch.arange(tokens * hidden, dtype=torch.float32)
            .reshape(tokens, hidden)
            .requires_grad_()
        )
        scores = (
            torch.arange(1, tokens * top_k + 1, dtype=torch.float32).reshape(
                tokens, top_k
            )
            / 8
        ).requires_grad_()
        seen_vectors = []

        def expected_order():
            flat_ids = ids.reshape(-1).tolist()
            return torch.tensor(
                sorted(range(len(flat_ids)), key=lambda row: flat_ids[row])
            )

        def permute(x, packed_source_indices_N, counts, capacity, ranks):
            self.assertIsNotNone(packed_source_indices_N)
            seen_vectors.append(packed_source_indices_N)
            order = expected_order()
            torch.testing.assert_close(
                packed_source_indices_N.long(), order, rtol=0, atol=0
            )
            output = torch.cat([x[order], x.new_zeros(capacity - x.shape[0], hidden)])
            metadata = torch.zeros(4, ep, dtype=torch.int64)
            metadata[2] = counts[0] * hidden
            return output, metadata.to(torch.uint32)

        def unpermute(output, packed_source_indices_N, counts, recv_counts, ranks):
            order = expected_order()
            self.assertIsNotNone(packed_source_indices_N)
            torch.testing.assert_close(
                packed_source_indices_N.long(), order, rtol=0, atol=0
            )
            return output[: order.numel()][torch.argsort(order)]

        with (
            unittest.mock.patch("torch.compiler.is_compiling", return_value=compiling),
            unittest.mock.patch.object(api, "_packed_permute_a2av_kernel", permute),
            unittest.mock.patch.object(api, "_packed_unpermute_a2av_kernel", unpermute),
        ):
            packed, _, metadata = dispatcher.dispatch(
                inputs, scores, ids, _counts_from_ids(ids, ep)
            )
            self.assertIsNotNone(metadata.packed_source_indices_N)
            # Distinct expert scales make wrong row ordering observable.
            factors = (ids.reshape(-1) + 1).float()[
                metadata.packed_source_indices_N.long()
            ]
            factors = torch.cat(
                [factors, factors.new_zeros(packed.shape[0] - factors.numel())]
            )
            combined = dispatcher.combine(
                packed * factors[:, None],
                metadata,
                inputs,
                num_local_tokens_after_padding=tokens,
                local_seq_len_after_padding=tokens,
            )
            weights = (
                torch.arange(tokens * hidden, dtype=torch.float32).reshape(
                    tokens, hidden
                )
                + 3
            )
            (combined * weights).sum().backward()

        scale = (scores.detach() * (ids + 1)).sum(1)[:, None]
        torch.testing.assert_close(combined, inputs.detach() * scale, rtol=0, atol=0)
        torch.testing.assert_close(inputs.grad, weights * scale, rtol=0, atol=0)
        torch.testing.assert_close(
            scores.grad,
            (weights * inputs.detach()).sum(1)[:, None] * (ids + 1),
            rtol=0,
            atol=0,
        )
        self.assertEqual(len(seen_vectors), 2)
        self.assertTrue(
            all(vector is metadata.packed_source_indices_N for vector in seen_vectors)
        )


if __name__ == "__main__":
    unittest.main()
