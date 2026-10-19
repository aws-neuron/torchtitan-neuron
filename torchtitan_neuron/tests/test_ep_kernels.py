# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

"""Unit tests for the Neuron EP kernel wrappers and their tensor contracts."""

import unittest
import unittest.mock

import torch

from torchtitan.distributed.neuron_ep import (
    permute_a2av_dispatch_compact,
    unpermute_a2av_combine_packed,
)


class TestPackedA2AVAutograd(unittest.TestCase):
    @staticmethod
    def _permute(hidden, indices, counts, capacity, _ranks):
        output = hidden.new_zeros(capacity, hidden.shape[1])
        output[: hidden.shape[0]] = hidden[indices.long()]
        metadata = torch.zeros(4, counts.shape[1], dtype=torch.uint32)
        metadata[2] = counts[0].to(torch.uint32) * hidden.shape[1]
        return output, metadata

    @staticmethod
    def _unpermute(
        output,
        indices,
        _send_counts,
        _recv_counts,
        _ranks,
    ):
        return output[: indices.shape[0]][torch.argsort(indices.long())]

    def test_kernel_io_rows_use_routed_and_expert_bounds(self):
        local_tokens = 6
        top_k = 4
        # Match the dispatcher's supported one-expert-per-EP-rank capacity.
        ep_size = 8
        hidden = 3
        routed_rows = local_tokens * top_k
        expert_rows = local_tokens * ep_size
        indices = torch.arange(routed_rows, dtype=torch.int32)
        counts = torch.full((1, ep_size), routed_rows // ep_size, dtype=torch.int32)
        api_globals = permute_a2av_dispatch_compact.__globals__

        with unittest.mock.patch.dict(
            api_globals,
            {
                "_packed_permute_a2av_kernel": self._permute,
                "_packed_unpermute_a2av_kernel": self._unpermute,
            },
        ):
            routed = torch.randn(routed_rows, hidden, requires_grad=True)
            packed, recv_counts = permute_a2av_dispatch_compact(
                routed, indices, counts, expert_rows, [list(range(ep_size))]
            )
            self.assertEqual(packed.shape, (expert_rows, hidden))
            packed.sum().backward()
            self.assertEqual(routed.grad.shape, (routed_rows, hidden))

            expert_output = torch.randn(expert_rows, hidden, requires_grad=True)
            combined = unpermute_a2av_combine_packed(
                expert_output,
                indices,
                counts,
                recv_counts,
                [list(range(ep_size))],
            )
            self.assertEqual(combined.shape, (routed_rows, hidden))
            combined.sum().backward()
            self.assertEqual(expert_output.grad.shape, (expert_rows, hidden))

    def test_packed_dispatch_and_combine_are_adjoints(self):
        n, hidden, capacity, ep_size = 8, 3, 16, 8
        indices = torch.arange(n, dtype=torch.int32)
        counts = torch.ones(1, ep_size, dtype=torch.int32)
        api_globals = permute_a2av_dispatch_compact.__globals__
        with unittest.mock.patch.dict(
            api_globals,
            {
                "_packed_permute_a2av_kernel": self._permute,
                "_packed_unpermute_a2av_kernel": self._unpermute,
            },
        ):
            routed = torch.randn(n, hidden, requires_grad=True)
            packed, recv_counts = permute_a2av_dispatch_compact(
                routed, indices, counts, capacity, [list(range(ep_size))]
            )
            self.assertEqual(packed.shape, (capacity, hidden))
            packed.sum().backward()
            torch.testing.assert_close(routed.grad, torch.ones_like(routed))

            expert_output = torch.randn(capacity, hidden, requires_grad=True)
            combined = unpermute_a2av_combine_packed(
                expert_output,
                indices,
                counts,
                recv_counts,
                [list(range(ep_size))],
            )
            combined.sum().backward()
            expected = torch.zeros_like(expert_output)
            expected[:n] = 1
            torch.testing.assert_close(expert_output.grad, expected)

    def test_recv_counts_divide_before_int32_narrowing(self):
        hidden = 4096
        indices = torch.zeros(1, dtype=torch.int32)
        counts = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.int32)

        def permute_with_large_metadata(input_tensor, *_args):
            metadata = torch.zeros(4, 8, dtype=torch.uint32)
            metadata[2, 0] = 2**31
            return input_tensor, metadata

        api_globals = permute_a2av_dispatch_compact.__globals__
        with unittest.mock.patch.dict(
            api_globals,
            {"_packed_permute_a2av_kernel": permute_with_large_metadata},
        ):
            _packed, recv_counts = permute_a2av_dispatch_compact(
                torch.zeros(1, hidden),
                indices,
                counts,
                1,
                [list(range(8))],
            )
        self.assertEqual(int(recv_counts[0, 0]), 2**31 // hidden)


class TestPackedA2AVLNCContract(unittest.TestCase):
    def test_lnc2_grid(self):
        from torchtitan.distributed.neuron_ep.api import _packed_lnc2_grid

        self.assertEqual(_packed_lnc2_grid(2), (2,))

    def test_non_lnc2_fails_before_kernel_launch(self):
        from torchtitan.distributed.neuron_ep.api import _packed_lnc2_grid

        with self.assertRaisesRegex(RuntimeError, "NEURON_LOGICAL_NC_CONFIG=2"):
            _packed_lnc2_grid(1)

    def test_replica_groups_require_complete_aligned_devices(self):
        from torchtitan.distributed.neuron_ep.api import _validate_packed_ep_ranks

        _validate_packed_ep_ranks([list(range(8)), list(range(8, 16))])
        for invalid in (
            [list(range(4))],
            [list(range(10))],
            [list(range(2, 10))],
            [[0, 1, 2, 3, 5, 6, 7, 8]],
        ):
            with (
                self.subTest(invalid=invalid),
                self.assertRaisesRegex(ValueError, "packed A2AV"),
            ):
                _validate_packed_ep_ranks(invalid)


class TestPackedSourceIndexAutograd(unittest.TestCase):
    def test_nonidentity_permutation_with_nonuniform_gradients(self):
        n, h, c, ep = 7, 3, 13, 8
        vector = torch.tensor([5, 1, 6, 4, 0, 3, 2], dtype=torch.int32)
        counts = torch.tensor([[2, 0, 1, 0, 3, 0, 0, 1]], dtype=torch.int32)
        seen = []

        def permute(x, packed_source_indices_N, cnt, cap, ranks):
            seen.append(packed_source_indices_N)
            torch.testing.assert_close(packed_source_indices_N, vector, rtol=0, atol=0)
            output = torch.cat(
                [x[packed_source_indices_N.long()], x.new_zeros(cap - n, h)]
            )
            metadata = torch.zeros(4, ep, dtype=torch.int64)
            metadata[2] = cnt[0] * h
            return output, metadata.to(torch.uint32)

        def unpermute(output, packed_source_indices_N, cnt, recv_counts, ranks):
            self.assertIs(packed_source_indices_N, vector)
            order = vector.long()
            return output[:n][torch.argsort(order)]

        with unittest.mock.patch.dict(
            permute_a2av_dispatch_compact.__globals__,
            {
                "_packed_permute_a2av_kernel": permute,
                "_packed_unpermute_a2av_kernel": unpermute,
            },
        ):
            ranks = [list(range(ep))]
            routed = (
                torch.arange(n * h, dtype=torch.float32).reshape(n, h).requires_grad_()
            )
            packed, recv_counts = permute_a2av_dispatch_compact(
                routed, vector, counts, c, ranks
            )
            packed_grad = torch.arange(c * h, dtype=torch.float32).reshape(c, h) + 2
            packed.backward(packed_grad)
            torch.testing.assert_close(
                routed.grad,
                packed_grad[:n][torch.argsort(vector.long())],
                rtol=0,
                atol=0,
            )
            expert = (
                torch.arange(c * h, dtype=torch.float32).reshape(c, h).requires_grad_()
            )
            combined = unpermute_a2av_combine_packed(
                expert, vector, counts, recv_counts, ranks
            )
            combined_grad = torch.arange(n * h, dtype=torch.float32).reshape(n, h) + 11
            combined.backward(combined_grad)
        self.assertEqual(len(seen), 2)
        self.assertTrue(all(x is vector for x in seen))
        expected = torch.cat([combined_grad[vector.long()], torch.zeros(c - n, h)])
        torch.testing.assert_close(expert.grad, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
