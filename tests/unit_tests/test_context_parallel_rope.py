# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

import tempfile
import unittest
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard

from torchtitan.distributed.context_parallel import (
    apply_cp_to_rope,
    prepare_context_parallel_input,
)
from torchtitan.models.common.rope import ComplexRoPE, CosSinRoPE


def _cp_mesh(rank, size):
    mesh = Mock(spec=DeviceMesh)
    mesh.ndim = 1
    mesh.mesh_dim_names = ("cp",)
    mesh.size.return_value = size
    mesh.get_local_rank.return_value = rank
    return mesh


class TestContextParallelRoPE(unittest.TestCase):
    def test_input_positions_are_resolved_before_sharding(self):
        inputs = torch.arange(16).view(2, 8)
        labels = inputs + 1
        packed_positions = torch.tensor([[0, 1, 2, 0, 1, 2, 3, 0]]).expand(2, -1)

        def shard_second_half(mesh, buffers, masks, load_balancer):
            return tuple(buffer[:, 4:] for buffer in buffers), masks

        for kwargs in ({}, {"positions": None}, {"positions": packed_positions}):
            with self.subTest(positions=kwargs.get("positions")):
                expected = kwargs.get("positions")
                if expected is None:
                    expected = torch.arange(8).expand(2, -1)
                with patch(
                    "torchtitan.distributed.context_parallel.cp_shard",
                    side_effect=shard_second_half,
                ):
                    (
                        local_inputs,
                        local_labels,
                        local_kwargs,
                    ) = prepare_context_parallel_input(
                        inputs,
                        labels,
                        kwargs,
                        _cp_mesh(1, 2),
                        torch.device("cpu"),
                        None,
                    )
                torch.testing.assert_close(local_inputs, inputs[:, 4:])
                torch.testing.assert_close(local_labels, labels[:, 4:])
                torch.testing.assert_close(local_kwargs["positions"], expected[:, 4:])

    def test_shards_match_global_rotation_and_gradients(self):
        """Every rank must rotate at global positions, including after reinit."""
        torch.manual_seed(42)
        for rope_cls in (
            ComplexRoPE,
            CosSinRoPE,
        ):
            for cp_size in (1, 2, 16):
                for load_balancer in (None, "headtail"):
                    with self.subTest(
                        rope=rope_cls.__name__, cp=cp_size, balancer=load_balancer
                    ):
                        seq_len = 64
                        config = rope_cls.Config(
                            dim=8,
                            max_seq_len=128,
                            use_prefix_cache_for_positions=True,
                        )
                        reference = rope_cls(config)
                        query = torch.randn(2, seq_len, 4, 8, requires_grad=True)
                        key = torch.randn(2, seq_len, 2, 8, requires_grad=True)
                        query_ref, key_ref = reference(query, key)
                        (query_ref.sum() + key_ref.sum()).backward()
                        for rank in range(cp_size):
                            indices = torch.arange(seq_len)
                            if cp_size > 1 and load_balancer == "headtail":
                                chunks = indices.chunk(2 * cp_size)
                                indices = torch.cat(
                                    (chunks[rank], chunks[2 * cp_size - rank - 1])
                                )
                            else:
                                indices = indices.chunk(cp_size)[rank]
                            rope = rope_cls(config)
                            apply_cp_to_rope(
                                torch.nn.Sequential(rope),
                                _cp_mesh(rank, cp_size),
                                load_balancer_type=load_balancer,
                            )
                            rope.to_empty(device="cpu")
                            rope.init_states()
                            query_local = query.detach()[:, indices].requires_grad_()
                            key_local = key.detach()[:, indices].requires_grad_()
                            # Prefix-cache semantics intentionally ignore document
                            # resets, just as the CP1 baseline does.
                            positions = torch.zeros(2, len(indices), dtype=torch.long)
                            query_out, key_out = rope(query_local, key_local, positions)
                            torch.testing.assert_close(
                                query_out, query_ref[:, indices], rtol=0, atol=0
                            )
                            torch.testing.assert_close(
                                key_out, key_ref[:, indices], rtol=0, atol=0
                            )
                            (query_out.sum() + key_out.sum()).backward()
                            torch.testing.assert_close(
                                query_local.grad, query.grad[:, indices], rtol=0, atol=0
                            )
                            torch.testing.assert_close(
                                key_local.grad, key.grad[:, indices], rtol=0, atol=0
                            )

    def test_position_gather_is_unchanged(self):
        rope = CosSinRoPE(CosSinRoPE.Config(dim=8, max_seq_len=64))
        query = torch.randn(2, 4, 2, 8)
        positions = torch.tensor([[10, 11, 0, 1], [2, 3, 4, 0]])
        expected = rope(query, query, positions)
        apply_cp_to_rope(rope, _cp_mesh(1, 2), load_balancer_type="ptrr")
        actual = rope(query, query, positions)
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, rtol=0, atol=0)

    def test_prefix_cache_rejects_ptrr_during_setup(self):
        for rope_cls, prefix_config in (
            (ComplexRoPE, True),
            (CosSinRoPE, True),
        ):
            with self.subTest(rope=rope_cls.__name__, prefix_config=prefix_config):
                rope = rope_cls(
                    rope_cls.Config(
                        dim=8,
                        max_seq_len=64,
                        use_prefix_cache_for_positions=prefix_config,
                    )
                )
                with self.assertRaisesRegex(ValueError, "Prefix RoPE caches require"):
                    apply_cp_to_rope(rope, _cp_mesh(1, 2), load_balancer_type="ptrr")

    def test_mesh_axis_is_validated(self):
        rope = ComplexRoPE(ComplexRoPE.Config(dim=8, max_seq_len=64))
        mesh = _cp_mesh(0, 2)
        mesh.mesh_dim_names = ("tp",)
        with self.assertRaisesRegex(ValueError, "mesh named 'cp'"):
            apply_cp_to_rope(rope, mesh, load_balancer_type=None)
        mesh.ndim = 2
        mesh.mesh_dim_names = ("dp", "cp")
        with self.assertRaisesRegex(ValueError, "1D mesh"):
            apply_cp_to_rope(rope, mesh, load_balancer_type=None)


class TestContextParallelRoPEDTensor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._temp_dir = tempfile.TemporaryDirectory()
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{cls._temp_dir.name}/store",
            world_size=1,
            rank=0,
        )

    @classmethod
    def tearDownClass(cls):
        dist.destroy_process_group()
        cls._temp_dir.cleanup()

    def test_tp_and_multi_axis_cache_placements(self):
        """Cache broadcasts over batch/heads and follows CP sequence sharding."""
        for rope_cls in (ComplexRoPE, CosSinRoPE):
            for axis_names, placements in (
                (("tp",), (Shard(2),)),
                (("dp", "cp", "tp"), (Shard(0), Shard(1), Shard(2))),
            ):
                with self.subTest(rope=rope_cls.__name__, axes=axis_names):
                    mesh = init_device_mesh(
                        "cpu", (1,) * len(axis_names), mesh_dim_names=axis_names
                    )
                    rope = rope_cls(
                        rope_cls.Config(
                            dim=8, max_seq_len=64, use_prefix_cache_for_positions=True
                        )
                    )
                    expected = rope.cache[4:8].view(1, 4, 1, -1).clone()
                    rope.cache = DTensor.from_local(
                        rope.cache, mesh, (Replicate(),) * mesh.ndim, run_check=False
                    )
                    apply_cp_to_rope(rope, _cp_mesh(1, 2), load_balancer_type=None)
                    query = DTensor.from_local(
                        torch.randn(2, 4, 2, 8), mesh, placements, run_check=False
                    )
                    cache = rope._reshape_cache(query)
                    torch.testing.assert_close(
                        cache.to_local(), expected, rtol=0, atol=0
                    )
                    self.assertEqual(
                        cache.placements,
                        tuple(
                            Shard(1) if axis == "cp" else Replicate()
                            for axis in axis_names
                        ),
                    )


if __name__ == "__main__":
    unittest.main()
