# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

"""Neuron FlexAttention kernel-lowering tests.

Exercise static causal and sliding-window masks and dynamic document masks
directly, without modifying the framework's document-mask builders.
"""

import logging
import unittest

import pytest
import torch

pytest.importorskip("torch_neuronx")

from torchtitan.tools.utils import device_type  # noqa: E402


_NKI_LOGGER = (
    "torch_neuronx.neuron_dynamo_backend.fx.passes.flex_attention_nki_lowering"
)
_FALLBACK_MARKERS = (
    "falling back to graph-based",
    "Failed to lower flex_attention",
    "not statically computable",
)


class _FallbackCatcher(logging.Handler):
    """Record whether the NKI-lowering pass fell back for flex."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.name.startswith(_NKI_LOGGER) and any(
            m in record.getMessage() for m in _FALLBACK_MARKERS
        ):
            self.records.append(record.getMessage())


@unittest.skipUnless(device_type == "neuron", "requires a Neuron device")
class TestFlexNKILoweringOnNeuron(unittest.TestCase):
    """Check NKI lowering for static masks and fallback for dynamic masks."""

    B, H, S, D = 1, 4, 512, 64

    def setUp(self):
        # Isolate torch.compile / dynamo state per test: the NKI-lowering pass
        # caches per compiled graph, so a fallback logged while compiling one
        # mask must not leak into a later test that compiles a different mask.
        torch._dynamo.reset()

    def tearDown(self):
        torch._dynamo.reset()

    def _run_flex(self, mask_mod, seq_len: int | None = None) -> bool:
        """Compile + run fwd/bwd for one mask_mod; True if it lowered to NKI."""
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention

        s = self.S if seq_len is None else seq_len
        catcher = _FallbackCatcher()
        root = logging.getLogger()
        root.addHandler(catcher)
        try:
            compiled_cbm = torch.compile(create_block_mask, backend="neuron")
            compiled_flex = torch.compile(flex_attention, backend="neuron")

            torch.manual_seed(0)
            qkv = [
                torch.randn(
                    self.B,
                    self.H,
                    s,
                    self.D,
                    device="neuron",
                    dtype=torch.bfloat16,
                    requires_grad=True,
                )
                for _ in range(3)
            ]
            block_mask = compiled_cbm(mask_mod, self.B, self.H, s, s, device="neuron")
            out = compiled_flex(*qkv, block_mask=block_mask)
            out.sum().backward()
        finally:
            root.removeHandler(catcher)
        return len(catcher.records) == 0

    def test_static_causal_mask_lowers_to_nki(self):
        def causal(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        self.assertTrue(
            self._run_flex(causal),
            "static causal mask_mod should lower to the NKI kernel (no fallback)",
        )

    def test_static_sliding_window_mask_lowers_to_nki(self):
        from torch.nn.attention.flex_attention import and_masks

        from torchtitan.models.common.attention import (
            get_causal_mask_mod,
            get_sliding_window_mask_mod,
        )

        seq_len = 1024
        composed = and_masks(
            get_causal_mask_mod(),
            get_sliding_window_mask_mod(128),
        )

        self.assertTrue(
            self._run_flex(composed, seq_len=seq_len),
            "static causal sliding-window mask should lower to the NKI kernel "
            "(no fallback)",
        )

    def test_dynamic_document_mask_falls_back(self):
        # A mask_mod that indexes a captured tensor is dynamic -> NKI cannot
        # statically evaluate it.
        doc_ids = torch.zeros(self.S, dtype=torch.int32, device="neuron")
        doc_ids[self.S // 2 :] = 1

        def document(b, h, q_idx, kv_idx):
            return (q_idx >= kv_idx) & (doc_ids[q_idx] == doc_ids[kv_idx])

        self.assertFalse(
            self._run_flex(document),
            "dynamic (captured-tensor) mask_mod is expected to fall back from NKI",
        )


if __name__ == "__main__":
    unittest.main()
