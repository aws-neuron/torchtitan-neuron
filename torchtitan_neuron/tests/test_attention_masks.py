# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

"""Unit tests for GPT-OSS attention mask construction (get_attention_masks).

GPT-OSS alternates full-attention and sliding-window-attention layers. Its
``GptOssModel.get_attention_masks`` builds a dict of block masks on the flex path:
always a ``basic_mask``, plus a ``sliding_window_mask`` when any layer requests a
window. This covers that branching logic and the composed mask_mods.

``create_attention_mask`` (which compiles ``create_block_mask``) is patched out so
the tests exercise GPT-OSS's own logic -- the mask_mod composition and per-mask
selection -- without depending on block-mask compilation (device/backend-specific).
The composed mask_mod is captured and evaluated directly to verify the window
restriction. ``get_attention_masks`` reads only ``self.config``, so we drive it with
the debug config rather than instantiating the full model.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import pytest

import torch

# Importing the gpt_oss model pulls torchtitan.distributed, which needs the Neuron
# stack. Gate so collection works on a host without it.
pytest.importorskip("spmd_types")

from torchtitan.models.gpt_oss import model_registry  # noqa: E402
from torchtitan.models.gpt_oss.model import GptOssModel  # noqa: E402


def _flex_debug_config():
    # attn_backend="flex" so inner_attention is FlexAttention.Config (the debug
    # model puts a sliding window on even-indexed layers).
    return model_registry("debugmodel", attn_backend="flex").model


class TestGptOssAttentionMasks(unittest.TestCase):
    S = 256

    def _build_masks(self, cfg):
        """Run get_attention_masks with create_attention_mask stubbed to record the
        composed mask_mod per named mask (returns the mask_mod itself as the mask)."""
        positions = torch.arange(self.S).unsqueeze(0)  # [B=1, S]

        def _capture(mask_mod, B, H, q_len, kv_len, **kwargs):
            return mask_mod  # stand in for the BlockMask; keep the mask_mod itself

        with mock.patch(
            "torchtitan.models.gpt_oss.model.create_attention_mask", _capture
        ):
            return GptOssModel.get_attention_masks(
                SimpleNamespace(config=cfg), positions
            )

    def test_flex_builds_basic_and_sliding_window_masks(self):
        masks = self._build_masks(_flex_debug_config())
        self.assertIsInstance(masks, dict)
        self.assertEqual(set(masks), {"basic_mask", "sliding_window_mask"})

    def test_only_basic_mask_when_no_layer_uses_sliding_window(self):
        cfg = _flex_debug_config()
        for layer in cfg.layers:
            layer.attention.sliding_window_size = None
        masks = self._build_masks(cfg)
        self.assertEqual(set(masks), {"basic_mask"})

    def test_sliding_window_mask_restricts_range(self):
        cfg = _flex_debug_config()
        # The debug model's windowed layers use sliding_window_size=128.
        window = next(
            layer.attention.sliding_window_size
            for layer in cfg.layers
            if layer.attention.sliding_window_size is not None
        )
        composed_mask_mod = self._build_masks(cfg)["sliding_window_mask"]

        def allowed(q_idx: int, kv_idx: int) -> bool:
            zero = torch.tensor(0)
            return bool(
                composed_mask_mod(zero, zero, torch.tensor(q_idx), torch.tensor(kv_idx))
            )

        # Causal + within-window (q - kv < window) -> attention allowed.
        self.assertTrue(allowed(200, 200 - (window - 1)))
        # Causal but beyond the window (q - kv >= window) -> masked out.
        self.assertFalse(allowed(200, 200 - window))
        # Future token (kv > q) -> masked by causality.
        self.assertFalse(allowed(200, 201))


if __name__ == "__main__":
    unittest.main()
