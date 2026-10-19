# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

"""Tests for attention backend selection via ``get_attention_config`` and the
model registries' ``attn_backend`` argument."""

import logging
import unittest

from torchtitan.models.common.attention import (
    FlexAttention,
    ScaledDotProductAttention,
    VarlenAttention,
)
from torchtitan.models.common.config_utils import get_attention_config


class TestGetAttentionConfig(unittest.TestCase):
    def test_backends_map_to_config_classes(self):
        for backend, cls in (
            ("flex", FlexAttention.Config),
            ("varlen", VarlenAttention.Config),
            ("sdpa", ScaledDotProductAttention.Config),
        ):
            self.assertIsInstance(get_attention_config(backend), cls)

    def test_unknown_backend_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown backend"):
            get_attention_config("bogus")

    def test_nki_not_constructed_generically(self):
        # nki is gpt_oss-only; its registry builds the wrapper directly.
        with self.assertRaisesRegex(ValueError, "Unknown backend"):
            get_attention_config("nki")

    def test_sdpa_warns_about_packed_documents(self):
        with self.assertLogs(level=logging.WARNING) as captured:
            get_attention_config("sdpa")
        self.assertTrue(
            any("per-document masking" in msg for msg in captured.output),
            captured.output,
        )


class TestRegistryAttnBackend(unittest.TestCase):
    def test_llama3_registry_selects_backend(self):
        from torchtitan.models.llama3 import model_registry

        for backend, cls in (
            ("flex", FlexAttention.Config),
            ("varlen", VarlenAttention.Config),
            ("sdpa", ScaledDotProductAttention.Config),
        ):
            spec = model_registry("debugmodel", attn_backend=backend)
            for layer in spec.model.layers:
                self.assertIsInstance(layer.attention.inner_attention, cls)

    def test_gpt_oss_sdpa_rejects_sliding_window(self):
        from torchtitan.models.gpt_oss import model_registry

        has_window = any(
            layer.attention.sliding_window_size is not None
            for layer in model_registry("debugmodel").model.layers
        )
        if not has_window:
            self.skipTest("debugmodel has no sliding-window layers")
        with self.assertRaisesRegex(ValueError, "sliding-window"):
            model_registry("debugmodel", attn_backend="sdpa")

    def test_gpt_oss_varlen_window_from_sliding_window_size(self):
        from torchtitan.models.gpt_oss import model_registry

        spec = model_registry("debugmodel", attn_backend="varlen")
        for layer in spec.model.layers:
            sws = layer.attention.sliding_window_size
            if sws is not None:
                self.assertEqual(
                    layer.attention.inner_attention.window_size, (sws - 1, 0)
                )


if __name__ == "__main__":
    unittest.main()
