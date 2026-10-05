# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torch.nn.attention.flex_attention import and_masks
from torchtitan.models.common.attention import (
    BaseAttention,
    FlexInnerAttention,
    get_causal_mask_mod,
    get_document_mask_mod,
    get_flex_causal_mask_mods,
)
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.gpt_oss import build_model_config as build_gpt_oss_config
from torchtitan.models.gpt_oss.model import GptOssModel
from torchtitan.models.muse_glimmer import (
    build_model_config as build_muse_glimmer_config,
)
from torchtitan.models.muse_glimmer.model import MuseGlimmerModel

PACKED_POSITIONS = torch.tensor(
    [0, 1, 2, 0, 1, 0, 1, 2, 0, 1, 0, 1, 2, 3, 0, 1], dtype=torch.int32
)
SINGLE_DOC_POSITIONS = torch.arange(16, dtype=torch.int32)
BLOCK_SIZE = 4


def _dense(mask_mod, seq_len: int) -> torch.Tensor:
    b, h = torch.tensor(0), torch.tensor(0)
    return torch.tensor(
        [
            [
                bool(mask_mod(b, h, torch.tensor(q), torch.tensor(kv)))
                for kv in range(seq_len)
            ]
            for q in range(seq_len)
        ]
    )


def _decoder_masks(flex_kwargs: dict) -> dict:
    attn_config = BaseAttention.Config(
        n_heads=1,
        inner_attention=FlexInnerAttention.Config(block_size=BLOCK_SIZE, **flex_kwargs),
    )
    decoder = Decoder.__new__(Decoder)
    return {
        "mask": decoder._create_flex_attention_mask_for_document(
            PACKED_POSITIONS, attn_config
        )
    }


def _model_masks(build_config, model_cls):
    def masks(flex_kwargs: dict) -> dict:
        config = build_config("debugmodel", seq_len=16, attn_backend="flex")
        for layer in config.layers:
            inner = layer.attention.inner_attention
            assert isinstance(inner, FlexInnerAttention.Config)
            inner.block_size = BLOCK_SIZE
            for key, value in flex_kwargs.items():
                setattr(inner, key, value)
        model = model_cls.__new__(model_cls)
        model.config = config
        return model.get_attention_masks(PACKED_POSITIONS)

    return masks


class TestFlexMaskType(unittest.TestCase):
    def test_mask_mods_match_mask_type(self):
        for positions in (PACKED_POSITIONS, SINGLE_DOC_POSITIONS):
            seq_len = positions.shape[0]
            causal = _dense(get_causal_mask_mod(), seq_len)
            same_doc = _dense(get_document_mask_mod(positions), seq_len)
            for mask_type, expected in (
                ("block_causal", causal & same_doc),
                ("causal", causal),
            ):
                with self.subTest(
                    single_doc=positions is SINGLE_DOC_POSITIONS, mask_type=mask_type
                ):
                    mask_mods = get_flex_causal_mask_mods(
                        positions, FlexInnerAttention.Config(mask_type=mask_type)
                    )
                    torch.testing.assert_close(
                        _dense(and_masks(*mask_mods), seq_len), expected
                    )

    def test_mask_builders_follow_mask_type(self):
        num_blocks = PACKED_POSITIONS.shape[0] // BLOCK_SIZE
        causal_blocks = torch.ones(num_blocks, num_blocks, dtype=torch.bool).tril()
        builders = {
            "decoder": _decoder_masks,
            "gpt_oss": _model_masks(build_gpt_oss_config, GptOssModel),
            "muse_glimmer": _model_masks(build_muse_glimmer_config, MuseGlimmerModel),
        }
        for name, build_masks in builders.items():
            default_masks = build_masks({})
            causal_masks = build_masks({"mask_type": "causal"})
            for key, causal_mask in causal_masks.items():
                with self.subTest(builder=name, mask=key):
                    torch.testing.assert_close(
                        causal_mask.to_dense()[0, 0].bool(), causal_blocks
                    )
                    # Default (block_causal) drops blocks in earlier documents.
                    self.assertLess(
                        default_masks[key].to_dense().sum().item(),
                        causal_mask.to_dense().sum().item(),
                    )


if __name__ == "__main__":
    unittest.main()
