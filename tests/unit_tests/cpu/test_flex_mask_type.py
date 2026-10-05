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

PACKED_POSITIONS = torch.tensor(
    [0, 1, 2, 0, 1, 0, 1, 2, 0, 1, 0, 1, 2, 3, 0, 1], dtype=torch.int32
)
SINGLE_DOC_POSITIONS = torch.arange(16, dtype=torch.int32)


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


def _decoder_block_mask(positions: torch.Tensor, mask_type: str) -> torch.Tensor:
    attn_config = BaseAttention.Config(
        n_heads=1,
        inner_attention=FlexInnerAttention.Config(block_size=4, mask_type=mask_type),
    )
    decoder = Decoder.__new__(Decoder)
    mask = decoder._create_flex_attention_mask_for_document(positions, attn_config)
    return mask.to_dense()


class TestFlexMaskType(unittest.TestCase):
    def test_default_is_block_causal(self):
        self.assertEqual(FlexInnerAttention.Config().mask_type, "block_causal")

    def test_mask_mods_match_mask_type(self):
        seq_len = PACKED_POSITIONS.shape[0]
        causal = _dense(get_causal_mask_mod(), seq_len)
        same_doc = _dense(get_document_mask_mod(PACKED_POSITIONS), seq_len)

        block_causal_mods = get_flex_causal_mask_mods(
            PACKED_POSITIONS, FlexInnerAttention.Config(mask_type="block_causal")
        )
        causal_mods = get_flex_causal_mask_mods(
            PACKED_POSITIONS, FlexInnerAttention.Config(mask_type="causal")
        )

        self.assertEqual(len(causal_mods), 1)
        torch.testing.assert_close(
            _dense(and_masks(*block_causal_mods), seq_len), causal & same_doc
        )
        torch.testing.assert_close(_dense(and_masks(*causal_mods), seq_len), causal)

    def test_decoder_mask_follows_mask_type(self):
        block_causal = _decoder_block_mask(PACKED_POSITIONS, "block_causal")
        causal = _decoder_block_mask(PACKED_POSITIONS, "causal")
        num_blocks = causal.shape[-1]

        torch.testing.assert_close(
            causal[0, 0].bool(),
            torch.ones(num_blocks, num_blocks, dtype=torch.bool).tril(),
        )
        # Packed documents mask out blocks that lie entirely in earlier documents.
        self.assertLess(block_causal.sum().item(), causal.sum().item())

    def test_mask_types_agree_for_single_document(self):
        torch.testing.assert_close(
            _decoder_block_mask(SINGLE_DOC_POSITIONS, "block_causal"),
            _decoder_block_mask(SINGLE_DOC_POSITIONS, "causal"),
        )


if __name__ == "__main__":
    unittest.main()
