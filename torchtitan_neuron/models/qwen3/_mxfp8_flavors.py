# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

"""Neuron MXFP8 Qwen3 model flavors for prequantization validation.

These alignment-specific flavors are kept out of the core Qwen3 registry. The
recipes that use them call ``register_mxfp8_flavors()`` to register them into
core's ``qwen3_configs`` on demand, so merely importing the Neuron package does
not mutate core's registry. See the per-flavor docstrings for the alignment
arithmetic.
"""

from functools import partial

from torchtitan.models.common import CosSinRoPE, Embedding, Linear
from torchtitan.models.qwen3 import (
    _build_qwen3_layers,
    _EMBEDDING_INIT,
    _output_linear_init,
    _qwen3_norm,
    qwen3_configs,
    Qwen3Model,
)


def _32b_dim6144_ffn24576(attn_backend: str, n_layers: int = 6) -> Qwen3Model.Config:
    """Qwen3-32B reshaped to dim=6144 / hidden_dim=24576 (5.26B params at 6 layers).

    Every per-rank weight shard is a multiple of 128 in both dims under TP=4 x
    FSDP=16 with FSDP on dim 1, meeting the MXFP8 quantize kernel's 128-element
    floor on the blocked dim; requires SimpleFSDP dim-1 sharding support. Alignment
    is per-weight, so n_layers does not affect it (6 layers keeps seed-checkpoint
    numerics runs cheap). Not a real Qwen3-32B shape and not a convergence target;
    lm_head and tok_embeddings keep the stock vocab_size, outside the alignment
    claim (the MXFP8 converters scope to fqns=["layers"]).
    """
    dim = 6144
    head_dim = 128
    vocab_size = 151936
    return Qwen3Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        norm=_qwen3_norm(dim),
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        layers=_build_qwen3_layers(
            fuse_qkv=True,
            n_layers=n_layers,
            dim=dim,
            n_heads=64,
            n_kv_heads=8,
            head_dim=head_dim,
            hidden_dim=24576,
            attn_backend=attn_backend,
            rope=CosSinRoPE.Config(
                dim=head_dim,
                max_seq_len=40960,
                theta=1000000.0,
            ),
        ),
    )


def _32b_dim8192_ffn32768(attn_backend: str, n_layers: int = 6) -> Qwen3Model.Config:
    """Qwen3-32B reshaped so every per-rank K is 512 (10.94B params at 6 layers).

    The wrapX packing needs each rank's local K to be a whole multiple of the quantize
    kernel's 512-element tile, otherwise the per-rank encodings do not concatenate into
    the layout whole-K quantization produces and the matmul silently contracts mismatched
    K (see ``MIN_K`` in ``mx_prequant.py``). These shapes satisfy that for all weights
    under TP=4 x FSDP=16 on dim 1; requires SimpleFSDP dim-1 sharding support. Not a real
    Qwen3-32B shape and not a convergence target; lm_head and tok_embeddings keep the stock
    vocab_size, outside the alignment claim (the MXFP8 converters scope to fqns=["layers"]).
    """
    dim = 8192
    head_dim = 128
    vocab_size = 151936
    return Qwen3Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        norm=_qwen3_norm(dim),
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        layers=_build_qwen3_layers(
            fuse_qkv=True,
            n_layers=n_layers,
            dim=dim,
            n_heads=256,
            n_kv_heads=32,
            head_dim=head_dim,
            hidden_dim=32768,
            attn_backend=attn_backend,
            rope=CosSinRoPE.Config(
                dim=head_dim,
                max_seq_len=40960,
                theta=1000000.0,
            ),
        ),
    )


# Alignment-validation flavors, keyed as model_registry(flavor) expects. Registered
# on demand by the recipes that use them (register_mxfp8_flavors), not at import.
_MXFP8_FLAVORS = {
    "32B_dim6144_ffn24576_6layers": partial(_32b_dim6144_ffn24576, n_layers=6),
    "32B_dim8192_ffn32768_6layers": partial(_32b_dim8192_ffn32768, n_layers=6),
}


def register_mxfp8_flavors() -> None:
    """Register the MXFP8 validation flavors in core's qwen3_configs."""
    for name, builder in _MXFP8_FLAVORS.items():
        qwen3_configs.setdefault(name, builder)
