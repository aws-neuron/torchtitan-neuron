# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

"""Neuron-specific Qwen3 model implementations."""

from dataclasses import replace

from torchtitan.models.common import CosSinRoPE, Embedding, Linear
from torchtitan.models.qwen3 import (
    _build_qwen3_moe_layers,
    _EMBEDDING_INIT,
    _output_linear_init,
    _qwen3_norm,
    model_registry as core_model_registry,
    Qwen3Model,
)
from torchtitan.models.utils import validate_converter_order
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.model_spec import ModelSpec


__all__ = ["model_registry"]


def _235b_a22b_scaled(
    attn_backend: str,
    moe_comm_backend: str = "standard",
) -> Qwen3Model.Config:
    """Build the reduced-shape 235B validation flavor."""
    dim = 1024
    head_dim = 128
    vocab_size = 151936
    return Qwen3Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        norm=_qwen3_norm(dim),
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        layers=_build_qwen3_moe_layers(
            fuse_qkv=True,
            n_layers=2,
            dim=dim,
            n_heads=8,
            n_kv_heads=2,
            head_dim=head_dim,
            moe_hidden_dim=512,
            num_experts=16,
            top_k=4,
            attn_backend=attn_backend,
            rope=CosSinRoPE.Config(
                dim=head_dim,
                max_seq_len=40960,
                theta=5000000.0,
            ),
            moe_comm_backend=moe_comm_backend,
        ),
    )


def model_registry(
    flavor: str,
    attn_backend: str = "flex",
    moe_comm_backend: str | None = None,
    converters: list[ModelConfigConverter.Config] | None = None,
) -> ModelSpec:
    """Resolve core Qwen3 flavors plus Neuron-owned validation flavors."""
    if flavor != "235B-A22B-scaled":
        return core_model_registry(
            flavor,
            attn_backend=attn_backend,
            moe_comm_backend=moe_comm_backend,
            converters=converters,
        )

    config = _235b_a22b_scaled(
        attn_backend=attn_backend,
        moe_comm_backend=moe_comm_backend or "standard",
    )
    if converters is not None:
        validate_converter_order(converters)
        for converter in converters:
            config = converter.build().convert(config)

    base_spec = core_model_registry(
        "235B-A22B",
        attn_backend=attn_backend,
        moe_comm_backend=moe_comm_backend,
    )
    return replace(base_spec, flavor=flavor, model=config)
