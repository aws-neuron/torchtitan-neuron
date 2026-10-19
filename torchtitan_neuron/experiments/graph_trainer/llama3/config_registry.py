# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

"""Llama3 SimpleFSDP recipes for the Neuron JIT GraphTrainer path."""

from torchtitan.components.quantization import MXFP8LinearConverter
from torchtitan.config import CompileConfig
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.experiments.graph_trainer.llama3 import model_registry
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitan.models.llama3.config_registry import llama3_70b, llama3_8b
from torchtitan.protocols.model import ModelConfigConverter

from torchtitan_neuron.experiments.graph_trainer.configs import neuron_simple_fsdp


def _neuron_simple_fsdp_llama3_config(
    config_factory,
    *,
    full_ac: bool = False,
    converters: list[ModelConfigConverter.Config] | None = None,
) -> GraphTrainer.Config:
    base = config_factory()
    base.model_spec = model_registry(
        base.model_spec.flavor, attn_backend="sdpa", converters=converters
    )
    # Neuron does not support the RoPE gather; use the prefix-cache positions path.
    for layer in base.model_spec.model.layers:
        layer.attention.rope.use_prefix_cache_for_positions = True
    base.activation_checkpoint = FullAC.Config() if full_ac else SelectiveAC.Config()
    base.compile = CompileConfig(
        enable=True,
        components=["model", "loss", "grad_clip"],
        backend="neuron",
    )
    config = neuron_simple_fsdp(base, model_registry, model_name="simple_fsdp/llama3")
    return config


def simple_fsdp_llama3_8b() -> GraphTrainer.Config:
    return _neuron_simple_fsdp_llama3_config(llama3_8b)


def simple_fsdp_llama3_8b_fullac() -> GraphTrainer.Config:
    return _neuron_simple_fsdp_llama3_config(llama3_8b, full_ac=True)


def simple_fsdp_llama3_70b_fullac() -> GraphTrainer.Config:
    return _neuron_simple_fsdp_llama3_config(llama3_70b, full_ac=True)


def simple_fsdp_llama3_8b_mxfp8() -> GraphTrainer.Config:
    """Llama3-8B SimpleFSDP TP4 x FSDP16 with MXFP8 linears.

    MXFP8 counterpart of ``simple_fsdp_llama3_8b``, same ``8B`` flavor.
    ``fqns=["layers"]`` converts 160 linears (per layer: ``wqkv``, ``wo``,
    ``w1``/``w2``/``w3``); ``lm_head`` and embeddings stay bf16.
    """
    return _neuron_simple_fsdp_llama3_config(
        llama3_8b,
        converters=[
            MXFP8LinearConverter.Config(
                fqns=["layers"],
                kernel_preference="neuron",
                model_compile_enabled=True,
            )
        ],
    )
