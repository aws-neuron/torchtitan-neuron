# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

"""Neuron Llama3 dense trainer configs."""

from torchtitan.components.metrics import MetricsProcessor
from torchtitan.config import CompileConfig
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.models.llama3 import model_registry
from torchtitan.models.llama3.config_registry import llama3_70b, llama3_8b
from torchtitan.trainer import Trainer


def _llama3_neuron_dense_config(
    flavor, base_factory, *, full_ac: bool, compile_config: CompileConfig | None = None
) -> Trainer.Config:
    config = base_factory()
    config.model_spec = model_registry(flavor, attn_backend="sdpa")
    # Neuron does not support the RoPE gather; use the prefix-cache positions path.
    # Applied here (alongside the sdpa attn backend) rather than via a dedicated
    # model flavor.
    for layer in config.model_spec.model.layers:
        layer.attention.rope.use_prefix_cache_for_positions = True
    config.metrics = MetricsProcessor.Config(log_freq=1)
    config.dataloader.dataset = "c4_test"
    config.activation_checkpoint = FullAC.Config() if full_ac else SelectiveAC.Config()
    if compile_config is not None:
        config.compile = compile_config
    return config


def llama3_8b_sac_eager() -> Trainer.Config:
    return _llama3_neuron_dense_config("8B", llama3_8b, full_ac=False)


def llama3_8b_fullac_eager() -> Trainer.Config:
    return _llama3_neuron_dense_config("8B", llama3_8b, full_ac=True)


def llama3_70b_fullac_eager() -> Trainer.Config:
    return _llama3_neuron_dense_config("70B", llama3_70b, full_ac=True)


def llama3_8b_sac_compile() -> Trainer.Config:
    return _llama3_neuron_dense_config(
        "8B",
        llama3_8b,
        full_ac=False,
        compile_config=CompileConfig(
            enable=True,
            components=["model", "loss", "grad_clip"],
            backend="neuron",
        ),
    )


def llama3_8b_fullac_compile() -> Trainer.Config:
    return _llama3_neuron_dense_config(
        "8B",
        llama3_8b,
        full_ac=True,
        compile_config=CompileConfig(
            enable=True,
            components=["model", "loss", "grad_clip"],
            backend="neuron",
        ),
    )


def llama3_70b_fullac_compile() -> Trainer.Config:
    return _llama3_neuron_dense_config(
        "70B",
        llama3_70b,
        full_ac=True,
        compile_config=CompileConfig(
            enable=True,
            components=["model", "loss", "grad_clip"],
            backend="neuron",
        ),
    )


def llama3_70b_16layers_fullac_compile() -> Trainer.Config:
    config = _llama3_neuron_dense_config(
        "70B",
        llama3_70b,
        full_ac=True,
        compile_config=CompileConfig(
            enable=True,
            components=["model", "loss", "grad_clip"],
            backend="neuron",
        ),
    )
    config.model_spec.model.layers = config.model_spec.model.layers[:16]
    return config
