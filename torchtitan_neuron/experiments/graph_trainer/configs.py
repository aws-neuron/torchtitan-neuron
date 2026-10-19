# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

"""Shared configuration helpers for Neuron GraphTrainer recipes."""

from collections.abc import Callable
from dataclasses import replace

from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    to_graph_trainer_config,
)
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.trainer import Trainer


def neuron_simple_fsdp(
    base_config: Trainer.Config,
    model_registry: Callable[[str], ModelSpec],
    *,
    model_name: str,
    model_spec: ModelSpec | None = None,
) -> GraphTrainer.Config:
    """Convert a Neuron Trainer recipe to a JIT SimpleFSDP recipe."""
    activation_checkpoint = base_config.activation_checkpoint
    compile_config = base_config.compile
    graph_config = to_graph_trainer_config(base_config, model_registry)
    graph_config.model_spec = (
        model_spec
        if model_spec is not None
        else replace(graph_config.model_spec, name=model_name)
    )
    graph_config.activation_checkpoint = activation_checkpoint
    graph_config.compile = GraphTrainerCompileConfig(
        enable=compile_config.enable,
        components=list(compile_config.components),
        backend=compile_config.backend,
        mode="jit",
    )
    return graph_config
