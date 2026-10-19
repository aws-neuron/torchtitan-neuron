# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

"""Qwen3 Neuron GraphTrainer configs."""

from dataclasses import fields

from torchtitan.experiments.graph_trainer.graph_pp.pipeline import graph_pipeline_llm
from torchtitan.experiments.graph_trainer.qwen3.model import GraphTrainerQwen3Model
from torchtitan.experiments.graph_trainer.qwen3.parallelize import parallelize_qwen3
from torchtitan.models.qwen3.state_dict_adapter import Qwen3StateDictAdapter
from torchtitan.protocols.model_spec import ModelSpec

from torchtitan_neuron.models.qwen3 import model_registry as qwen3_model_registry


def model_registry(
    flavor: str,
    attn_backend: str = "flex",
    moe_comm_backend: str | None = None,
) -> ModelSpec:
    """Build GraphTrainer specs for core and Neuron-owned Qwen3 flavors."""
    base = qwen3_model_registry(
        flavor,
        attn_backend=attn_backend,
        moe_comm_backend=moe_comm_backend,
    ).model
    config = GraphTrainerQwen3Model.Config(
        **{field.name: getattr(base, field.name) for field in fields(base)}
    )
    return ModelSpec(
        name="graph_trainer/qwen3",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_qwen3,
        pipelining_fn=graph_pipeline_llm,
        post_optimizer_build_fn=None,
        state_dict_adapter=Qwen3StateDictAdapter,
    )
