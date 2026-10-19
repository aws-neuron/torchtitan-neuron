# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

from dataclasses import fields

from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.experiments.graph_trainer.graph_pp.pipeline import graph_pipeline_llm
from torchtitan.models.gpt_oss import gptoss_configs
from torchtitan.models.gpt_oss.state_dict_adapter import GptOssStateDictAdapter
from torchtitan.protocols.model_spec import ModelSpec

from .model import GraphTrainerGptOssModel
from .parallelize import parallelize_gptoss


def model_registry(
    flavor: str,
    moe_comm_backend: str = "standard",
    attn_backend: str = "varlen",
) -> ModelSpec:
    base = gptoss_configs[flavor](
        moe_comm_backend=moe_comm_backend,
        attn_backend=attn_backend,
    )
    config = GraphTrainerGptOssModel.Config(
        **{f.name: getattr(base, f.name) for f in fields(base)}
    )
    return ModelSpec(
        name="graph_trainer/gpt_oss",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_gptoss,
        pipelining_fn=graph_pipeline_llm,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=GptOssStateDictAdapter,
    )
