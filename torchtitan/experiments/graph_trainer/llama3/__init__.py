# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Modifications Copyright (c) Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the BSD-3-Clause license.

from dataclasses import fields

from torchtitan.experiments.graph_trainer.graph_pp.pipeline import graph_pipeline_llm
from torchtitan.models.llama3 import llama3_configs
from torchtitan.models.llama3.state_dict_adapter import Llama3StateDictAdapter
from torchtitan.models.utils import validate_converter_order
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.model_spec import ModelSpec

from ..common_utils import build_decoder_config_for_backend
from .model import GraphTrainerLlama3Model
from .parallelize import parallelize_llama


def _parallelize_fn(model, *, compile_config, **kwargs):
    if compile_config.enable_autoparallel:
        from .parallelize_autoparallel import parallelize_autoparallel_llama

        return parallelize_autoparallel_llama(
            model, compile_config=compile_config, **kwargs
        )
    return parallelize_llama(model, compile_config=compile_config, **kwargs)


def model_registry(
    flavor: str,
    attn_backend: str = "flex",
    converters: list[ModelConfigConverter.Config] | None = None,
) -> ModelSpec:
    base = build_decoder_config_for_backend(llama3_configs[flavor], attn_backend)
    config = GraphTrainerLlama3Model.Config(
        **{f.name: getattr(base, f.name) for f in fields(base)}
    )
    if converters is not None:
        validate_converter_order(converters)
        for c in converters:
            config = c.build().convert(config)
    return ModelSpec(
        name="graph_trainer/llama3",
        flavor=flavor,
        model=config,
        parallelize_fn=_parallelize_fn,
        pipelining_fn=graph_pipeline_llm,
        post_optimizer_build_fn=None,
        state_dict_adapter=Llama3StateDictAdapter,
    )
