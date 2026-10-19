# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

"""Qwen3 NKI MoE SimpleFSDP Neuron configs."""

from dataclasses import fields, replace

from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import (
    default_adamw,
    register_moe_load_balancing_hook,
)
from torchtitan.components.quantization import MXFP8LinearConverter
from torchtitan.config import CompileConfig
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    to_graph_trainer_config,
)
from torchtitan.experiments.graph_trainer.qwen3.model import GraphTrainerQwen3Model
from torchtitan.experiments.graph_trainer.qwen3.parallelize import parallelize_qwen3
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitan.models.qwen3 import model_registry as dense_qwen3_model_registry
from torchtitan.models.qwen3.config_registry import qwen3_32b, use_global_batch_lbl
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.trainer import Trainer

from torchtitan_neuron.components.quantization.mx_prequant import (
    NeuronMXFP8LinearConverter,
)
from torchtitan_neuron.experiments.graph_trainer.configs import neuron_simple_fsdp
from torchtitan_neuron.models.qwen3._mxfp8_flavors import register_mxfp8_flavors
from torchtitan_neuron.models.qwen3.config_registry import (
    qwen3_235b_4layers_ep64_nle1_tp4_neuron_ep,
    qwen3_235b_6layers_ep128_nle1_tp4_neuron_ep_compile,
    qwen3_235b_6layers_ep128_nle1_tp4_neuron_ep_pp2_compile,
    qwen3_235b_6layers_ep64_nle1_tp4_neuron_ep_compile,
    qwen3_235b_6layers_ep64_nle1_tp4_neuron_ep_pp2_compile,
    qwen3_235b_scaled_64rank_noep,
    qwen3_nki_moe_config,
)

from . import model_registry as graph_model_registry


def graph_trainer_qwen3_235b_scaled() -> GraphTrainer.Config:
    config = to_graph_trainer_config(
        qwen3_235b_scaled_64rank_noep(),
        graph_model_registry,
    )
    config.compile = GraphTrainerCompileConfig(enable=True)
    return config


def graph_trainer_qwen3_235b_4layers_ep64_nle1_tp4_neuron_ep() -> (GraphTrainer.Config):
    config = to_graph_trainer_config(
        qwen3_235b_4layers_ep64_nle1_tp4_neuron_ep(),
        graph_model_registry,
    )
    config.compile = GraphTrainerCompileConfig(enable=True)
    return config


def _neuron_qwen3_simple_fsdp_config(
    flavor: str,
    *,
    tensor_parallel_degree: int,
    global_batch_size: int,
    selective_ac: bool = False,
    lr: float | None = None,
    converters: list[ModelConfigConverter.Config] | None = None,
) -> GraphTrainer.Config:
    base = qwen3_32b()
    if lr is not None:
        base.optimizer = default_adamw(lr=lr)
    base.model_spec = dense_qwen3_model_registry(
        flavor,
        attn_backend="sdpa",
        converters=converters,
    )
    for layer in base.model_spec.model.layers:
        layer.attention.rope.use_prefix_cache_for_positions = True
    base.hf_assets_path = f"./assets/hf/Qwen3-{flavor}"
    base.metrics = MetricsProcessor.Config(log_freq=1)
    base.dataloader.dataset = "c4_test"
    base.training.local_batch_size = 1
    base.training.steps = 10
    base.training.global_batch_size = global_batch_size
    base.parallelism.tensor_parallel_degree = tensor_parallel_degree
    base.parallelism.data_parallel_shard_degree = -1
    base.parallelism.context_parallel_degree = 1
    base.parallelism.pipeline_parallel_degree = 1
    config = to_graph_trainer_config(base, graph_model_registry)
    config.model_spec = replace(config.model_spec, name="simple_fsdp/qwen3")
    config.activation_checkpoint = (
        SelectiveAC.Config() if selective_ac else FullAC.Config()
    )
    config.compile = GraphTrainerCompileConfig(
        enable=True,
        components=["model", "loss", "grad_clip"],
        backend="neuron",
        mode="jit",
    )
    return config


def simple_fsdp_qwen3_0_6b_tp1_fsdp64_fullac() -> GraphTrainer.Config:
    return _neuron_qwen3_simple_fsdp_config(
        "0.6B",
        tensor_parallel_degree=1,
        global_batch_size=64,
        lr=3e-4,
    )


def simple_fsdp_qwen3_1_7b_tp1_fsdp64_fullac() -> GraphTrainer.Config:
    return _neuron_qwen3_simple_fsdp_config(
        "1.7B",
        tensor_parallel_degree=1,
        global_batch_size=64,
    )


def simple_fsdp_qwen3_4b_tp4_fsdp16_sac() -> GraphTrainer.Config:
    return _neuron_qwen3_simple_fsdp_config(
        "4B",
        tensor_parallel_degree=4,
        global_batch_size=16,
        selective_ac=True,
    )


def simple_fsdp_qwen3_8b_tp1_fsdp64_fullac() -> GraphTrainer.Config:
    return _neuron_qwen3_simple_fsdp_config(
        "8B",
        tensor_parallel_degree=1,
        global_batch_size=64,
    )


def simple_fsdp_qwen3_14b_tp4_fsdp16_fullac() -> GraphTrainer.Config:
    return _neuron_qwen3_simple_fsdp_config(
        "14B",
        tensor_parallel_degree=4,
        global_batch_size=16,
    )


def simple_fsdp_qwen3_32b_tp4_fsdp16_fullac() -> GraphTrainer.Config:
    return _neuron_qwen3_simple_fsdp_config(
        "32B",
        tensor_parallel_degree=4,
        global_batch_size=16,
    )


def simple_fsdp_qwen3_32b_tp4_fsdp16_sac() -> GraphTrainer.Config:
    return _neuron_qwen3_simple_fsdp_config(
        "32B",
        tensor_parallel_degree=4,
        global_batch_size=16,
        selective_ac=True,
    )


def qwen3_moe_simple_fsdp_neuron_model_registry(
    base_config: Trainer.Config,
) -> ModelSpec:
    base = base_config.model_spec.model
    model_config = GraphTrainerQwen3Model.Config(
        **{field.name: getattr(base, field.name) for field in fields(base)}
    )
    return ModelSpec(
        name="simple_fsdp/qwen3",
        flavor=base_config.model_spec.flavor,
        model=model_config,
        parallelize_fn=parallelize_qwen3,
        pipelining_fn=None,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=None,
    )


def _to_qwen3_neuron_graph_trainer_config(
    config: Trainer.Config,
) -> GraphTrainer.Config:
    """Use the core Qwen3 JIT parallelizer while preserving Neuron recipe state."""
    model_spec = qwen3_moe_simple_fsdp_neuron_model_registry(config)
    return neuron_simple_fsdp(
        config,
        lambda _flavor: model_spec,
        model_name="simple_fsdp/qwen3",
        model_spec=model_spec,
    )


def _qwen3_moe_simple_fsdp_neuron_compile(
    preset_name: str,
    *,
    n_layers: int | None = None,
    num_experts: int | None = None,
    top_k: int | None = None,
) -> GraphTrainer.Config:
    config = qwen3_nki_moe_config(
        preset_name,
        compile_config=CompileConfig(
            enable=True,
            components=["model", "loss", "grad_clip"],
            backend="neuron",
        ),
        n_layers=n_layers,
        num_experts=num_experts,
        top_k=top_k,
    )
    return _to_qwen3_neuron_graph_trainer_config(config)


def qwen3_30b_a3b_nki_fullac_simplefsdp() -> GraphTrainer.Config:
    return _qwen3_moe_simple_fsdp_neuron_compile("30B-A3B")


def qwen3_30b_a3b_nki_global_lbl_fullac_simplefsdp() -> GraphTrainer.Config:
    config = qwen3_nki_moe_config(
        "30B-A3B", compile_config=CompileConfig(enable=True, backend="neuron")
    )
    return _to_qwen3_neuron_graph_trainer_config(use_global_batch_lbl(config))


def qwen3_235b_6layers_ep64_nle1_tp4_neuron_ep_compile_simplefsdp() -> (
    GraphTrainer.Config
):
    """Six-layer SimpleFSDP neuron_ep compile configuration."""
    config = qwen3_235b_6layers_ep64_nle1_tp4_neuron_ep_compile()
    return _to_qwen3_neuron_graph_trainer_config(config)


def qwen3_235b_6layers_ep128_nle1_tp4_neuron_ep_compile_simplefsdp() -> (
    GraphTrainer.Config
):
    """EP128/4K SimpleFSDP counterpart of the FSDP2 configuration."""
    config = qwen3_235b_6layers_ep128_nle1_tp4_neuron_ep_compile()
    return _to_qwen3_neuron_graph_trainer_config(config)


def qwen3_235b_6layers_ep64_nle1_tp4_neuron_ep_pp2_compile_simplefsdp() -> (
    GraphTrainer.Config
):
    """pp=2 (1F1B) SimpleFSDP variant of the reduced-EP config.

    Inherits pp=2 / 1F1B / lbs=16 from the FSDP2 pp2 config, then swaps in the
    SimpleFSDP model_spec. That fresh spec defaults pipelining_fn=None, so we
    restore PP here
    """
    config = qwen3_235b_6layers_ep64_nle1_tp4_neuron_ep_pp2_compile()
    config = _to_qwen3_neuron_graph_trainer_config(config)
    config.model_spec = replace(config.model_spec, pipelining_fn=pipeline_llm)
    return config


def qwen3_235b_6layers_ep128_nle1_tp4_neuron_ep_pp2_compile_simplefsdp() -> (
    GraphTrainer.Config
):
    """pp=2 (1F1B) SimpleFSDP variant of the EP128/4K config (4 Trn3 nodes).

    Inherits pp=2 / 1F1B / lbs=16 from the FSDP2 ep128 pp2 config, then swaps in the
    SimpleFSDP model_spec. That fresh spec defaults pipelining_fn=None, so we
    restore PP here
    """
    config = qwen3_235b_6layers_ep128_nle1_tp4_neuron_ep_pp2_compile()
    config = _to_qwen3_neuron_graph_trainer_config(config)
    config.model_spec = replace(config.model_spec, pipelining_fn=pipeline_llm)
    return config


# --- Dense Qwen3 SimpleFSDP MXFP8 (reuses the Neuron SimpleFSDP helper;
# dim6144/dim8192 flavors are registered by
# torchtitan_neuron.models.qwen3._mxfp8_flavors) ---


def simple_fsdp_qwen3_32b_tp4_fsdp16_fullac_mxfp8() -> GraphTrainer.Config:
    """Qwen3-32B SimpleFSDP TP4 x FSDP16 FullAC, MXFP8 on all in-layer linears.

    ``fqns=["layers"]`` converts per-layer wqkv/wo/w1/w2/w3; lm_head and embeddings
    stay bf16.
    """
    return _neuron_qwen3_simple_fsdp_config(
        "32B",
        tensor_parallel_degree=4,
        global_batch_size=16,
        converters=[
            MXFP8LinearConverter.Config(
                fqns=["layers"],
                kernel_preference="neuron",
                model_compile_enabled=True,
            )
        ],
    )


def simple_fsdp_qwen3_32b_tp4_fsdp16_sac_mxfp8() -> GraphTrainer.Config:
    """Qwen3-32B SimpleFSDP TP4 x FSDP16 SAC, MXFP8 on all in-layer linears.

    Selective-AC variant of the FullAC MXFP8 config; same ``fqns=["layers"]``
    conversion.
    """
    return _neuron_qwen3_simple_fsdp_config(
        "32B",
        tensor_parallel_degree=4,
        global_batch_size=16,
        selective_ac=True,
        converters=[
            MXFP8LinearConverter.Config(
                fqns=["layers"],
                kernel_preference="neuron",
                model_compile_enabled=True,
            )
        ],
    )


def simple_fsdp_qwen3_dim6144_6layers_tp4_fsdp16_fullac_mxfp8() -> GraphTrainer.Config:
    """Qwen3 6-layer SimpleFSDP TP4 x FSDP16 FullAC MXFP8 on 128-aligned shapes.

    Pre-quantizes the SimpleFSDP weights with the 1x32 scheme (128-aligned per-rank
    shards). Requires SimpleFSDP dim-1 sharding support.
    """
    register_mxfp8_flavors()
    config = _neuron_qwen3_simple_fsdp_config(
        "32B_dim6144_ffn24576_6layers",
        tensor_parallel_degree=4,
        global_batch_size=16,
        converters=[
            NeuronMXFP8LinearConverter.Config(
                fqns=["layers"],
                kernel_preference="neuron",
                model_compile_enabled=True,
                prequantize_weights=True,
                prequant_scheme="1x32",
            )
        ],
    )
    # Random init on c4_test, no HF weights loaded, so only the tokenizer is needed and
    # the in-repo one suffices (its 2016 entries fit inside the model's vocab_size).
    config.hf_assets_path = "./tests/assets/tokenizer"
    return config


def simple_fsdp_qwen3_dim8192_6layers_tp4_fsdp16_fullac_mxfp8() -> GraphTrainer.Config:
    """Qwen3 6-layer SimpleFSDP TP4 x FSDP16 FullAC MXFP8 on 512-aligned shapes.

    Pre-quantizes the SimpleFSDP weights with the wrapX scheme (512-aligned per-rank
    K). Requires SimpleFSDP dim-1 sharding support.
    """
    register_mxfp8_flavors()
    config = _neuron_qwen3_simple_fsdp_config(
        "32B_dim8192_ffn32768_6layers",
        tensor_parallel_degree=4,
        global_batch_size=16,
        converters=[
            NeuronMXFP8LinearConverter.Config(
                fqns=["layers"],
                kernel_preference="neuron",
                model_compile_enabled=True,
                prequantize_weights=True,
                prequant_scheme="wrapX",
            )
        ],
    )
    # Random init on c4_test, so only the tokenizer is needed and the in-repo one fits.
    config.hf_assets_path = "./tests/assets/tokenizer"
    return config
