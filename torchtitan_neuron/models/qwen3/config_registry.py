# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

"""Neuron Qwen3 dense and NKI MoE trainer configs."""

from collections.abc import Callable
from dataclasses import dataclass

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw
from torchtitan.components.quantization import MXFP8LinearConverter
from torchtitan.config import CompileConfig, derive, ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.qwen3.config_registry import qwen3_30b_a3b, qwen3_32b
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.tools.profiler import Profiler
from torchtitan.trainer import Trainer

from . import model_registry
from .nki_moe import BWMMGroupedExperts, to_nki_moe_config


def qwen3_235b_a22b() -> Trainer.Config:
    model_spec = model_registry("235B-A22B")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/Qwen3-235B-A22B",
        model_spec=model_spec,
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4"),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=600),
        training=TrainingConfig(
            local_batch_size=2,
            seq_len=4096,
            steps=3000,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=1,
            context_parallel_degree=1,
            pipeline_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=500,
            last_save_model_only=False,
            export_dtype="float16",
        ),
        activation_checkpoint=FullAC.Config(),
    )


@dataclass(frozen=True)
class Qwen3NKIMoEPreset:
    config_factory: Callable[[], Trainer.Config]
    base_flavor: str
    flavor: str


_QWEN3_NKI_MOE_PRESETS = {
    "30B-A3B": Qwen3NKIMoEPreset(
        config_factory=qwen3_30b_a3b,
        base_flavor="30B-A3B",
        flavor="30B-A3B-nki",
    ),
    "235B-A22B": Qwen3NKIMoEPreset(
        config_factory=qwen3_235b_a22b,
        base_flavor="235B-A22B",
        flavor="235B-A22B-nki",
    ),
}


def _reduce_moe_experts(config, *, num_experts: int, top_k: int) -> None:
    for layer in config.model_spec.model.layers:
        moe = getattr(layer, "moe", None)
        if moe is None:
            continue
        moe.num_experts = num_experts
        moe.router.num_experts = num_experts
        moe.router.top_k = top_k
        moe.router.gate.out_features = num_experts
        moe.experts.num_experts = num_experts
        moe.experts.token_dispatcher.num_experts = num_experts
        moe.experts.token_dispatcher.top_k = top_k


def select_neuron_components(model_spec: ModelSpec) -> None:
    for layer in model_spec.model.layers:
        layer.attention.rope.use_prefix_cache_for_positions = True
        moe = getattr(layer, "moe", None)
        if moe is not None:
            layer.moe = to_nki_moe_config(moe)


def qwen3_nki_moe_config(
    preset_name: str,
    *,
    compile_config: CompileConfig | None,
    n_layers: int | None = None,
    num_experts: int | None = None,
    top_k: int | None = None,
) -> Trainer.Config:
    if (num_experts is None) != (top_k is None):
        raise ValueError("num_experts and top_k must be set together")
    if num_experts is not None and top_k is not None and top_k > num_experts:
        raise ValueError(f"top_k={top_k} exceeds num_experts={num_experts}")

    preset = _QWEN3_NKI_MOE_PRESETS[preset_name]
    config = preset.config_factory()
    config.model_spec = model_registry(preset.base_flavor, attn_backend="sdpa")

    flavor = preset.flavor
    if num_experts is not None:
        flavor = f"{flavor}-e{num_experts}-topk{top_k}"
    if n_layers is not None:
        flavor = f"{flavor}-{n_layers}l"
        config.model_spec.model.layers = config.model_spec.model.layers[:n_layers]
    config.model_spec.flavor = flavor

    if num_experts is not None:
        assert top_k is not None
        _reduce_moe_experts(config, num_experts=num_experts, top_k=top_k)
    select_neuron_components(config.model_spec)

    config.compile = compile_config or CompileConfig(enable=False, backend="neuron")
    config.override.imports = []
    config.training.steps = 10
    config.dataloader.dataset = "c4_test"
    config.lr_scheduler.warmup_steps = 10
    for group in config.optimizer.param_groups:
        group.optimizer_kwargs["lr"] = 3e-4
    config.metrics.log_freq = 1
    return config


def qwen3_30b_a3b_nki_fullac_eager() -> Trainer.Config:
    return qwen3_nki_moe_config(
        "30B-A3B",
        compile_config=CompileConfig(enable=False, backend="neuron"),
    )


def qwen3_30b_a3b_nki_fullac_compile() -> Trainer.Config:
    return qwen3_nki_moe_config(
        "30B-A3B",
        compile_config=CompileConfig(
            enable=True,
            components=["model", "loss", "grad_clip"],
            backend="neuron",
        ),
    )


def qwen3_235b_a22b_neuron() -> Trainer.Config:
    """Full-scale Qwen3-235B Neuron recipe."""
    config = qwen3_235b_a22b()
    config.model_spec = model_registry(
        "235B-A22B",
        attn_backend="sdpa",
        moe_comm_backend="neuron_ep",
    )
    config.model_spec.flavor = "235B-A22B-neuron"
    for layer in config.model_spec.model.layers:
        layer.moe.experts = derive(
            layer.moe.experts,
            BWMMGroupedExperts.Config,
        )

    config.training.local_batch_size = 1
    config.training.seq_len = 4096
    config.parallelism = ParallelismConfig(
        data_parallel_replicate_degree=4,
        data_parallel_shard_degree=32,
        tensor_parallel_degree=4,
        pipeline_parallel_degree=2,
        expert_parallel_degree=128,
    )
    config.activation_checkpoint = SelectiveAC.Config()
    config.compile = CompileConfig(
        enable=True,
        components=["model", "loss", "grad_clip"],
        backend="neuron",
    )
    config.override.imports = []
    return config


def qwen3_8b_tp4_fsdp16_sac() -> Trainer.Config:
    model_spec = model_registry("8B", attn_backend="sdpa")
    config = Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/Qwen3-8B",
        model_spec=model_spec,
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4"),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=10),
        training=TrainingConfig(local_batch_size=1, seq_len=4096, steps=50),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=4,
            context_parallel_degree=1,
            pipeline_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=500,
            last_save_model_only=False,
            export_dtype="float16",
        ),
        activation_checkpoint=SelectiveAC.Config(),
    )
    for layer in config.model_spec.model.layers:
        layer.attention.rope.use_prefix_cache_for_positions = True
    return config


def qwen3_8b_tp4_fsdp16_compile_sac() -> Trainer.Config:
    config = qwen3_8b_tp4_fsdp16_sac()
    config.compile = CompileConfig(
        enable=True,
        components=["model", "loss", "grad_clip"],
        backend="neuron",
    )
    return config


def _qwen3_neuron_dense_config(
    flavor: str,
    *,
    tensor_parallel_degree: int,
    global_batch_size: int,
    compile_config: CompileConfig | None,
    selective_ac: bool = False,
    lr: float | None = None,
) -> Trainer.Config:
    config = qwen3_32b()
    # The base is qwen3_32b() (lr 8e-4). 0.6B uses a different lr (3e-4), so allow
    # a per-size override.
    if lr is not None:
        config.optimizer = default_adamw(lr=lr)
    config.model_spec = model_registry(flavor, attn_backend="sdpa")
    # Preserve the sequential-position behavior selected by these recipes.
    for layer in config.model_spec.model.layers:
        layer.attention.rope.use_prefix_cache_for_positions = True
    config.hf_assets_path = f"./assets/hf/Qwen3-{flavor}"
    config.metrics = MetricsProcessor.Config(log_freq=1)
    config.dataloader.dataset = "c4_test"
    config.training.local_batch_size = 1
    config.training.steps = 10
    config.training.global_batch_size = global_batch_size
    config.parallelism.tensor_parallel_degree = tensor_parallel_degree
    config.parallelism.data_parallel_shard_degree = -1
    config.parallelism.context_parallel_degree = 1
    config.parallelism.pipeline_parallel_degree = 1
    config.activation_checkpoint = (
        SelectiveAC.Config() if selective_ac else FullAC.Config()
    )
    config.compile = compile_config or CompileConfig()
    return config


def qwen3_0_6b_tp1_fsdp64_sac_eager() -> Trainer.Config:
    return _qwen3_neuron_dense_config(
        "0.6B",
        tensor_parallel_degree=1,
        global_batch_size=64,
        compile_config=None,
        selective_ac=True,
        lr=3e-4,
    )


def qwen3_0_6b_tp1_fsdp64_fullac_compile() -> Trainer.Config:
    return _qwen3_neuron_dense_config(
        "0.6B",
        tensor_parallel_degree=1,
        global_batch_size=64,
        compile_config=CompileConfig(
            enable=True,
            components=["model", "loss", "grad_clip"],
            backend="neuron",
        ),
        lr=3e-4,
    )


def qwen3_1_7b_tp1_fsdp64_fullac_eager() -> Trainer.Config:
    return _qwen3_neuron_dense_config(
        "1.7B", tensor_parallel_degree=1, global_batch_size=64, compile_config=None
    )


def qwen3_1_7b_tp1_fsdp64_fullac_compile() -> Trainer.Config:
    return _qwen3_neuron_dense_config(
        "1.7B",
        tensor_parallel_degree=1,
        global_batch_size=64,
        compile_config=CompileConfig(
            enable=True,
            components=["model", "loss", "grad_clip"],
            backend="neuron",
        ),
    )


def qwen3_4b_tp4_fsdp16_sac_eager() -> Trainer.Config:
    return _qwen3_neuron_dense_config(
        "4B",
        tensor_parallel_degree=4,
        global_batch_size=16,
        compile_config=None,
        selective_ac=True,
    )


def qwen3_4b_tp4_fsdp16_sac_compile() -> Trainer.Config:
    return _qwen3_neuron_dense_config(
        "4B",
        tensor_parallel_degree=4,
        global_batch_size=16,
        compile_config=CompileConfig(
            enable=True,
            components=["model", "loss", "grad_clip"],
            backend="neuron",
        ),
        selective_ac=True,
    )


def qwen3_8b_tp1_fsdp64_fullac_eager() -> Trainer.Config:
    return _qwen3_neuron_dense_config(
        "8B", tensor_parallel_degree=1, global_batch_size=64, compile_config=None
    )


def qwen3_8b_tp1_fsdp64_fullac_compile() -> Trainer.Config:
    return _qwen3_neuron_dense_config(
        "8B",
        tensor_parallel_degree=1,
        global_batch_size=64,
        compile_config=CompileConfig(
            enable=True,
            components=["model", "loss", "grad_clip"],
            backend="neuron",
        ),
    )


def qwen3_14b_tp4_fsdp16_fullac_eager() -> Trainer.Config:
    return _qwen3_neuron_dense_config(
        "14B", tensor_parallel_degree=4, global_batch_size=16, compile_config=None
    )


def qwen3_14b_tp4_fsdp16_fullac_compile() -> Trainer.Config:
    return _qwen3_neuron_dense_config(
        "14B",
        tensor_parallel_degree=4,
        global_batch_size=16,
        compile_config=CompileConfig(
            enable=True,
            components=["model", "loss", "grad_clip"],
            backend="neuron",
        ),
    )


def qwen3_32b_tp4_fsdp16_fullac_eager() -> Trainer.Config:
    return _qwen3_neuron_dense_config(
        "32B", tensor_parallel_degree=4, global_batch_size=16, compile_config=None
    )


def qwen3_32b_tp4_fsdp16_fullac_compile() -> Trainer.Config:
    return _qwen3_neuron_dense_config(
        "32B",
        tensor_parallel_degree=4,
        global_batch_size=16,
        compile_config=CompileConfig(
            enable=True,
            components=["model", "loss", "grad_clip"],
            backend="neuron",
        ),
    )


def _qwen3_235b_4layers(
    moe_comm_backend: str,
    n_layers: int = 4,
    ep_degree: int = 4,
    num_experts: int | None = None,
    top_k: int | None = None,
    tp_degree: int = 1,
) -> Trainer.Config:
    """Build a reduced-layer Qwen3-235B expert-parallel recipe."""
    if (num_experts is None) != (top_k is None):
        raise ValueError("num_experts and top_k must be set together.")
    model_spec = model_registry(
        "235B-A22B",
        attn_backend="sdpa",
        moe_comm_backend=moe_comm_backend,
    )
    model_spec.model.layers = model_spec.model.layers[:n_layers]
    if num_experts is not None:
        assert top_k is not None
        for layer in model_spec.model.layers:
            moe = layer.moe
            moe.num_experts = num_experts
            moe.router.num_experts = num_experts
            moe.router.top_k = top_k
            moe.router.gate.out_features = num_experts
            moe.experts.num_experts = num_experts
            moe.experts.token_dispatcher.num_experts = num_experts
            moe.experts.token_dispatcher.top_k = top_k
    if moe_comm_backend == "neuron_ep":
        for layer in model_spec.model.layers:
            layer.moe.experts = derive(
                layer.moe.experts,
                BWMMGroupedExperts.Config,
            )
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=default_adamw(lr=3e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
        training=TrainingConfig(local_batch_size=1, seq_len=512, steps=5),
        parallelism=ParallelismConfig(
            expert_parallel_degree=ep_degree,
            tensor_parallel_degree=tp_degree,
        ),
        checkpoint=CheckpointManager.Config(
            interval=1000,
            last_save_model_only=False,
            export_dtype="float16",
        ),
        activation_checkpoint=None,
    )


def _scaled_235b_config(moe_comm_backend: str, ep_degree: int) -> Trainer.Config:
    model_spec = model_registry(
        "235B-A22B-scaled",
        attn_backend="sdpa",
        moe_comm_backend=moe_comm_backend,
    )
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=default_adamw(lr=3e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
        training=TrainingConfig(local_batch_size=1, seq_len=512, steps=5),
        parallelism=ParallelismConfig(expert_parallel_degree=ep_degree),
        checkpoint=CheckpointManager.Config(
            interval=1000,
            last_save_model_only=False,
            export_dtype="float16",
        ),
        activation_checkpoint=None,
    )


def qwen3_235b_scaled_singlerank() -> Trainer.Config:
    return _scaled_235b_config("neuron_ep", ep_degree=1)


def _qwen3_235b_scaled_ep4(moe_comm_backend: str) -> Trainer.Config:
    return _scaled_235b_config(moe_comm_backend, ep_degree=4)


def qwen3_235b_scaled_ep4_standard() -> Trainer.Config:
    return _qwen3_235b_scaled_ep4(moe_comm_backend="standard")


def qwen3_235b_scaled_64rank_noep() -> Trainer.Config:
    return _scaled_235b_config("standard", ep_degree=1)


def qwen3_235b_4layers_neuron_standard() -> Trainer.Config:
    return _qwen3_235b_4layers(moe_comm_backend="standard")


def qwen3_235b_2layers_neuron_standard() -> Trainer.Config:
    return _qwen3_235b_4layers(moe_comm_backend="standard", n_layers=2)


def qwen3_235b_4layers_ep64_nle1_neuron_ep() -> Trainer.Config:
    return _qwen3_235b_4layers(
        moe_comm_backend="neuron_ep",
        ep_degree=64,
        num_experts=64,
        top_k=8,
    )


def qwen3_235b_4layers_ep64_nle1_neuron_ep_profile() -> Trainer.Config:
    config = qwen3_235b_4layers_ep64_nle1_neuron_ep()
    config.training.steps = 8
    config.profiler = Profiler.Config(
        enable_profiling=True,
        profile_freq=6,
        profiler_warmup=2,
        profiler_active=1,
    )
    return config


def qwen3_235b_4layers_ep64_nle1_tp4_neuron_ep() -> Trainer.Config:
    return _qwen3_235b_4layers(
        moe_comm_backend="neuron_ep",
        ep_degree=64,
        num_experts=64,
        top_k=8,
        tp_degree=4,
    )


def qwen3_235b_4layers_ep64_nle1_tp4_neuron_ep_compile() -> Trainer.Config:
    config = qwen3_235b_4layers_ep64_nle1_tp4_neuron_ep()
    config.compile = CompileConfig(
        enable=True,
        components=["loss"],
        backend="neuron",
    )
    return config


def qwen3_235b_6layers_ep64_nle1_tp4_neuron_ep() -> Trainer.Config:
    config = _qwen3_235b_4layers(
        moe_comm_backend="neuron_ep",
        n_layers=6,
        ep_degree=64,
        num_experts=64,
        top_k=8,
        tp_degree=4,
    )
    config.training.seq_len = 4096
    config.activation_checkpoint = SelectiveAC.Config()
    return config


def qwen3_235b_6layers_ep64_nle1_tp4_neuron_ep_compile() -> Trainer.Config:
    config = qwen3_235b_6layers_ep64_nle1_tp4_neuron_ep()
    config.compile = CompileConfig(
        enable=True,
        components=["model", "loss"],
        backend="neuron",
    )
    return config


def qwen3_235b_6layers_ep128_nle1_tp4_neuron_ep_compile() -> Trainer.Config:
    config = _qwen3_235b_4layers(
        moe_comm_backend="neuron_ep",
        n_layers=6,
        ep_degree=128,
        num_experts=128,
        top_k=8,
        tp_degree=4,
    )
    config.training.seq_len = 4096
    config.activation_checkpoint = SelectiveAC.Config()
    config.compile = CompileConfig(
        enable=True,
        components=["model", "loss"],
        backend="neuron",
    )
    return config


def qwen3_235b_94layers_ep128_tp4_fsdp64_pp1_4k_neuron_ep_compile() -> (Trainer.Config):
    config = _qwen3_235b_4layers(
        moe_comm_backend="neuron_ep",
        n_layers=94,
        ep_degree=128,
        num_experts=128,
        top_k=8,
        tp_degree=4,
    )
    config.training.seq_len = 4096
    config.training.local_batch_size = 1
    config.activation_checkpoint = FullAC.Config()
    config.compile = CompileConfig(
        enable=True,
        components=["model", "loss"],
        backend="neuron",
    )
    config.parallelism.data_parallel_replicate_degree = 1
    config.parallelism.data_parallel_shard_degree = 64
    config.parallelism.pipeline_parallel_degree = 1
    return config


def qwen3_235b_6layers_ep64_nle1_tp4_neuron_ep_pp2_compile() -> Trainer.Config:
    config = qwen3_235b_6layers_ep64_nle1_tp4_neuron_ep_compile()
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.pipeline_parallel_schedule = "1F1B"
    config.parallelism.pipeline_parallel_first_stage_less_layers = 0
    config.parallelism.pipeline_parallel_last_stage_less_layers = 0
    config.training.local_batch_size = 16
    return config


def qwen3_235b_6layers_ep128_nle1_tp4_neuron_ep_pp2_compile() -> Trainer.Config:
    config = qwen3_235b_6layers_ep128_nle1_tp4_neuron_ep_compile()
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.pipeline_parallel_schedule = "1F1B"
    config.parallelism.pipeline_parallel_first_stage_less_layers = 0
    config.parallelism.pipeline_parallel_last_stage_less_layers = 0
    config.training.local_batch_size = 16
    return config


def qwen3_235b_4layers_ep64_neuron_standard() -> Trainer.Config:
    return _qwen3_235b_4layers(moe_comm_backend="standard", ep_degree=64)


# --- Neuron trn3 dense qwen3-8B MXFP8 (eager by default) ---
# Gating, backend selection, and the torch.compile path: see
# torchtitan_neuron/docs/neuron_mxfp8.md.


def qwen3_8b_mxfp8() -> Trainer.Config:
    """qwen3-8B MXFP8 on all in-layer linears (attention + FFN); LM head stays bf16.

    Inherits ``qwen3_8b_tp4_fsdp16_sac`` (SDPA + prefix RoPE cache, TP4 x FSDP16,
    SelectiveAC) and only attaches ``MXFP8LinearConverter`` (``fqns=["layers"]``,
    ``kernel_preference="neuron"``).
    """
    config = qwen3_8b_tp4_fsdp16_sac()
    model_compile_enabled = (
        config.compile.enable and "model" in config.compile.components
    )
    config.model_spec.model = (
        MXFP8LinearConverter.Config(
            fqns=["layers"],
            kernel_preference="neuron",
            model_compile_enabled=model_compile_enabled,
        )
        .build()
        .convert(config.model_spec.model)
    )
    return config
