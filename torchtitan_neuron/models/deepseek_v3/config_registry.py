# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

"""Neuron-native DeepSeek-V3 trainer configs."""

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw
from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.deepseek_v3 import model_registry
from torchtitan.trainer import Trainer

from .nki_moe import to_nki_moe_config


def deepseek_v3_16b_neuron_tp4_fsdp16() -> Trainer.Config:
    model_spec = model_registry("16B", attn_backend="sdpa")
    config = Trainer.Config(
        compile=CompileConfig(
            enable=True,
            components=["model", "loss", "grad_clip"],
            backend="neuron",
        ),
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=default_adamw(lr=2.2e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=10,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=4,
            global_batch_size=64,
            seq_len=4096,
            steps=20,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=4,
            expert_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            enable=True,
            interval=10,
            load_only=True,
        ),
        activation_checkpoint=FullAC.Config(),
    )

    for layer in config.model_spec.model.layers:
        layer.attention.rope.use_prefix_cache_for_positions = True
        if layer.moe is not None:
            layer.moe = to_nki_moe_config(layer.moe)

    return config
