# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

"""Neuron-native GPT-OSS trainer configs."""

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw
from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.gpt_oss import model_registry
from torchtitan.trainer import Trainer

from .nki_moe import to_nki_moe_config


def gpt_oss_20b_nki_tp4_fsdp16() -> Trainer.Config:
    model_spec = model_registry("20b", attn_backend="nki")
    for layer in model_spec.model.layers:
        layer.moe = to_nki_moe_config(layer.moe)
        # Neuron does not support the RoPE gather; use the prefix-cache path.
        layer.attention.rope.use_prefix_cache_for_positions = True
    return Trainer.Config(
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
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=10,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=4096,
            steps=10,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=4,
            expert_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(interval=10),
        activation_checkpoint=FullAC.Config(),
    )
