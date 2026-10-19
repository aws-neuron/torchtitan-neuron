# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

"""DeepSeek-V3 recipes that require NKI kernels and the Neuron JIT path."""

from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.experiments.graph_trainer.deepseek_v3 import model_registry
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer

from torchtitan_neuron.experiments.graph_trainer.configs import neuron_simple_fsdp
from torchtitan_neuron.models.deepseek_v3.config_registry import (
    deepseek_v3_16b_neuron_tp4_fsdp16,
)


def simple_fsdp_deepseek_v3_16b_nki_tp4_fsdp16() -> GraphTrainer.Config:
    config = neuron_simple_fsdp(
        deepseek_v3_16b_neuron_tp4_fsdp16(),
        model_registry,
        model_name="simple_fsdp/deepseek_v3",
    )
    config.training.local_batch_size = 4
    config.training.global_batch_size = 64
    config.training.seq_len = 4096
    config.activation_checkpoint = SelectiveAC.Config()
    return config


def simple_fsdp_deepseek_v3_16b_nki_tp4_fsdp16_fac() -> GraphTrainer.Config:
    config = simple_fsdp_deepseek_v3_16b_nki_tp4_fsdp16()
    config.activation_checkpoint = FullAC.Config()
    return config
