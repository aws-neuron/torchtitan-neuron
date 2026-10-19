# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

"""GPT-OSS recipes that require NKI kernels and the Neuron JIT path."""

from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer

from torchtitan_neuron.experiments.graph_trainer.configs import neuron_simple_fsdp
from torchtitan_neuron.models.gpt_oss.config_registry import gpt_oss_20b_nki_tp4_fsdp16

from . import model_registry


def simple_fsdp_gpt_oss_20b_nki_tp4_fsdp16() -> GraphTrainer.Config:
    config = neuron_simple_fsdp(
        gpt_oss_20b_nki_tp4_fsdp16(),
        model_registry,
        model_name="simple_fsdp/gpt_oss",
    )
    config.training.local_batch_size = 4
    config.training.seq_len = 4096
    config.activation_checkpoint = FullAC.Config()
    return config
