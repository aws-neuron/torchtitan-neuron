# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp
from torchtitan.experiments.graph_trainer.common_utils import (
    annotate_module_fqns,
    annotate_moe_ep_regions,
    apply_simple_fsdp,
)
from torchtitan.experiments.graph_trainer.compile import apply_compile
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig

from .model import GraphTrainerGptOssModel


def annotate_gptoss(model: GraphTrainerGptOssModel) -> None:
    """Attach module FQN annotations for graph-trainer passes."""
    if any(layer.moe is not None for layer in model.config.layers):
        annotate_moe_ep_regions()
    annotate_module_fqns(model)


def parallelize_gptoss(
    model: GraphTrainerGptOssModel,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: GraphTrainerCompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
):
    assert (
        not parallel_dims.cp_enabled
    ), "GraphTrainer GPT-OSS NKI does not currently support context parallelism"
    assert (
        not parallel_dims.ep_enabled
    ), "GraphTrainer GPT-OSS NKI does not currently support expert parallelism"

    assert (
        training.seq_len % parallel_dims.seq_len_divisor == 0
    ), f"""
        Sequence length {training.seq_len} must be divisible by the product of TP degree
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}), i.e. {parallel_dims.seq_len_divisor}.
        """

    annotate_gptoss(model)

    if parallel_dims.tp_enabled:
        model.parallelize(parallel_dims)

    if parallel_dims.tp_enabled:
        maybe_enable_async_tp(parallelism, compile_config, parallel_dims.get_mesh("tp"))

    if compile_config.mode == "jit" and ac_config is not None:
        # JIT mode does not run graph-trainer's AOT memory-policy passes, so
        # apply the eager activation-checkpoint policy before SimpleFSDP.
        ac_config.build(dump_folder=dump_folder).apply(model)

    # Apply SimpleFSDP unconditionally. The fsdp mesh exists even at degree 1,
    # which lets MixedPrecisionPolicy still cast parameters in single-rank runs.
    model = apply_simple_fsdp(model, parallel_dims=parallel_dims, training=training)

    model = apply_compile(
        model,
        compile_config=compile_config,
        parallelism=parallelism,
        parallel_dims=parallel_dims,
        dump_folder=dump_folder,
    )

    return model
