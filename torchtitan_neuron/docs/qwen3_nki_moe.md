# Qwen3 NKI MoE

The Neuron Qwen3 registries replace the standard mixture-of-experts
implementation with the Qwen3 NKI MoE kernel. They provide recipes for both
Trainer and GraphTrainer.

## Neuron components

`torchtitan_neuron.models.qwen3.config_registry` constructs Qwen3 with:

- [causal SDPA attention](neuron_attention_backends.md#sdpa)
- the sequential RoPE prefix-cache path, which avoids a per-position gather
- `NKIMoE.Config` in place of the standard MoE implementation
- the compiled gradient-norm override when model compilation is enabled

The launchable NKI MoE recipes documented below use the Qwen3 `30B-A3B`
model preset.

Attention remains SDPA when NKI MoE is enabled. Documents packed into the
same sequence can attend across document boundaries; see
[packed-document behavior](neuron_attention_backends.md#packed-document-behavior)
before choosing a data pipeline.

## Trainer recipes

Run Trainer with model compilation disabled:

```bash
python -m torchtitan.train \
  --module torchtitan_neuron.models.qwen3.config_registry \
  --config qwen3_30b_a3b_nki_fullac_eager \
  --hf_assets_path=./assets/hf/Qwen3-30B-A3B
```

Run Trainer with the Neuron compilation backend:

```bash
python -m torchtitan.train \
  --module torchtitan_neuron.models.qwen3.config_registry \
  --config qwen3_30b_a3b_nki_fullac_compile \
  --hf_assets_path=./assets/hf/Qwen3-30B-A3B
```

## GraphTrainer recipe

The GraphTrainer recipe uses SimpleFSDP and Neuron JIT compilation:

```bash
python -m torchtitan.train \
  --module torchtitan_neuron.experiments.graph_trainer.qwen3.config_registry \
  --config qwen3_30b_a3b_nki_fullac_simplefsdp \
  --hf_assets_path=./assets/hf/Qwen3-30B-A3B \
  --parallelism.tensor_parallel_degree=4 \
  --parallelism.data_parallel_shard_degree=-1 \
  --parallelism.enable_async_tensor_parallel
```

## Global-batch load-balancing loss

`qwen3_30b_a3b_nki_global_lbl_fullac_simplefsdp` trains with Qwen3's
global-batch load-balancing loss (`load_balancing="global_aux_loss"`) instead
of the default expert bias. Each MoE layer adds
`load_balance_coeff * E * sum_i f_i * P_i`, where `f_i` is the fraction of
routed tokens assigned to expert `i` (from token counts summed over the data-
and context-parallel ranks) and `P_i` is expert `i`'s mean router probability
over the global batch. To enable it on another
Qwen3 configuration, apply
`torchtitan.models.qwen3.config_registry.use_global_batch_lbl`. It is refused
with a non-default `spmd_backend`, GraphTrainer EP overlap, and precompile.

## Training configuration

These recipes use the bundled `c4_test` dataset, 10 training steps, 10 warmup
steps, and per-step metrics so that a new installation can perform a bounded
smoke run. Before starting a training workload, override the dataset, asset
path, step count, checkpoint, output, and topology settings for the target
environment.
