# Neuron attention backends

TorchTitan's Neuron recipes select the attention implementation for training on
AWS Trainium. Select a recipe through `--module` and `--config`; the recipe
configures attention together with the model's Neuron components and compilation
settings.

This guide describes the attention behavior of the recipes in
`torchtitan_neuron`. Run them in an environment with PyTorch and `torch_neuronx`
configured for Neuron. Recipes using NKI kernels also require `nkilib`.

## Backends used by Neuron recipes

| Model | Attention backend | Behavior |
| --- | --- | --- |
| Llama3 | `sdpa` | Causal scaled dot-product attention, including grouped-query attention |
| DeepSeek-V3 | `sdpa` | Causal scaled dot-product attention |
| Qwen3 | `sdpa` | Causal scaled dot-product attention, including grouped-query attention |
| GPT-OSS | `nki` | Causal attention with native attention sinks, grouped-query attention, and per-layer sliding windows |

The attention backend is separate from `compile.backend`. For example, a
compiled Llama3 recipe uses `sdpa` attention and the `neuron` compilation
backend. Selecting an NKI MoE recipe for DeepSeek-V3 or Qwen3 does not change
its attention backend to `nki`.

### SDPA

Llama3, DeepSeek-V3, and Qwen3 recipes use TorchTitan's
`ScaledDotProductAttention` wrapper around PyTorch's
`scaled_dot_product_attention`. These recipes use causal attention: a token
can attend to itself and earlier tokens in its sequence.

The TorchTitan wrapper controls masking through `is_causal` and rejects
explicit `attention_masks`. It does not implement sliding-window attention
or packed-document isolation. This is a constraint of the TorchTitan wrapper,
not of the full PyTorch SDPA API.

### GPT-OSS NKI attention

The GPT-OSS recipe selects
[`NKIFlashAttentionWrapper`](../models/gpt_oss/nki_attention.py).
The wrapper uses NKI forward and backward kernels and computes gradients for
queries, keys, values, and learned attention sinks. It is integrated with
`torch.compile` through Neuron custom operators.

GPT-OSS alternates between 128-token causal windows and full causal attention.
The wrapper accepts `sliding_window=128` or `sliding_window=None`; other window
sizes raise `ValueError` when the module is constructed. Keep the window
selection supplied by the model recipe.

The `nki` attention backend is specific to GPT-OSS. Replacing it with `sdpa`
is not supported for its sliding-window layers: the model registry rejects
that combination.

## Packed-document behavior

**Neither SDPA nor GPT-OSS NKI attention enforces document boundaries within
a packed sequence.** If a sequence contains document A followed by document B,
tokens in B can attend to earlier tokens in A. A sliding window limits how far
back a token can attend, but does not isolate documents.

TorchTitan's text dataloader packs documents into sequences. Position resets
and end-of-document tokens do not add an attention mask to these backends.
Constructing an SDPA configuration logs a warning about this behavior.

If your training objective requires strict document isolation, use a data
pipeline that keeps each sequence within one document, or an attention
implementation that consumes document boundaries. The Neuron recipes listed
here do not provide document-boundary masking.

## Selecting a Trainer recipe

Pass the registry module as `--module` and the recipe name as `--config`.
The following recipes select the backends described above:

| Registry module | Recipe |
| --- | --- |
| [`torchtitan_neuron.models.llama3.config_registry`](../models/llama3/config_registry.py) | `llama3_8b_sac_compile` |
| [`torchtitan_neuron.models.deepseek_v3.config_registry`](../models/deepseek_v3/config_registry.py) | `deepseek_v3_16b_neuron_tp4_fsdp16` |
| [`torchtitan_neuron.models.qwen3.config_registry`](../models/qwen3/config_registry.py) | `qwen3_30b_a3b_nki_fullac_compile` |
| [`torchtitan_neuron.models.gpt_oss.config_registry`](../models/gpt_oss/config_registry.py) | `gpt_oss_20b_nki_tp4_fsdp16` |

For example, launch the GPT-OSS recipe on a single host with `torchrun`.
Set `NEURON_PROCESSES_PER_NODE` to the worker count configured for your Neuron
host and `HF_ASSETS_PATH` to the local GPT-OSS tokenizer and model assets:

```bash
torchrun --standalone --nproc_per_node="${NEURON_PROCESSES_PER_NODE}" \
  -m torchtitan.train \
  --module torchtitan_neuron.models.gpt_oss.config_registry \
  --config gpt_oss_20b_nki_tp4_fsdp16 \
  --hf_assets_path="${HF_ASSETS_PATH}"
```

This recipe uses tensor parallelism of degree 4 and infers its data-parallel
shard degree from the world size. The total worker count must be divisible by
4; TP4 with FSDP16 uses 64 workers. Choose a topology with enough memory for
the model and training state. For multiple hosts, use your distributed
launcher's node and rendezvous settings with the same module and recipe.

### GraphTrainer recipes

The Neuron GraphTrainer recipes retain the corresponding Trainer recipe's
attention selection and use SimpleFSDP with Neuron JIT compilation:

| Registry module | Recipe |
| --- | --- |
| [`torchtitan_neuron.experiments.graph_trainer.llama3.config_registry`](../experiments/graph_trainer/llama3/config_registry.py) | `simple_fsdp_llama3_8b` |
| [`torchtitan_neuron.experiments.graph_trainer.deepseek_v3.config_registry`](../experiments/graph_trainer/deepseek_v3/config_registry.py) | `simple_fsdp_deepseek_v3_16b_nki_tp4_fsdp16` |
| [`torchtitan_neuron.experiments.graph_trainer.qwen3.config_registry`](../experiments/graph_trainer/qwen3/config_registry.py) | `qwen3_30b_a3b_nki_fullac_simplefsdp` |
| [`torchtitan_neuron.experiments.graph_trainer.gpt_oss.config_registry`](../experiments/graph_trainer/gpt_oss/config_registry.py) | `simple_fsdp_gpt_oss_20b_nki_tp4_fsdp16` |

Use the same training entrypoint, replacing `--module` and `--config` with
the GraphTrainer selection.

## Configuring a training workload

Dataset and schedule defaults vary by recipe. The Trainer recipes listed above
use the bundled `c4_test` dataset; Llama3 GraphTrainer inherits `c4` from the core
Llama3 configuration. GPT-OSS and Qwen3 NKI recipes default to 10 training steps;
DeepSeek-V3 defaults to 20. Set
`--dataloader.dataset`, `--hf_assets_path`, `--training.steps`, batch sizes,
sequence length, learning-rate schedule, checkpoint settings, and
`--dump_folder` for your workload. The recipe defaults do not define a full
training schedule.

See [dataset configuration](../../docs/datasets.md) for input data,
[checkpointing](../../docs/checkpoint.md) for saving and resuming training,
and [Qwen3 NKI MoE](qwen3_nki_moe.md) for Qwen3 recipe details.
