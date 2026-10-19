# TorchTitan on AWS Neuron

`torchtitan_neuron` contains the Neuron-specific components used to train
TorchTitan models on AWS Trainium. It is installed alongside `torchtitan` and
uses the same configuration and launch interfaces.

## Package boundary

This package owns self-contained Neuron kernels, kernel wrappers, recipes,
configuration builders, tests, and documentation. It may import the
device-independent TorchTitan core. Core code should only import
`torchtitan_neuron` lazily at a routing boundary.

## Layout

```
torchtitan_neuron/
├── models/       Neuron modules and Trainer config registries.
├── experiments/ GraphTrainer config registries and shared conversion helpers.
├── docs/         Neuron-specific documentation.
└── tests/        Neuron kernel, Trainer, and GraphTrainer tests.
```

## Included components

- DeepSeek-V3 NKI MoE Trainer and GraphTrainer recipes.
- GPT-OSS NKI attention and MoE Trainer and GraphTrainer recipes.
- Llama3 dense Trainer and SimpleFSDP GraphTrainer recipes.
- Qwen3 dense/MXFP8 Trainer and NKI MoE Trainer/GraphTrainer recipes.
- The shared `neuron_simple_fsdp` GraphTrainer converter.

Select a recipe by passing its registry module and configuration name to the
TorchTitan training entrypoint.

For example, the Qwen3 dense prefix-cache recipe is selected explicitly:

```bash
NGPU=64 MODULE=torchtitan_neuron.models.qwen3 \
  CONFIG=qwen3_8b_tp4_fsdp16_sac ./run_train.sh
```

RoPE remains a core component. Recipes that select sequential prefix positions
set `layer.attention.rope.use_prefix_cache_for_positions = True`; importing
`torchtitan_neuron` or selecting a device does not enable this behavior.
This preserves the earlier overrides' position handling, including ignoring
document position resets. Core recipes retain their default position gather.


## Attention

Llama3, DeepSeek-V3, and Qwen3 Neuron recipes use causal SDPA attention.
GPT-OSS uses NKI attention with native attention sinks and alternates between
128-token causal windows and full causal attention. The recipe selects attention
independently of its MoE implementation and compilation backend.

These attention paths do not isolate documents packed into the same sequence.
See [Neuron attention backends](docs/neuron_attention_backends.md) for masking
behavior, recipe selection, launch instructions, and training configuration.

## Documentation

- [Dense model tutorials](docs/dense-tutorials.md)
- [MoE model tutorials](docs/moe-tutorials.md)
- [Neuron attention backends](docs/neuron_attention_backends.md)
- [Qwen3 NKI MoE](docs/qwen3_nki_moe.md)
- [Full Qwen3-235B training on a Trn3 UltraServer](docs/qwen3_235b_full_ep128.md)
