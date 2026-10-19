<div align="center">

# TorchTitan on AWS Neuron

### Train generative AI models at scale on AWS Trainium

[Overview](#overview) | [Quick start](#quick-start) | [Documentation](#documentation) | [Testing](#testing)

</div>

## Overview

TorchTitan on AWS Neuron provides Neuron-specific components for training
generative AI models on AWS Trainium. The `torchtitan_neuron` package is
installed alongside `torchtitan` and uses the same configuration and launch
interfaces.

The package includes:

- Neuron training recipes for Llama3, Qwen3, DeepSeek-V3, and GPT-OSS.
- Dense and mixture-of-experts training with eager mode, `torch.compile`,
  FSDP2, and SimpleFSDP configurations.
- Neuron Kernel Interface (NKI) attention and MoE kernels.
- MXFP8 training support on Trainium3.
- Neuron-specific configuration builders, tests, and documentation.

See the [`torchtitan_neuron` package overview](torchtitan_neuron/README.md) for
the package boundary, layout, included components, and recipe selection.

## Quick start

Install the repository in a configured AWS Neuron environment:

```bash
cd /workspace/torchtitan_neuron
pip install -r requirements.txt
pip install -e .
```

Select a Neuron recipe by passing its registry module and configuration name to
the TorchTitan training entrypoint. For example:

```bash
NGPU=64 MODULE=torchtitan_neuron.models.qwen3 \
  CONFIG=qwen3_8b_tp4_fsdp16_sac ./run_train.sh
```

Continue with the [dense model](torchtitan_neuron/docs/dense-tutorials.md) or
[MoE model](torchtitan_neuron/docs/moe-tutorials.md) tutorial for model assets,
topology settings, and complete launch commands.

## Documentation

| Topic | Guide |
| --- | --- |
| Dense Qwen3 and Llama3 training | [Dense model tutorials](torchtitan_neuron/docs/dense-tutorials.md) |
| Qwen3, DeepSeek-V3, and GPT-OSS MoE training | [MoE model tutorials](torchtitan_neuron/docs/moe-tutorials.md) |
| SDPA and NKI attention behavior | [Neuron attention backends](torchtitan_neuron/docs/neuron_attention_backends.md) |
| MXFP8 training on Trainium3 | [Neuron MXFP8](torchtitan_neuron/docs/neuron_mxfp8.md) |
| Qwen3 NKI mixture-of-experts recipes | [Qwen3 NKI MoE](torchtitan_neuron/docs/qwen3_nki_moe.md) |
| Full Qwen3-235B-A22B training | [Four-node Trainium3 UltraServer guide](torchtitan_neuron/docs/qwen3_235b_full_ep128.md) |
| Qwen3 inference with `torch.compile` | [Qwen3 inference](torchtitan_neuron/docs/qwen3_inference.md) |
| Neuron extension test suite | [Testing `torchtitan_neuron`](torchtitan_neuron/tests/README.md) |

## Package layout

```text
torchtitan_neuron/
|-- models/       Neuron modules and Trainer configuration registries
|-- experiments/  GraphTrainer configuration registries
|-- components/   Neuron-specific reusable components
|-- docs/         AWS Neuron documentation
`-- tests/        Neuron kernel, Trainer, and GraphTrainer tests
```

## Testing

Run the Neuron extension test suite from the repository root:

```bash
pytest -s torchtitan_neuron/tests/
```

Hardware-specific tests require AWS Trainium and the corresponding AWS Neuron
dependencies. Tests that do not require Neuron hardware skip when those
dependencies are unavailable.

## License

This project is licensed under the [BSD 3-Clause License](LICENSE).
