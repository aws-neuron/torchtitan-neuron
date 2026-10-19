# TorchTitan Neuron

TorchTitan Neuron is the AWS Neuron fork of [pytorch/torchtitan](https://github.com/pytorch/torchtitan). It adds AWS Trainium support through [TorchNeuron](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/torch/about-pytorch-neuron.html), the native PyTorch backend for Neuron. TorchTitan's models, parallelism strategies, and `torch.compile` integration run on the `neuron` device. The Neuron-specific recipes, NKI kernel integrations, and docs are in [`torchtitan_neuron/`](torchtitan_neuron/README.md).

## Support matrix

| Component | Supported version |
|---|---|
| Neuron SDK | 2.33.0 (TorchNeuron public beta) |
| PyTorch / `torch-neuronx` | 2.14.0 / 2.14.3.0.* |
| Python | 3.12, 3.13, 3.14 (Linux x86_64) |
| Instances | Trn2, Trn3 |
| Execution | Eager, and `torch.compile` with `--compile.backend neuron` (static shapes only) |
| Models | Llama3, Qwen3, DeepSeek-V3, GPT-OSS |
| Parallelism | FSDP2, SimpleFSDP, TP, CP, EP, PP, DDP, activation checkpointing |

## Quick start

Before you start, install and verify `torch-neuronx` on a Trn2 or Trn3 instance by following [Quickstart: Install and verify torch-neuronx](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/torch/get-started/quickstart-torch-neuron.html). Then install this fork:

```bash
git clone https://github.com/aws-neuron/torchtitan-neuron.git
cd torchtitan-neuron
git checkout release_2.33.0
pip install -r requirements.txt
pip install -e .
```

Run a short verification on one `trn2.48xlarge` (64 logical NeuronCores, TP4 × FSDP16) using the bundled test tokenizer and dataset:

```bash
NGPU=64 MODULE=torchtitan_neuron.models.qwen3 CONFIG=qwen3_8b_tp4_fsdp16_sac \
  ./run_train.sh --training.steps 10 --metrics.log_freq 1 \
  --dataloader.dataset c4_test --hf_assets_path=./tests/assets/tokenizer
```

## Documentation

- [What is TorchTitan Neuron?](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/training/torchtitan/about-torchtitan-neuron.html): architecture, supported models, distributed strategies, and known limitations
- [Set up TorchTitan Neuron](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/training/torchtitan/get-started/setup-torchtitan-neuron.html)
- [TorchTitan Neuron tutorials](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/training/torchtitan/tutorials/index.html)
- [Release notes](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/components/torchtitan-neuron.html)

### Repository guides

Start with the [Neuron package overview](torchtitan_neuron/README.md) for the
package boundary, layout, included components, and recipe selection.

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

## Issues

Report problems with running on Neuron or Trainium in this repository's [issues](https://github.com/aws-neuron/torchtitan-neuron/issues). Report problems that also occur on upstream torchtitan to [pytorch/torchtitan](https://github.com/pytorch/torchtitan/issues).

## Package layout

```text
torchtitan_neuron/
|-- models/       Neuron modules and Trainer configuration registries
|-- experiments/  GraphTrainer configuration registries
|-- components/   Neuron-specific reusable components
|-- docs/         AWS Neuron documentation
`-- tests/        Neuron kernel, Trainer, and GraphTrainer tests
```

## License

This project is licensed under the [BSD 3-Clause License](LICENSE).
