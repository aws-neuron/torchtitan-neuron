# Training MoE models with torchtitan-neuron

## Setup

```shell
cd /workspace/torchtitan_neuron
pip install -r requirements.txt
pip install -e .
```

## Train Qwen3-30B-A3B with SimpleFSDP + torch.compile

Download the tokenizer:

```shell
python scripts/download_hf_assets.py --repo_id Qwen/Qwen3-30B-A3B --assets tokenizer
```

Run:

```bash
torchrun --nproc_per_node=64 --local-ranks-filter 0 -m torchtitan.train \
  --module torchtitan_neuron.experiments.graph_trainer.qwen3 \
  --config qwen3_30b_a3b_nki_fullac_simplefsdp \
  --parallelism.tensor_parallel_degree=4 \
  --parallelism.data_parallel_shard_degree=-1 \
  --parallelism.enable_async_tensor_parallel \
  --training.local_batch_size=4 \
  --training.global_batch_size=64
```

## Train DeepSeek-V3-16B with SimpleFSDP + torch.compile

```bash
torchrun --nproc_per_node=64 --local-ranks-filter 0 -m torchtitan.train \
  --module torchtitan_neuron.experiments.graph_trainer.deepseek_v3 \
  --config simple_fsdp_deepseek_v3_16b_nki_tp4_fsdp16 \
  --parallelism.tensor_parallel_degree=4 \
  --parallelism.data_parallel_shard_degree=-1 \
  --parallelism.expert_parallel_degree=1 \
  --training.local_batch_size=4 \
  --training.global_batch_size=64 \
  --training.steps=20
```

## Train GPT-OSS-20B with SimpleFSDP + torch.compile

```bash
torchrun --nproc_per_node=64 --local-ranks-filter 0 -m torchtitan.train \
  --module torchtitan_neuron.experiments.graph_trainer.gpt_oss \
  --config simple_fsdp_gpt_oss_20b_nki_tp4_fsdp16 \
  --training.local_batch_size=4 \
  --training.seq_len=4096 \
  --training.steps=20
```

The first training step will take a few minutes, but after that, expect
consistent throughput.
