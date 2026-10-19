# Training dense models with torchtitan-neuron

## Setup

```shell
cd /workspace/torchtitan_neuron
pip install -r requirements.txt
pip install -e .
```

## Download the Qwen3-32B tokenizer

```shell
python scripts/download_hf_assets.py --repo_id Qwen/Qwen3-32B --assets tokenizer
```

## Train Qwen3-32B (Eager Mode)

```bash
torchrun --nproc_per_node=64 --local-ranks-filter 0 -m torchtitan.train \
  --module torchtitan_neuron.models.qwen3 \
  --config qwen3_32b_tp4_fsdp16_fullac_eager \
  --parallelism.tensor_parallel_degree=4 \
  --parallelism.data_parallel_shard_degree=-1 \
  --training.local_batch_size=1 \
  --training.global_batch_size=16
```

## Train Qwen3-32B with torch.compile

```bash
torchrun --nproc_per_node=64 --local-ranks-filter 0 -m torchtitan.train \
  --module torchtitan_neuron.models.qwen3 \
  --config qwen3_32b_tp4_fsdp16_fullac_compile \
  --parallelism.tensor_parallel_degree=4 \
  --parallelism.data_parallel_shard_degree=-1 \
  --training.local_batch_size=1 \
  --training.global_batch_size=16
```

## Train Qwen3-32B with SimpleFSDP + torch.compile

```bash
torchrun --nproc_per_node=64 --local-ranks-filter 0 -m torchtitan.train \
  --module torchtitan_neuron.experiments.graph_trainer.qwen3 \
  --config simple_fsdp_qwen3_32b_tp4_fsdp16_fullac \
  --parallelism.tensor_parallel_degree=4 \
  --parallelism.data_parallel_shard_degree=-1 \
  --parallelism.context_parallel_degree=1 \
  --training.local_batch_size=1 \
  --training.global_batch_size=16
```

## Download the Llama3-8B tokenizer

```shell
python scripts/download_hf_assets.py --repo_id meta-llama/Llama-3.1-8B --assets tokenizer --hf_token=...
```

## Train Llama3-8B (Eager Mode)

```bash
torchrun --nproc_per_node=64 --local-ranks-filter 0 -m torchtitan.train \
  --module torchtitan_neuron.models.llama3 \
  --config llama3_8b_sac_eager \
  --parallelism.tensor_parallel_degree=4 \
  --parallelism.data_parallel_shard_degree=-1 \
  --parallelism.context_parallel_degree=1 \
  --training.global_batch_size=16
```

## Train Llama3-8B with torch.compile

```bash
torchrun --nproc_per_node=64 --local-ranks-filter 0 -m torchtitan.train \
  --module torchtitan_neuron.models.llama3 \
  --config llama3_8b_sac_compile \
  --parallelism.tensor_parallel_degree=4 \
  --parallelism.data_parallel_shard_degree=-1 \
  --parallelism.context_parallel_degree=1 \
  --training.global_batch_size=16
```

## Train Llama3-8B with SimpleFSDP + torch.compile

```bash
torchrun --nproc_per_node=64 --local-ranks-filter 0 -m torchtitan.train \
  --module torchtitan_neuron.experiments.graph_trainer.llama3 \
  --config simple_fsdp_llama3_8b \
  --dataloader.dataset=c4_test \
  --parallelism.tensor_parallel_degree=4 \
  --parallelism.data_parallel_shard_degree=-1 \
  --parallelism.context_parallel_degree=1 \
  --training.global_batch_size=16
```

## Multi-node training

```shell
export MASTER_ADDR=<hostname or IP of the rank 0 node>
export CCOM_SOCKET_IFNAME=eth0
```

### Llama3 8B with HSDP

Runs on four nodes. With torch.compile:

```bash
torchrun --nnodes=4 --node_rank=${NODE_RANK} \
  --master_addr=${MASTER_ADDR} --master_port=29500 \
  --nproc_per_node=64 --local-ranks-filter 0 -m torchtitan.train \
  --module torchtitan_neuron.models.llama3 \
  --config llama3_8b_fullac_compile \
  --parallelism.tensor_parallel_degree=4 \
  --parallelism.data_parallel_shard_degree=16 \
  --parallelism.data_parallel_replicate_degree=4 \
  --parallelism.context_parallel_degree=1 \
  --training.global_batch_size=128
```

### Llama3 70B

Download the tokenizer:

```shell
python scripts/download_hf_assets.py --repo_id meta-llama/Llama-3.1-70B --assets tokenizer --hf_token=...
```

Eager:

```bash
torchrun --nnodes=4 --node_rank=${NODE_RANK} \
  --master_addr=${MASTER_ADDR} --master_port=29500 \
  --nproc_per_node=64 --local-ranks-filter 0 -m torchtitan.train \
  --module torchtitan_neuron.models.llama3 \
  --config llama3_70b_fullac_eager \
  --parallelism.tensor_parallel_degree=4 \
  --parallelism.data_parallel_shard_degree=64 \
  --parallelism.context_parallel_degree=1 \
  --training.local_batch_size=1 \
  --training.global_batch_size=128 \
  --training.steps=10
```

torch.compile:

```bash
torchrun --nnodes=4 --node_rank=${NODE_RANK} \
  --master_addr=${MASTER_ADDR} --master_port=29500 \
  --nproc_per_node=64 --local-ranks-filter 0 -m torchtitan.train \
  --module torchtitan_neuron.models.llama3 \
  --config llama3_70b_fullac_compile \
  --parallelism.tensor_parallel_degree=4 \
  --parallelism.data_parallel_shard_degree=64 \
  --parallelism.context_parallel_degree=1 \
  --training.local_batch_size=1 \
  --training.global_batch_size=128 \
  --training.steps=10
```

SimpleFSDP + torch.compile:

```bash
torchrun --nnodes=4 --node_rank=${NODE_RANK} \
  --master_addr=${MASTER_ADDR} --master_port=29500 \
  --nproc_per_node=64 --local-ranks-filter 0 -m torchtitan.train \
  --module torchtitan_neuron.experiments.graph_trainer.llama3 \
  --config simple_fsdp_llama3_70b_fullac \
  --dataloader.dataset=c4_test \
  --parallelism.tensor_parallel_degree=4 \
  --parallelism.data_parallel_shard_degree=64 \
  --parallelism.context_parallel_degree=1 \
  --training.local_batch_size=1 \
  --training.global_batch_size=128 \
  --training.steps=10
```

## Host collectives (HostCC)

To run any config above with host collectives, export these before the command:

```shell
export TORCH_NEURONX_ENABLE_HOST_CC=1
export TORCH_NEURONX_ENABLE_STREAMS=1
```

On more than one node, also set the UltraServer mode. Use `1` for a multiple of
four nodes and `3` for two nodes only:

```shell
export NEURON_RT_ULTRASERVER_MODE=1
```

See the [UltraServer
documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/deploy/eks/ultraserver.html)
for more detail.

The first training step will take a few minutes, but after that, expect
consistent throughput.
