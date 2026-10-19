# Qwen3-235B training on a four-node Trainium3 UltraServer

This recipe runs 50 training steps of the full 94-layer Qwen3-235B-A22B model using
TorchTitan, FSDP2, and full activation checkpointing (FullAC).

It uses TorchTitan's bundled tokenizer and C4 sample, starting from randomly
initialized model weights.

## Prerequisites

Start with a working four-node Trainium3 UltraServer using your image on all four
nodes. Use the image's `python` and `torchrun` from `PATH`.

Open a Bash shell inside each node's container. Paste the command blocks below
into those shells in order, keeping the same shell open so the exported variables
carry into the launch command. You do not need to create a Bash script.
No package installation, branch checkout, or cherry-pick is needed.

| Setting &emsp;&emsp; | Value |
| --- | --- |
| Model &emsp;&emsp; | Qwen3-235B-A22B, all 94 layers |
| Parallelism &emsp;&emsp; | EP128, TP4, FSDP64 |
| Sequence length &emsp;&emsp; | 4,096 tokens |
| Batch size &emsp;&emsp; | 1 local, 64 global |
| Activation checkpointing &emsp;&emsp; | FullAC |
| Training steps &emsp;&emsp; | 50 |

The four nodes run 256 processes. Dense parameters use FSDP64 and TP4. For expert
parameters, EP128 distributes the 128 experts across 128 positions, and FSDP
splits each expert's weights across two ranks. FSDP gathers those weights when
needed for computation. This is the named configuration's standard layout.

## 1. Select the tokenizer and dataset

The tokenizer and C4 sample are included in your image. Set these paths on every
node before proceeding to the launch environment.

```bash
mkdir -p /workspace/outputs
export ASSET_ROOT=/workspace/torchtitan/tests/assets
export TOKENIZER_PATH="$ASSET_ROOT/tokenizer"
```

The loader tokenizes and packs the C4 sample into training sequences, repeating
the sample as needed.

## 2. Resolve node topology

Run this inside each node's container to identify its physical UltraServer position:

```bash
neuron-ls --topology
```

Confirm all four outputs have the same `ultraserver-id`, `ultraserver-size: 4`,
and distinct `ultraserver-instance-index` values covering `0`, `1`, `2`, and `3`.
On each node, set `NODE_RANK` to its reported index:

```bash
export NODE_RANK=0  # Use this node's ultraserver-instance-index: 0, 1, 2, or 3.
```

For this four-node Trn3 layout, physical indices `0` and `1` share one rack,
and `2` and `3` share the other. With 64 processes per node, the configuration
forms EP128 groups from node ranks `0` + `1` and `2` + `3`, keeping each EP
group within a rack. FSDP communication can still cross racks. Use the physical
indices, not hostname or container ordering; `torchrun` does not choose this
placement for you.

## 3. Set the launch environment

On every node, use the same reachable address of the node whose
`ultraserver-instance-index` is `0`, and the same master port.
Keep the `NODE_RANK` assigned in the previous step.

```bash
export MASTER_ADDR='NODE_0_REACHABLE_ADDRESS'
export MASTER_PORT=29500

export TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=0
export NEURON_CC_FLAGS="--enable-mixed-precision-accumulation --enable-saturate-infinity --auto-cast=none"
export CCOM_SOCKET_IFNAME=eth0
export NEURON_RT_DBG_MAX_A2AV_BLOCK_SIZE=8388608
```

Set `CCOM_SOCKET_IFNAME` to the reachable network interface if it differs from `eth0`.

**Why set `NEURON_RT_DBG_MAX_A2AV_BLOCK_SIZE=8388608`?** This performance knob
specifies an upper bound on the A2A-v payload per peer, in bytes. For this
configuration, 8 MiB avoids scheduling unnecessary transfer iterations.

Set the flag on every rank. Recalculate the bound if you change the batch, sequence
length, precision, or parallelism: an undersized value can omit required transfers.

## 4. Launch once on each node

Paste this command into each of the four prepared shells, using the respective
`NODE_RANK` set above. Start all four launches; the first waits for the others to join.

`-m torchtitan.train` selects the installed Python entrypoint.
`--module` selects TorchTitan's configuration registry.

```bash
cd /workspace/torchtitan || exit 1
torchrun \
  --nnodes=4 \
  --nproc-per-node=64 \
  --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" \
  --master-port="$MASTER_PORT" \
  --max-restarts=0 \
  -m torchtitan.train \
  --module torchtitan_neuron.models.qwen3.config_registry \
  --config qwen3_235b_94layers_ep128_tp4_fsdp64_pp1_4k_neuron_ep_compile \
  --training.local_batch_size=1 \
  --training.global_batch_size=64 \
  --training.seq_len=4096 \
  --training.dtype=float32 \
  --training.mixed_precision_param=bfloat16 \
  --parallelism.enable_sequence_parallel \
  --debug.seed=42 \
  --training.steps=50 \
  --lr_scheduler.warmup_steps=2 \
  --metrics.log_freq=1 \
  --dataloader.dataset=c4_test \
  --dataloader.dataset_path="$ASSET_ROOT/c4_test" \
  --hf_assets_path="$TOKENIZER_PATH" \
  --dump_folder=/workspace/outputs/qwen235b-full94
```

The named configuration supplies FullAC, the parallelism degrees, and Neuron
model/loss compilation. The command sets the run length to 50 steps and the seed
to 42. Checkpoint saving is disabled by default; ordinary Titan logging remains
enabled.

## 5. Read the result

Check the startup configuration for 256 ranks, 94 layers, global batch 64,
sequence length 4,096, and FullAC.

Initial compilation can make early steps substantially slower. Use Titan's
standard loss, gradient-norm, throughput, step-time, and MFU records, and identify
the steady-state step window when reporting performance. Learning-rate warmup
alone does not establish that compilation has finished.

Record the delivered image digest, tokenizer/dataset paths, and any configuration
changes alongside results.
