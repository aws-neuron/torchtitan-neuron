#!/usr/bin/env python3
# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

"""Full Qwen3-235B-A22B MoE expert-block microbenchmark (router INCLUDED).

Unlike moe_dispatcher_microbench.py -- which feeds a pre-synthesized routing
decision straight into dispatch -- this bench runs the WHOLE MoE expert block end
to end, starting from the hidden activation:

    router (gate -> softmax -> top-k -> route_norm)
      -> dispatch (router-reorder + permute_a2av)
        -> blockwise experts (moe_cte NKI kernel)
          -> combine (unpermute_a2av)

The router is the REAL Qwen3 ``TokenChoiceTopKRouter`` built from the exact
235B-A22B recipe (dim=4096, num_experts=128, top_k=8, score_func="softmax",
route_norm=True); the gate is a live nn.Linear, so expert assignments are
data-dependent on the input activation rather than pre-baked.

Fidelity note -- there is NO single importable Qwen3-235B block that is
simultaneously (a2av-dispatched) AND (blockwise-NKI experts):
  * torchtitan/models/common/moe.py::MoE      -> real router + permute/unpermute
    a2av, but grouped_mm experts (NOT blockwise).
  * torchtitan_neuron/models/qwen3/nki_moe.py::NKIMoE -> real router + blockwise nki_fwd,
    but TP-ONLY (raises NotImplementedError under EP; no a2av at all).
So the a2av+blockwise combination only exists as a COMPOSITION. This bench imports
the real Qwen3 router and the real neuron_ep dispatcher (permute_a2av /
unpermute_a2av) and composes them with the blockwise moe_cte experts -- the same
composition validated in moe_dispatcher_microbench.py, now fronted by the router.

Parallelism (token-count-only fidelity):
  Real 235B run = TP4 x DP16 = 64 ranks, EP=64 overlaying all ranks. Under EP,
  torchtitan requires sequence-parallel activations, so each rank's router input is
  the SP shard of its DP replica's sequence: T_local = seq_per_replica // TP. The
  router gate is REPLICATED (not TP-sharded) in the EP path, and expert weights are
  pure-EP (this benchmark uses 64 experts, one per rank at EP=64, with no
  expert-TP). So TP4's
  ONLY effect here is the per-rank token count -- no weight/DTensor sharding is
  materialized; each rank builds its own num_local_experts of full expert weights.

  Defaults: seq_per_replica=4096, TP=4 -> T_local=1024 tokens/rank; EP=64 ->
  num_local_experts=1; N = T_local*top_k = 8192; packed expert capacity is
  align(T_local*EP) = 65536 rows/rank.

Timing: the full block is ONE timed region with NO intra-block host barriers -- a
single torch.neuron.synchronize() at the end lands router+dispatch+experts+combine
before the timer stops, preserving device overlap. The correctness checksum is taken
outside the timed region (last collected iter only) so its D2H copy never pollutes
timing or a --profile capture.

Launch (64 ranks == 64 NeuronCores):
    torchrun --nproc_per_node=64 moe_block_microbench.py

Environment variables (shared with moe_dispatcher_microbench.py):
    NEURON_RT_VISIBLE_CORES=0-63
    NEURON_USE_NKI_ALLTOALLV=1
    NEURON_LOGICAL_NC_CONFIG=2
    TORCH_NEURONX_ENABLE_LAZY_ALLOC=0
    NEURON_RT_EXEC_TIMEOUT=120
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from functools import partial

import torch
import torch.distributed as dist

# Registers the "neuron" device + distributed backend + the mesh_registry hook
# that neuron_ep's build_ep_replica_group_ranks reads.
import torch_neuronx  # noqa: F401
from torch import nn
from torch.distributed.device_mesh import init_device_mesh


# ---- Qwen3-235B-A22B MoE recipe (torchtitan/models/qwen3/__init__.py::_235b_a22b) ----
H = 4096  # dim (D)
FFN = 1536  # moe_hidden_dim (F); per-expert intermediate
TOP_K = 8
NUM_EXPERTS = 64  # reduced from the 235B recipe's 128 (at EP=64 -> num_local_experts=1)
SCORE_FUNC = "softmax"
ROUTE_NORM = True
DEFAULT_SEQ_PER_REPLICA = 4096  # sequence length per DP replica
DEFAULT_TP = 4  # tensor-parallel degree (folds into per-rank token count)
DEFAULT_EP = 64  # expert-parallel group size (= world for pure EP)
DEFAULT_BLOCK_SIZE = 512


def log(rank, msg):
    print(f"[rank {rank}] {msg}", flush=True)
    sys.stdout.flush()


def build_router(device):
    """Build the REAL Qwen3-235B TokenChoiceTopKRouter and initialize the gate.

    Uses the same factory the model uses (make_router_config) with the 235B recipe.
    The gate is a live nn.Linear(H, NUM_EXPERTS); trunc_normal init mirrors the
    model's _LINEAR_INIT so scores are realistic. Router runs on the (replicated) EP
    path, so no TP sharding is applied.
    """
    from torchtitan.models.common.config_utils import make_router_config

    router_cfg = make_router_config(
        dim=H,
        num_experts=NUM_EXPERTS,
        gate_param_init={
            "weight": partial(nn.init.trunc_normal_, std=0.02),
            "bias": nn.init.zeros_,
        },
        top_k=TOP_K,
        score_func=SCORE_FUNC,
        route_norm=ROUTE_NORM,
    )
    router = router_cfg.build()
    router.init_states(buffer_device=torch.device(device))
    return router.to(device)


def route(router, x_TD):
    """Run the real router and derive the dispatcher's routing-decision contract.

    Mirrors MoE.forward's router glue (torchtitan/models/common/moe.py): the router
    returns per-token top-k scores and expert ids; num_local_tokens_per_expert_E is
    the per-expert routed-token count from the one-hot routing map. Returns
    (topk_scores_TK f32, topk_expert_ids_TK i32, num_local_tokens_per_expert_E i32).
    """
    T, D = x_TD.shape
    # Router expects (B, L, D); treat the rank's tokens as one (1, T, D) batch.
    x_BLD = x_TD.view(1, T, D)
    topk_scores_BLK, topk_expert_ids_BLK, scores_BLE = router(x_BLD, None)

    topk_scores_TK = topk_scores_BLK.view(T, TOP_K).to(torch.float32)
    topk_expert_ids_TK = topk_expert_ids_BLK.view(T, TOP_K).to(torch.int32)

    routing_map_TE = torch.zeros_like(scores_BLE, dtype=torch.bool).view(T, NUM_EXPERTS)
    routing_map_TE.scatter_(-1, topk_expert_ids_TK.to(torch.int64), True)
    num_local_tokens_per_expert_E = routing_map_TE.sum(dim=0).to(torch.int32)
    return topk_scores_TK, topk_expert_ids_TK, num_local_tokens_per_expert_E


def make_expert_weights(num_local_experts, device):
    """Random per-expert SwiGLU weights in GroupedExperts param shapes (bf16-ready)."""
    g = torch.Generator().manual_seed(42)
    scale = 1.0 / (H**0.5)
    w1_EFD = (torch.randn(num_local_experts, FFN, H, generator=g) * scale).to(device)
    w3_EFD = (torch.randn(num_local_experts, FFN, H, generator=g) * scale).to(device)
    w2_EDF = (torch.randn(num_local_experts, H, FFN, generator=g) * scale).to(device)
    return w1_EFD, w3_EFD, w2_EDF


def repack_weights_for_moe_cte(weights):
    """Repack grouped-mm expert weights into moe_cte's layout.

    Same as moe_dispatcher_microbench.repack_weights_for_moe_cte:
      w1/w3_EFD [E,FFN,H], w2_EDF [E,H,FFN]  ->
      gate_up [E, H, 2, FFN] (0=gate 1=up), down [E, FFN, H], both bf16.
    """
    w1_EFD, w3_EFD, w2_EDF = weights
    gate = w1_EFD.transpose(-2, -1)  # [E, H, FFN]
    up = w3_EFD.transpose(-2, -1)  # [E, H, FFN]
    gate_up = (
        torch.stack([gate, up], dim=2).contiguous().to(torch.bfloat16)
    )  # [E,H,2,FFN]
    down = w2_EDF.transpose(-2, -1).contiguous().to(torch.bfloat16)  # [E,FFN,H]
    return gate_up, down


def moe_cte_experts_forward(
    x_RD, num_local_experts, gate_up_w, down_w, block_size, spec, lnc
):
    """Blockwise-NKI expert compute over the padded expert-major dispatch buffer.

    Identical to moe_dispatcher_microbench.moe_cte_experts_forward: x_RD is
    [num_local_experts * (EP*cap), H] with each expert's rows contiguous, so block
    metadata is IDENTITY (block b -> expert (b*B)//rows_per_expert,
    token_position_to_id = arange(R), affinity = 1; routing scores applied in
    combine). moe_cte tiles internally by block_size, so it compiles where a single
    grouped-mm over R rows would overflow the neuronx-cc 5M-instruction limit.
    """
    from nki_blockwise_moe_cte import wrapped_moe_cte  # shared Neuron test helper

    R, Hd = x_RD.shape
    device = x_RD.device
    B = block_size
    assert R % B == 0, f"R={R} not divisible by block_size={B}"
    rows_per_expert = R // num_local_experts
    assert rows_per_expert % B == 0, "rows_per_expert must be a block_size multiple"
    n_blk = R // B

    hidden = torch.cat([x_RD, x_RD.new_zeros(1, Hd)], dim=0)  # [R+1, H] (dummy pad row)
    aff = torch.zeros(R + 1, num_local_experts, dtype=torch.float32, device=device)
    row_expert = torch.arange(R, device=device) // rows_per_expert
    aff[torch.arange(R, device=device), row_expert] = 1.0
    aff = aff.reshape((R + 1) * num_local_experts, 1)
    tpid = torch.arange(R, device=device, dtype=torch.int32)  # identity, all rows real
    blk_expert = torch.arange(n_blk, device=device) * B // rows_per_expert
    block_to_expert = blk_expert.to(torch.int32).reshape(-1, 1)

    out = wrapped_moe_cte[(lnc,)](
        hidden,
        aff,
        gate_up_w,
        down_w,
        tpid,
        block_to_expert,
        B,
        spec,
    )
    out = out[0] if isinstance(out, tuple) else out
    return out[:R].type_as(x_RD)


def summarize(name, samples_s, rank):
    """Log mean / p50 / p90 for a phase (samples in seconds -> ms)."""
    ms = sorted(s * 1e3 for s in samples_s)
    mean = statistics.mean(ms)
    p50 = ms[len(ms) // 2]
    p90 = ms[min(len(ms) - 1, int(0.9 * len(ms)))]
    log(
        rank,
        f"  {name:<12} mean={mean:8.2f}ms  p50={p50:8.2f}ms  p90={p90:8.2f}ms  (n={len(ms)})",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ep", type=int, default=DEFAULT_EP, help="expert-parallel group size"
    )
    ap.add_argument(
        "--tp",
        type=int,
        default=DEFAULT_TP,
        help="tensor-parallel degree; only sets per-rank token count "
        "(T_local = seq-per-replica // tp)",
    )
    ap.add_argument(
        "--seq-per-replica",
        type=int,
        default=DEFAULT_SEQ_PER_REPLICA,
        help="sequence length per DP replica (before SP sharding)",
    )
    ap.add_argument(
        "--block-size", type=int, default=DEFAULT_BLOCK_SIZE, help="moe_cte block size"
    )
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument(
        "--profile",
        action="store_true",
        help="capture Neuron DEVICE (.ntff) + RUNTIME (system) traces after timing",
    )
    ap.add_argument(
        "--profile-dir",
        default=None,
        help="profile output dir (default: $MB_PROFILE_DIR or ./mb_profile)",
    )
    ap.add_argument(
        "--profile-active",
        type=int,
        default=2,
        help="profiled steady-state steps to capture",
    )
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    ep = args.ep
    dist.init_process_group(backend="neuron")

    assert world_size % ep == 0, f"world_size {world_size} not divisible by ep {ep}"
    assert NUM_EXPERTS % ep == 0, f"num_experts {NUM_EXPERTS} not divisible by ep {ep}"
    assert ep == NUM_EXPERTS, (
        f"packed neuron_ep requires one local expert per rank: "
        f"ep ({ep}) must equal num_experts ({NUM_EXPERTS})."
    )
    assert (
        args.seq_per_replica % args.tp == 0
    ), f"seq-per-replica {args.seq_per_replica} not divisible by tp {args.tp}"
    num_local_experts = NUM_EXPERTS // ep
    # SP shards the DP-replica sequence across the TP group: each rank routes T_local.
    T = args.seq_per_replica // args.tp
    device = "neuron"

    # Real neuron_ep dispatcher (permute_a2av / unpermute_a2av), packed capacity.
    # Build from the package factory so the benchmark exercises the package implementation
    # directly, with no standalone copy to drift.
    from torchtitan.models.common.config_utils import make_token_dispatcher_config

    dispatcher = make_token_dispatcher_config(
        num_experts=NUM_EXPERTS, top_k=TOP_K, comm_backend="neuron_ep"
    ).build()

    # 2D (efsdp, ep) mesh; ep innermost so replica groups are contiguous, matching the
    # parallel_dims sparse-mesh ordering. wire_meshes(tp_mesh=None): experts are pure-EP.
    mesh = init_device_mesh(
        "neuron", (world_size // ep, ep), mesh_dim_names=("efsdp", "ep")
    )
    ep_mesh = mesh["ep"]
    dispatcher.wire_meshes(ep_mesh=ep_mesh, tp_mesh=None)

    from torch_neuronx.utils import get_logical_neuron_cores

    lnc = int(get_logical_neuron_cores())

    router = build_router(device)
    x_TD = torch.randn(T, H, dtype=torch.bfloat16, device=device)
    weights = make_expert_weights(num_local_experts, device)
    moe_cte_gw, moe_cte_dw = repack_weights_for_moe_cte(weights)
    import nki_blockwise_moe_cte as _bw

    moe_cte_spec = _bw.default_spec()

    if rank == 0:
        log(
            rank,
            f"Qwen3-235B-A22B MoE block  world={world_size} ep={ep} tp={args.tp} "
            f"local_experts={num_local_experts} seq_per_replica={args.seq_per_replica} "
            f"T_local={T} H={H} FFN={FFN} top_k={TOP_K} score={SCORE_FUNC} "
            f"route_norm={ROUTE_NORM}",
        )
        log(
            rank,
            f"experts=moe_cte block_size={args.block_size} lnc={lnc} "
            f"warmup={args.warmup} iters={args.iters}",
        )

    def one_step(collect):
        """Run the FULL MoE block once: router -> dispatch -> blockwise experts ->
        combine. No intra-block host barriers; a single torch.neuron.synchronize()
        lands the whole pipeline before the timer stops. Checksum is taken outside
        the timed region (only when ``collect``) so its D2H never pollutes timing.
        """
        t0 = time.time()
        # Router: gate -> softmax -> top-k -> route_norm (data-dependent on x).
        scores_TK, ids_TK, counts_E = route(router, x_TD)
        # Dispatch: router-reorder + permute_a2av.
        routed_input_RD, _counts_e, meta = dispatcher.dispatch(
            x_TD, scores_TK, ids_TK, counts_E
        )
        # Experts: blockwise moe_cte NKI kernel over the padded expert-major buffer.
        routed_output_RD = moe_cte_experts_forward(
            routed_input_RD,
            num_local_experts,
            moe_cte_gw,
            moe_cte_dw,
            args.block_size,
            moe_cte_spec,
            lnc,
        )
        # Combine: unpermute_a2av (+ scatter_add with routing scores).
        out_TD = dispatcher.combine(
            routed_output_RD,
            meta,
            x_TD,
            num_local_tokens_after_padding=T,
            local_seq_len_after_padding=T,
        )
        torch.neuron.synchronize()  # single barrier for the whole block
        t1 = time.time()
        if collect:
            checksum = float(out_TD.detach().to("cpu").float().abs().sum().item())
            return (t1 - t0,), checksum
        return None, None

    # ---- Warmup (compile + cache the NEFFs; excluded from timing) ----
    dist.barrier()
    log(rank, f"=== warmup x{args.warmup} ===")
    for _ in range(args.warmup):
        one_step(collect=False)
    dist.barrier()

    # ---- Timed iters (full MoE-block latency) ----
    log(rank, f"=== timing x{args.iters} ===")
    full = []
    last_checksum = None
    for _ in range(args.iters):
        times, last_checksum = one_step(collect=True)
        full.append(times[0])
    dist.barrier()

    log(
        rank,
        f"=== RESULTS Qwen3-235B MoE block ep={ep} checksum={last_checksum:.3e} ===",
    )
    summarize("full-block", full, rank)

    if args.profile:
        _capture_profile(args, rank, one_step)

    dist.barrier()
    if rank == 0:
        log(rank, "ALL RANKS COMPLETED SUCCESSFULLY (Qwen3-235B MoE block)")
    dist.destroy_process_group()


def _capture_profile(args, rank, one_step):
    """Capture DEVICE (.ntff hardware) + RUNTIME (system) traces for Neuron Explorer.

    Same knobs as moe_dispatcher_microbench._capture_profile: both DEVICE and RUNTIME
    modes plus CPU_UTIL/HOST_MEMORY, oversized event ring buffer to avoid silent drops,
    and both LNC cores captured. NEFFs are already warm from the timing loop.
    """
    from torch.profiler import profile, ProfilerActivity, schedule
    from torch_neuronx.profiling import NeuronConfig, NeuronProfiler, ProfileMode

    profile_dir = args.profile_dir or os.environ.get("MB_PROFILE_DIR") or "./mb_profile"
    rank_dir = os.path.join(
        profile_dir, f"qwen3_235b_moe_block_ep{args.ep}", f"rank{rank}"
    )
    os.makedirs(rank_dir, exist_ok=True)
    neuron_config = NeuronConfig(
        modes=[
            ProfileMode.DEVICE,
            ProfileMode.RUNTIME,
            ProfileMode.CPU_UTIL,
            ProfileMode.HOST_MEMORY,
        ],
        profile_output_dir=rank_dir,
        max_events_per_nc=10_000_000,
        capture_enabled_for_nc="0,1",
    )
    exporter = NeuronProfiler(neuron_config)

    if rank == 0:
        log(
            rank,
            f"=== PROFILING (DEVICE+RUNTIME) active={args.profile_active} dir={profile_dir} ===",
        )
    dist.barrier()
    sched = schedule(wait=1, warmup=1, active=args.profile_active, repeat=1)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
        experimental_config=neuron_config,
        schedule=sched,
        on_trace_ready=exporter.export_trace,
        record_shapes=True,
    ) as prof:
        for _ in range(2 + args.profile_active):
            one_step(collect=False)
            torch.neuron.synchronize()
            prof.step()
    try:
        prof.export_chrome_trace(os.path.join(rank_dir, f"framework_rank{rank}.json"))
    except Exception as e:  # noqa: BLE001
        log(rank, f"chrome-trace export skipped: {e}")
    dist.barrier()
    if rank == 0:
        log(rank, f"=== PROFILE session dir: {neuron_config.get_session_dir()} ===")
        log(rank, "=== PROFILE DONE ===")


if __name__ == "__main__":
    main()
