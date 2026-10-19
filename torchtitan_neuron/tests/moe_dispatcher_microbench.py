#!/usr/bin/env python3
# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

"""Dispatcher-agnostic 64-rank MoE token-dispatch microbenchmark.

Runs the full MoE expert path in isolation -- NO model, NO router network:

    dispatch(tokens) -> grouped-mm (SwiGLU expert compute) -> combine(outputs)

against ANY Neuron-supported token dispatcher, selected by --backend:

    * standard   -> AllToAllTokenDispatcher   (c10d all_to_all_single, NKI a2a-v under
                                               NEURON_USE_NKI_ALLTOALLV=1)
    * neuron_ep  -> NeuronEPTokenDispatcher    (NKI permute_a2av / unpermute_a2av)
    * local      -> LocalTokenDispatcher       (EP=1 baseline, no collective)

All three implement the SAME dispatch()/combine() contract from
torchtitan/models/common/token_dispatcher.py, so the harness body is identical
across backends -- the only difference is the factory line. The grouped-mm in the
middle mirrors GroupedExperts._experts_forward in moe.py (silu(x@w1) * (x@w3) then
@w2, with offs = cumsum(per-expert counts)), including the Neuron 128-alignment
padding that the standard backend needs (neuron_ep pre-pads in its dispatcher).

It measures the FULL expert block as a single timed region --

    dispatch(router-reorder + permute_a2av) -> blockwise experts -> combine(unpermute_a2av)

-- with NO intra-block host barriers: one torch.neuron.synchronize() at the end of the
block forces the whole device pipeline to land before the timer is stamped. This is a
true full-block latency (device dispatch + collectives + compute overlap intact), unlike
per-phase timing which needs a D2H sync after every phase and serializes host/device.
Reports mean / p50 / p90 per rank. A correctness checksum is taken on the LAST iter only,
AFTER the timer stops, so its D2H copy never pollutes timing or a --profile capture.

Launch (64 ranks == 64 NeuronCores):
    torchrun --nproc_per_node=64 moe_dispatcher_microbench.py --backend neuron_ep
    torchrun --nproc_per_node=64 moe_dispatcher_microbench.py --backend standard
    torchrun --nproc_per_node=64 moe_dispatcher_microbench.py --backend local --ep 1

Env (set by pod_run_moe_dispatcher_microbench.sh):
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

import torch
import torch.distributed as dist
import torch.nn.functional as F

# Registers the "neuron" device + distributed backend + the mesh_registry hook
# that neuron_ep's build_ep_replica_group_ranks reads.
import torch_neuronx  # noqa: F401
from torch.distributed.device_mesh import init_device_mesh


# ---- Model-shaped defaults (Qwen3-235B-A22B, matches the a2av microbenches) ----
H = 4096  # hidden dim (D)
FFN = 1536  # per-expert intermediate dim (F); Qwen3-235B moe_inter_dim
TOP_K = 8
NUM_EXPERTS = 64  # reduced from the 235B recipe's 128 (at EP=64 -> num_local_experts=1)
DEFAULT_SEQ_LEN = 512  # T tokens / rank
DEFAULT_EP = 64  # packed neuron_ep requires EP >= 8 and one local expert


def log(rank, msg):
    print(f"[rank {rank}] {msg}", flush=True)
    sys.stdout.flush()


def make_dispatcher(backend: str):
    """Return an unbuilt dispatcher for the requested Neuron-supported backend.

    Uses the SAME factory the model uses (config_utils.make_token_dispatcher_config),
    so this stays in lockstep with whatever backends torchtitan supports. ``local``
    is an alias for ``standard`` at EP=1 -- the AllToAllTokenDispatcher falls back to
    LocalTokenDispatcher when ``ep_mesh is None`` (see AllToAllTokenDispatcher.dispatch),
    so passing ``standard`` with ep=1 exercises the exact local-dispatch code path.
    """
    # neuron_ep packed A2AV is a first-class backend in the torchtitan factory,
    # so it flows through the generic path below like every other backend.
    from torchtitan.models.common.config_utils import make_token_dispatcher_config

    factory_backend = "standard" if backend == "local" else backend
    cfg = make_token_dispatcher_config(
        num_experts=NUM_EXPERTS, top_k=TOP_K, comm_backend=factory_backend
    )
    return cfg.build()


def build_router(rank: int, seq_len: int, imbalance: str, device):
    """Synthesize a router decision (no gate network): per-token top_k expert ids.

    imbalance:
        uniform  -> every rank's tokens spread evenly over all experts
        skewed   -> per-rank random routing (data-dependent counts, mimics a real
                    router's load imbalance across EP ranks)
    Returns (topk_scores_TK float32, topk_expert_ids_TK int32,
             num_local_tokens_per_expert_E int32) on ``device``.
    """
    g = torch.Generator().manual_seed(3000 + rank)
    if imbalance == "uniform":
        # Round-robin: exact balance across experts.
        ids_TK = (
            torch.arange(seq_len * TOP_K, dtype=torch.int64) % NUM_EXPERTS
        ).reshape(seq_len, TOP_K)
    else:
        # Each token picks top_k DISTINCT experts; per-rank seed -> imbalance.
        ids_TK = torch.stack(
            [torch.randperm(NUM_EXPERTS, generator=g)[:TOP_K] for _ in range(seq_len)]
        )
    ids_TK = ids_TK.to(torch.int32)
    scores_TK = torch.rand(seq_len, TOP_K, generator=g, dtype=torch.float32)
    counts_E = torch.bincount(
        ids_TK.reshape(-1).to(torch.int64), minlength=NUM_EXPERTS
    ).to(torch.int32)
    return (
        scores_TK.to(device),
        ids_TK.to(device),
        counts_E.to(device),
    )


def make_expert_weights(num_local_experts: int, device):
    """Random per-expert SwiGLU weights, matching GroupedExperts param shapes."""
    g = torch.Generator().manual_seed(42)
    scale = 1.0 / (H**0.5)
    w1_EFD = (torch.randn(num_local_experts, FFN, H, generator=g) * scale).to(device)
    w3_EFD = (torch.randn(num_local_experts, FFN, H, generator=g) * scale).to(device)
    w2_EDF = (torch.randn(num_local_experts, H, FFN, generator=g) * scale).to(device)
    return w1_EFD, w3_EFD, w2_EDF


def gmm_align_value() -> int:
    from torchtitan.distributed.neuron_ep import gmm_align

    return gmm_align()


def pad_experts_for_neuron(x_RD, counts_e, align):
    """Pad each expert's token block to an ``align`` multiple for this benchmark.

    grouped_mm on Neuron requires the token dim and every offs boundary to be
    align-divisible. neuron_ep already pre-pads (counts come back aligned, so this
    is a no-op); the standard/local backends return raw counts and need it here.

    Returns (x_padded, padded_counts_e, gather_R) where gather_R selects the R real
    rows back out of the padded output (None when no padding was needed).
    """
    real_counts_e = counts_e.to(torch.int64)
    padded_counts_e = torch.ceil(real_counts_e / align).to(torch.int64) * align
    if bool((padded_counts_e == real_counts_e).all()):
        return x_RD, counts_e, None

    device = x_RD.device
    num_experts = real_counts_e.shape[0]
    r_pad = int(padded_counts_e.sum())
    real_starts_e = real_counts_e.cumsum(0) - real_counts_e
    padded_starts_e = padded_counts_e.cumsum(0) - padded_counts_e
    expert_of_row_R = torch.arange(num_experts, device=device).repeat_interleave(
        real_counts_e
    )
    padded_positions_R = (
        padded_starts_e[expert_of_row_R]
        + torch.arange(expert_of_row_R.shape[0], device=device)
        - real_starts_e[expert_of_row_R]
    )
    x_padded = x_RD.new_zeros((r_pad, x_RD.shape[-1]))
    x_padded[padded_positions_R] = x_RD
    return x_padded, padded_counts_e.to(torch.int32), padded_positions_R


def repack_weights_for_moe_cte(weights):
    """Repack grouped-mm expert weights into moe_cte's layout.

    make_expert_weights gives w1_EFD/w3_EFD [E,FFN,H] and w2_EDF [E,H,FFN] (the
    grouped-mm convention: weight applied as x @ w.transpose(-2,-1)). moe_cte wants:
      gate_up_proj_weight [E, H, 2, I_TP]  (h @ gate_up[e] -> [.,2*I_TP]; 0=gate 1=up)
      down_proj_weight    [E, I_TP, H]     (inter @ down[e] -> [.,H])
    so gate/up = w1/w3 transposed to [H,FFN]; down = w2 transposed to [FFN,H].
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

    Replaces the grouped-mm ``experts_forward`` for neuron_ep: ``x_RD`` is
    ``[num_local_experts * rows_per_expert, H]`` with each expert's rows contiguous
    (rows_per_expert = EP*cap). Since rows are already expert-grouped and R is a
    block_size multiple, block metadata is IDENTITY: block b -> expert (b*B)//rpe,
    token_position_to_id = arange(R). Affinity = 1 (routing scores are applied later in
    combine, matching the grouped-mm path). Runs moe_cte (tiles internally by block_size),
    so it COMPILES where the single grouped-mm over R rows overflowed the 5M limit.
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
    # affinity[row, expert_of_row] = 1; padding row 0. Flattened [(R+1)*e, 1].
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


def experts_forward(x_RD, counts_e, weights, align):
    """SwiGLU grouped-mm expert compute -- mirror of GroupedExperts._experts_forward.

    dispatch gives expert-major rows + per-local-expert counts; run the 3 grouped
    matmuls with offs = cumsum(counts), then drop any Neuron alignment padding so
    the (R, D) -> (R, D) contract combine expects is preserved.
    """
    w1_EFD, w3_EFD, w2_EDF = weights
    x_RD, counts_e, gather_R = pad_experts_for_neuron(x_RD, counts_e, align)
    offsets_E = torch.cumsum(counts_e, dim=0, dtype=torch.int32)

    h_RF = F.silu(
        torch._grouped_mm(
            x_RD.bfloat16(), w1_EFD.bfloat16().transpose(-2, -1), offs=offsets_E
        )
    )
    h_RF = h_RF * torch._grouped_mm(
        x_RD.bfloat16(), w3_EFD.bfloat16().transpose(-2, -1), offs=offsets_E
    )
    out_RD = torch._grouped_mm(
        h_RF, w2_EDF.bfloat16().transpose(-2, -1), offs=offsets_E
    ).type_as(x_RD)

    if gather_R is not None:
        out_RD = out_RD[gather_R]
    return out_RD


def summarize(name, samples_s, rank):
    """Log mean / p50 / p90 for a phase (samples in seconds -> ms)."""
    ms = sorted(s * 1e3 for s in samples_s)
    mean = statistics.mean(ms)
    p50 = ms[len(ms) // 2]
    p90 = ms[min(len(ms) - 1, int(0.9 * len(ms)))]
    log(
        rank,
        "  {name:<12} mean={mean:8.2f}ms  p50={p50:8.2f}ms  "
        "p90={p90:8.2f}ms  (n={count})".format(
            name=name,
            mean=mean,
            p50=p50,
            p90=p90,
            count=len(ms),
        ),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--backend", choices=["standard", "neuron_ep", "local"], default="neuron_ep"
    )
    ap.add_argument(
        "--ep", type=int, default=DEFAULT_EP, help="expert-parallel group size"
    )
    ap.add_argument(
        "--seq-len", type=int, default=DEFAULT_SEQ_LEN, help="T tokens per rank"
    )
    ap.add_argument("--imbalance", choices=["uniform", "skewed"], default="skewed")
    ap.add_argument(
        "--experts",
        choices=["grouped_mm", "moe_cte"],
        default="grouped_mm",
        help="expert compute: grouped-mm (default) or blockwise moe_cte NKI kernel",
    )
    ap.add_argument("--block-size", type=int, default=512, help="moe_cte block size")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument(
        "--profile",
        action="store_true",
        help="capture Neuron DEVICE (.ntff hardware) + RUNTIME (system) "
        "traces after timing, for Neuron Explorer",
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
    ep = 1 if args.backend == "local" else args.ep
    dist.init_process_group(backend="neuron")

    assert world_size % ep == 0, f"world_size {world_size} not divisible by ep {ep}"
    assert NUM_EXPERTS % ep == 0, f"num_experts {NUM_EXPERTS} not divisible by ep {ep}"
    num_local_experts = NUM_EXPERTS // ep
    device = "neuron"

    dispatcher = make_dispatcher(args.backend)

    # Build a 2D (efsdp, ep) mesh; ep is the innermost axis so replica groups are
    # contiguous, exactly matching parallel_dims sparse-mesh ordering. Wire ep_mesh
    # into the dispatcher (None for local/EP=1 -> local-dispatch fallback).
    ep_mesh = None
    if ep > 1:
        mesh = init_device_mesh(
            "neuron", (world_size // ep, ep), mesh_dim_names=("efsdp", "ep")
        )
        ep_mesh = mesh["ep"]
    dispatcher.wire_meshes(ep_mesh=ep_mesh, tp_mesh=None)

    if rank == 0:
        log(
            rank,
            f"backend={args.backend} world={world_size} ep={ep} "
            f"local_experts={num_local_experts} T={args.seq_len} H={H} FFN={FFN} "
            f"top_k={TOP_K} imbalance={args.imbalance}",
        )
        log(
            rank,
            f"warmup={args.warmup} iters={args.iters} gmm_align={gmm_align_value()}",
        )

    scores_TK, ids_TK, counts_E = build_router(
        rank, args.seq_len, args.imbalance, device
    )
    x_TD = torch.randn(args.seq_len, H, dtype=torch.bfloat16, device=device)
    weights = make_expert_weights(num_local_experts, device)
    align = gmm_align_value()
    T = args.seq_len

    # Expert-compute path: grouped-mm (default) or the blockwise moe_cte NKI kernel.
    # moe_cte tiles internally by block_size, so it compiles where the single grouped-mm
    # over the neuron_ep padded buffer (num_local_experts*EP*cap rows)
    # overflows the neuronx-cc 5M-instruction limit.
    moe_cte_gw = moe_cte_dw = moe_cte_spec = None
    from torch_neuronx.utils import get_logical_neuron_cores

    lnc = int(get_logical_neuron_cores())
    if args.experts == "moe_cte":
        # moe_cte_experts_forward assumes the neuron_ep fixed-capacity layout
        # (each local expert = exactly EP*cap contiguous rows, zero-padded). The
        # standard/local dispatchers return variable cumsum-packed buffers, so
        # rows_per_expert = R // num_local_experts and the block->expert mapping would be
        # WRONG (silently, if the block-size asserts happen to pass). Require the matching
        # backend rather than risk a wrong answer.
        if args.backend != "neuron_ep":
            raise ValueError(
                f"--experts moe_cte requires --backend neuron_ep (got "
                f"{args.backend}). The moe_cte glue assumes the packed "
                "one-local-expert capacity layout."
            )
        import nki_blockwise_moe_cte as _bw

        moe_cte_gw, moe_cte_dw = repack_weights_for_moe_cte(weights)
        moe_cte_spec = _bw.default_spec()
        if rank == 0:
            log(rank, f"experts=moe_cte block_size={args.block_size} lnc={lnc}")

    def one_step(collect):
        """Run the full expert block once: dispatch (router-reorder + permute_a2av) ->
        blockwise experts -> combine (unpermute_a2av).

        NO intra-block host barriers -- the phases queue on the device back-to-back and a
        single torch.neuron.synchronize() at the end forces the whole pipeline to land
        before the timer stops, so device dispatch/collective/compute overlap is preserved.
        The correctness checksum is taken outside the timed region (and only when
        ``collect`` is set, i.e. never during the timing loop's steady state -- see caller)
        so its D2H copy never pollutes timing or a --profile capture.
        """
        t0 = time.time()
        routed_input_RD, counts_e, meta = dispatcher.dispatch(
            x_TD, scores_TK, ids_TK, counts_E
        )

        if args.experts == "moe_cte":
            routed_output_RD = moe_cte_experts_forward(
                routed_input_RD,
                num_local_experts,
                moe_cte_gw,
                moe_cte_dw,
                args.block_size,
                moe_cte_spec,
                lnc,
            )
        else:
            routed_output_RD = experts_forward(
                routed_input_RD, counts_e, weights, align
            )

        out_TD = dispatcher.combine(
            routed_output_RD,
            meta,
            x_TD,
            num_local_tokens_after_padding=T,
            local_seq_len_after_padding=T,
        )
        # Single barrier for the whole block -- lands dispatch + experts + combine.
        torch.neuron.synchronize()
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

    # ---- Timed iters (full expert-block latency) ----
    log(rank, f"=== timing x{args.iters} ===")
    full = []
    last_checksum = None
    for _ in range(args.iters):
        times, last_checksum = one_step(collect=True)
        full.append(times[0])
    dist.barrier()

    log(
        rank,
        "=== RESULTS backend={} ep={} checksum={:.3e} ===".format(
            args.backend, ep, last_checksum
        ),
    )
    summarize("full-block", full, rank)

    # ---- Optional Neuron device + system profile (post-timing, NEFFs already cached) ----
    if args.profile:
        _capture_profile(args, rank, one_step)

    dist.barrier()
    if rank == 0:
        log(rank, f"ALL RANKS COMPLETED SUCCESSFULLY (backend={args.backend})")
    dist.destroy_process_group()


def _capture_profile(args, rank, one_step):
    """Capture DEVICE (.ntff hardware) + RUNTIME (system) traces for Neuron Explorer.

    Uses NeuronProfiler with both ProfileMode.DEVICE and ProfileMode.RUNTIME, driven by
    a torch.profiler.schedule so a couple of steady-state steps are captured (NEFFs are
    already warm from the timing loop above). Each rank writes its own session under
    profile_dir; post-process rank 0 with `neuron-explorer view` into a .pftrace. DEVICE
    mode reserves ~5GB HBM per NC -- fine at these sizes.
    """
    from torch.profiler import profile, ProfilerActivity, schedule
    from torch_neuronx.profiling import NeuronConfig, NeuronProfiler, ProfileMode

    profile_dir = args.profile_dir or os.environ.get("MB_PROFILE_DIR") or "./mb_profile"
    # Per-rank subdir so 64 ranks don't clobber one session dir.
    rank_dir = os.path.join(
        profile_dir,
        f"{args.backend}_ep{args.ep if args.backend != 'local' else 1}",
        f"rank{rank}",
    )
    os.makedirs(rank_dir, exist_ok=True)
    # max_events_per_nc sizes the system-trace ring buffer. The NeuronConfig default
    # (100K events/NC) silently drops the large majority of events on a 64-rank MoE step
    # (~837K events/step measured on a TP4/FSDP16 run -> ~86% dropped at 100K).
    # 10M gives headroom. This is the real anti-drop knob -- the
    # runtime drops SILENTLY (no log message), so oversize is the only prevention.
    # capture_enabled_for_nc="0,1" captures both cores of the LNC=2 group (the guide's
    # host-cc example sets it; default may capture NC 0 only). CPU_UTIL/HOST_MEMORY add
    # the cpu_util.pb / host_mem.pb host-side context traces.
    # DEVICE mode captures the .ntff hardware trace but its notification capture can
    # collide with the in-kernel NKI a2a-v collective coordination (permute_a2av does
    # cross-rank all_to_all_v INSIDE the kernel) -> ranks desync -> NRT status 1200
    # "Stuck in collectives op (missing notification(s))". Set MB_PROFILE_RUNTIME_ONLY=1
    # to drop DEVICE mode and capture only the system (ntrace.pb) + host traces, which do
    # not perturb the collective. RUNTIME still shows dispatch/collective/combine timing.
    _runtime_only = os.environ.get("MB_PROFILE_RUNTIME_ONLY") == "1"
    _modes = [ProfileMode.RUNTIME, ProfileMode.CPU_UTIL, ProfileMode.HOST_MEMORY]
    if not _runtime_only:
        _modes = [ProfileMode.DEVICE] + _modes
    neuron_config = NeuronConfig(
        modes=_modes,
        profile_output_dir=rank_dir,
        max_events_per_nc=10_000_000,
        capture_enabled_for_nc="0,1",
    )
    exporter = NeuronProfiler(neuron_config)

    if rank == 0:
        _tag = "RUNTIME-only" if _runtime_only else "DEVICE+RUNTIME"
        log(
            rank,
            f"=== PROFILING ({_tag}) active={args.profile_active} dir={profile_dir} ===",
        )
    dist.barrier()
    # wait=1, warmup=1 to settle, then capture `active` steady-state steps.
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
    # Also drop a per-rank framework chrome trace next to the session.
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
