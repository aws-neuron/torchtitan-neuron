# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Modifications Copyright (c) Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the BSD-3-Clause license.

"""Kineto profiler + memory-snapshot lifecycle."""

import os
import pickle
import time
from dataclasses import dataclass
from typing import Annotated

import torch
import tyro
from torchtitan.config import Configurable
from torchtitan.config.function import Function
from torchtitan.observability import structured_logger as sl
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import device_module, device_type

# Paths expects by meta internal tooling
PROFILE_DIR = "profiling/traces"  # Profiler.Config.save_traces_folder default
PROFILE_ITER_DIR = "iteration_{step}"  # PROFILE_DIR/{PROFILE_ITER_DIR}
PROFILE_FILE = "rank{rank}_trace.json.gz"  # PROFILE_DIR/PROFILE_ITER_DIR/{PROFILE_FILE}

# --- Neuron (torch_neuronx) profiling -----------------------------------------
# On Neuron the torch profiler is driven by torch_neuronx: the Neuron runtime is
# exposed as the PrivateUse1 profiler activity and a NeuronConfig is passed as the
# profiler's experimental_config. RUNTIME/CPU_UTIL/HOST_MEMORY (lightweight system
# + host metrics) are always captured; the heavy DEVICE profile (hardware
# instruction traces) reserves ~5GB HBM, so it is opt-in via an env var.
NEURON_PROFILE_DEVICE_ENV = "NEURON_PROFILE_DEVICE"  # "1" -> add ProfileMode.DEVICE
# NRT system-trace ring buffer size. The 100K-events/NC NeuronConfig default drops
# the vast majority of events for even one active step of a multi-core distributed
# run (~837K observed); 10M gives headroom. Set via NeuronConfig(max_events_per_nc).
NEURON_MAX_EVENTS_PER_NC = 10_000_000
# NeuronCores whose system/device trace is captured. Defaults to "0,1" (each .ntff
# is ~1GB, so capturing every core of a 64-core run is huge and the profile is
# usually analyzed rank-0-first); override with NEURON_PROFILE_CAPTURE_NC, e.g.
# "0,1,2,3" or "0-63".
NEURON_CAPTURE_NC_ENV = "NEURON_PROFILE_CAPTURE_NC"
NEURON_DEFAULT_CAPTURE_ENABLED_FOR_NC = "0,1"
# Global-rank allowlist, defaulting to rank 0. This is distinct from
# capture_enabled_for_nc: with one process per NeuronCore, every process sees its
# assigned core as NC0. Override with a comma-separated list, e.g. "0,1,2,3".
NEURON_PROFILE_RANKS_ENV = "NEURON_PROFILE_RANKS"
NEURON_DEFAULT_PROFILE_RANKS = "0"
# On session stop the runtime copies the loaded NEFFs from this dir into the
# session's neffs/, matched to the DEVICE .ntff by the UUID embedded in each NEFF,
# so neuron-explorer can resolve the device trace (else "unable to get NEFFs by
# UID"). torch_neuronx's default compile cache is /tmp/neff_cache; override with
# NEURON_PROFILE_NEFF_CACHE_DIR if the run's NEFF cache lives elsewhere.
NEURON_NEFF_CACHE_DIR_ENV = "NEURON_PROFILE_NEFF_CACHE_DIR"
NEURON_DEFAULT_NEFF_CACHE_DIR = "/tmp/neff_cache"

MEMORY_DIR = (
    "profiling/memory_snapshot"  # Profiler.Config.save_memory_snapshot_folder default
)
MEMORY_STEP_DIR = "step_{step:012d}"  # MEMORY_DIR/{MEMORY_STEP_DIR}
MEMORY_EXIT_DIR = "step_{step:012d}_exit"  # OOM dump variant
MEMORY_FILE = (
    "{rank:06d}_step_{step}.pickle"  # MEMORY_DIR/MEMORY_STEP_DIR/{MEMORY_FILE}
)


class MemoryProfiler:
    """Records periodic memory snapshots during training.

    Started by :meth:`Profiler.build_memory_profiler` when memory snapshots are
    enabled. Call :meth:`step` once per training iteration to trigger periodic
    dumps; pass ``exit_ctx=True`` to force a final dump on OOM.
    """

    def __init__(
        self,
        step_num: int,
        freq: int,
        snapshot_dir: str,
        leaf_folder: str,
        rank: int,
        max_entries: int,
    ) -> None:
        device_module.memory._record_memory_history(
            # stacks="python" records only Python frames (not C++), which is much
            # cheaper to capture and serialize, keeping the snapshot dump from
            # taking minutes -- the default stacks="all" symbolizes C++ frames,
            # which is very slow for torchtitan workload.
            stacks="python",
            max_entries=max_entries,
        )
        # when resume training, we start from the last step
        self.step_num = step_num
        self.freq = freq
        self._snapshot_dir = snapshot_dir
        self._leaf_folder = leaf_folder
        self._rank = rank

    def set_step_num(self, step_num: int) -> None:
        self.step_num = step_num

    def step(self, exit_ctx: bool = False) -> None:
        self.step_num += 1
        if not exit_ctx and self.step_num % self.freq != 0:
            return
        if not exit_ctx:
            curr_step = self.step_num
            dir_name = MEMORY_STEP_DIR.format(step=curr_step)
        else:
            # dump as step_000000000000_exit if OOM at iter 1
            curr_step = self.step_num - 1
            dir_name = MEMORY_EXIT_DIR.format(step=curr_step)
        curr_snapshot_dir = os.path.join(
            self._snapshot_dir, dir_name, self._leaf_folder
        )
        if not os.path.exists(curr_snapshot_dir):
            os.makedirs(curr_snapshot_dir, exist_ok=True)
        logger.info(f"Dumping memory snapshot at step {curr_step}")
        begin = time.monotonic()
        output_file = os.path.join(
            curr_snapshot_dir, MEMORY_FILE.format(rank=self._rank, step=curr_step)
        )
        with open(output_file, "wb") as output:
            # Protocol 4 for compatibility with pytorch.org/memory_viz JS parser
            pickle.dump(device_module.memory._snapshot(), output, protocol=4)
        logger.info(
            f"Finished dumping memory snapshot in {time.monotonic() - begin:.2f} seconds"
        )


class Profiler(Configurable):
    """Owns profiling and memory snapshot lifecycle for a training run.

    Example::

        with Profiler(config, global_step=step, base_folder=folder) as prof:
            for step in training_loop:
                ...
                prof.step()

    Args:
        config: A ``Profiler.Config`` instance.
        global_step: The training step at which profiling begins.  When
            resuming from a checkpoint this should be the loaded step so that
            trace directories are named correctly (e.g. ``iteration_100``
            instead of ``iteration_0``) and memory-snapshot frequency alignment
            is preserved.
        base_folder: Root directory for profiler trace and memory snapshot output.
        leaf_folder: Optional subdirectory appended to trace/snapshot paths
            (e.g. per-replica folder in fault-tolerant training).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        enable_profiling: bool = False
        """Whether to enable pytorch profiler."""

        save_traces_folder: str = PROFILE_DIR
        """Trace files location."""

        profile_freq: int = 10
        """How often to collect profile traces, in iterations."""

        profiler_repeat: int | None = None
        """
        The number of times to repeat the profiling cycle

        This is used to configure torch.profiler.schedule.
        """

        profiler_skip_first: int | None = None
        """
        The number of initial profiling cycles to skip

        This is used to configure torch.profiler.schedule.
        """

        profiler_skip_first_wait: int | None = None
        """
        The number of initial profiling cycles to skip the wait time

        This is used to configure torch.profiler.schedule.
        """

        profiler_active: int = 1
        """
        The steps profiler is active for.

        This is used to configure torch.profiler.schedule.
        """

        profiler_warmup: int = 3
        """
        The number of warmup steps before the active step in each profiling cycle.

        This is used to configure torch.profiler.schedule.
        """

        enable_memory_snapshot: bool = False
        """Whether to dump memory snapshot."""

        save_memory_snapshot_folder: str = MEMORY_DIR
        """Memory snapshot files location."""

        memory_snapshot_max_entries: int = 1_000_000
        """Max alloc/free events recorded per memory snapshot (ring buffer).

        Caps the history passed to ``_record_memory_history``; the oldest events
        are dropped once full. Bounds host memory and snapshot size / dump time.
        """

        trace_post_processor: Annotated[
            Function.Config | None, tyro.conf.Suppress
        ] = None
        """Optional hook invoked with the trace path after each export.

        Wraps ``fn(trace_path: str) -> None``.
        Set programmatically (not via CLI) — tyro cannot parse Callable types.
        """

    def __init__(
        self,
        config: Config,
        *,
        global_step: int = 0,
        base_folder: str = "",
        leaf_folder: str = "",
    ) -> None:
        self._config = config
        self._global_step = global_step
        self._base_folder = base_folder
        self._leaf_folder = leaf_folder
        self.torch_profiler = None
        self.memory_profiler = None

    def active(
        self,
        *,
        global_step: int = 0,
        base_folder: str = "",
        leaf_folder: str = "",
    ) -> "Profiler":
        """Update runtime args and return self for use as a context manager.

        This allows a pre-built :class:`Profiler` (e.g. built during
        ``__init__``) to be activated later with runtime parameters::

            self.profiler = config.profiler.build()
            ...
            with self.profiler.active(global_step=step, base_folder=folder) as p:
                ...
        """
        self._global_step = global_step
        self._base_folder = base_folder
        self._leaf_folder = leaf_folder
        self.set_memory_profiler_step(global_step)
        return self

    def __enter__(self) -> "Profiler":
        self.torch_profiler = self.build_torch_profiler(
            global_step=self._global_step,
            base_folder=self._base_folder,
            leaf_folder=self._leaf_folder,
        )
        self.start_memory_profiler()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self.torch_profiler is not None:
            self.torch_profiler.__exit__(exc_type, exc_val, exc_tb)
            self.torch_profiler = None
        if self.memory_profiler is not None:
            if isinstance(exc_val, torch.OutOfMemoryError):
                self.memory_profiler.step(exit_ctx=True)
            self.memory_profiler = None
        return False

    def step(self) -> None:
        """Advance all active profilers by one training step."""

        if self.torch_profiler is not None:
            with sl.log_trace_span("kineto_profiler_step_call"):
                self.torch_profiler.step()
        if self.memory_profiler is not None:
            with sl.log_trace_span("memory_profiler_step_call"):
                self.memory_profiler.step()

    def start_memory_profiler(self):
        """Start memory-history recording without starting the torch profiler."""

        if self.memory_profiler is None:
            self.memory_profiler = self.build_memory_profiler(
                global_step=self._global_step,
                base_folder=self._base_folder,
                leaf_folder=self._leaf_folder,
            )
        return self.memory_profiler

    def set_memory_profiler_step(self, global_step: int) -> None:
        if self.memory_profiler is not None:
            self.memory_profiler.set_step_num(global_step)

    def build_torch_profiler(
        self,
        *,
        global_step: int,
        base_folder: str,
        leaf_folder: str,
    ):
        """Create, start, and return the torch profiler, or ``None`` if disabled.

        Calls ``torch.profiler.profile.__enter__()`` so the returned handle is
        already active. :meth:`__exit__` is responsible for stopping it.
        """
        cfg = self._config
        if not cfg.enable_profiling:
            return None

        trace_dir = os.path.join(base_folder, cfg.save_traces_folder)
        profile_freq, warmup, active = (
            cfg.profile_freq,
            cfg.profiler_warmup,
            cfg.profiler_active,
        )

        rank = torch.distributed.get_rank()
        if device_type == "neuron":
            profile_ranks = os.environ.get(
                NEURON_PROFILE_RANKS_ENV, NEURON_DEFAULT_PROFILE_RANKS
            )
            enabled_ranks = {int(value) for value in profile_ranks.split(",")}
            if rank not in enabled_ranks:
                logger.info(
                    "Neuron profiling disabled on rank %d by %s=%s",
                    rank,
                    NEURON_PROFILE_RANKS_ENV,
                    profile_ranks,
                )
                return None
        post_processor = (
            cfg.trace_post_processor.build() if cfg.trace_post_processor else None
        )

        def trace_handler(prof):
            curr_trace_dir_name = PROFILE_ITER_DIR.format(step=prof.step_num)
            curr_trace_dir = os.path.join(trace_dir, curr_trace_dir_name, leaf_folder)
            if not os.path.exists(curr_trace_dir):
                os.makedirs(curr_trace_dir, exist_ok=True)

            logger.info(f"Dumping profiler traces at step {prof.step_num}")
            begin = time.monotonic()

            output_file = os.path.join(curr_trace_dir, PROFILE_FILE.format(rank=rank))
            prof.export_chrome_trace(output_file)

            if post_processor is not None:
                post_processor(output_file)

            logger.info(
                f"Finished dumping profiler traces in {time.monotonic() - begin:.2f} seconds"
            )

        logger.info(f"Profiling active. Traces will be saved at {trace_dir}")

        if not os.path.exists(trace_dir):
            os.makedirs(trace_dir, exist_ok=True)

        additional_params = {
            key: val
            for key, val in [
                ("repeat", cfg.profiler_repeat),
                ("skip_first", cfg.profiler_skip_first),
                ("skip_first_wait", cfg.profiler_skip_first_wait),
            ]
            if val is not None
        }

        wait = profile_freq - (active + warmup)
        assert (
            wait >= 0
        ), "profile_freq must be greater than or equal to warmup + active"
        activities = [torch.profiler.ProfilerActivity.CPU]
        profile_kwargs: dict = {}
        neuron_trace_handler = None
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        elif torch.xpu.is_available():
            activities.append(torch.profiler.ProfilerActivity.XPU)
        elif device_type == "neuron":
            neuron_trace_handler = self._enable_neuron_profiling(
                activities, profile_kwargs, trace_dir
            )
            if neuron_trace_handler is not None and post_processor is not None:
                logger.warning(
                    "trace_post_processor is set but is ignored on Neuron; the "
                    "Neuron profiler uses its own trace exporter."
                )

        # On Neuron the Neuron exporter fully replaces the chrome-trace handler:
        # it writes the native .ntff/.pb session dirs + per-rank framework
        # trace.json instead of the chrome PROFILE_FILE, so the chrome
        # `trace_handler` and its `trace_post_processor` hook do not run there
        # (the warning above surfaces an ignored post_processor). CUDA/CPU/XPU keep
        # the chrome `trace_handler`.
        torch_profiler = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=wait, warmup=warmup, active=active, **additional_params
            ),
            on_trace_ready=neuron_trace_handler or trace_handler,
            record_shapes=True,
            **profile_kwargs,
        )
        torch_profiler.__enter__()
        torch_profiler.step_num = global_step
        return torch_profiler

    def _enable_neuron_profiling(
        self,
        activities: list,
        profile_kwargs: dict,
        trace_dir: str,
    ):
        """Wire torch_neuronx profiling into the torch profiler (in place).

        Appends the Neuron profiler activity to ``activities``, sets the
        ``experimental_config``/``with_stack`` entries in ``profile_kwargs``, and
        returns the ``NeuronProfiler.export_trace`` handler that writes the
        ``.ntff``/``.pb`` session dirs (plus the per-rank framework trace) under
        ``trace_dir``. With ``enable_memory_snapshot``, the handler also writes
        ``memory_snapshot.pickle`` into the same session dir.
        ``RUNTIME``/``CPU_UTIL``/``HOST_MEMORY`` are always enabled;
        ``DEVICE`` (hardware instruction traces, ~5GB HBM) is added only when
        ``NEURON_PROFILE_DEVICE=1``.

        Returns ``None`` (leaving ``activities``/``profile_kwargs`` untouched) if
        ``torch_neuronx`` profiling is unavailable, so the caller falls back to a
        CPU-only trace.
        """
        try:
            from torch_neuronx.profiling import (
                NeuronConfig,
                NeuronProfiler,
                ProfileMode,
            )
        except ImportError:
            logger.warning(
                "torch_neuronx.profiling unavailable; falling back to CPU-only "
                "profiling on Neuron."
            )
            return None

        modes = [ProfileMode.RUNTIME, ProfileMode.CPU_UTIL, ProfileMode.HOST_MEMORY]
        if os.environ.get(NEURON_PROFILE_DEVICE_ENV, "0") == "1":
            modes.append(ProfileMode.DEVICE)

        neff_cache_dir = os.environ.get(
            NEURON_NEFF_CACHE_DIR_ENV, NEURON_DEFAULT_NEFF_CACHE_DIR
        )
        capture_enabled_for_nc = os.environ.get(
            NEURON_CAPTURE_NC_ENV, NEURON_DEFAULT_CAPTURE_ENABLED_FOR_NC
        )

        neuron_config = NeuronConfig(
            modes=modes,
            profile_output_dir=trace_dir,
            max_events_per_nc=NEURON_MAX_EVENTS_PER_NC,
            capture_enabled_for_nc=capture_enabled_for_nc,
            neff_cache_dir=neff_cache_dir,
        )
        activities.append(torch.profiler.ProfilerActivity.PrivateUse1)
        profile_kwargs["experimental_config"] = neuron_config
        profile_kwargs["with_stack"] = True
        logger.info(f"Neuron profiling enabled with modes: {[m.name for m in modes]}")
        neuron_profiler = NeuronProfiler(neuron_config)
        if not self._config.enable_memory_snapshot:
            return neuron_profiler.export_trace

        def export_trace_and_snapshot(prof):
            neuron_profiler.export_trace(prof)
            neuron_profiler.export_snapshot()

        return export_trace_and_snapshot

    def build_memory_profiler(
        self,
        *,
        global_step: int,
        base_folder: str,
        leaf_folder: str,
    ):
        """Create and return a :class:`MemoryProfiler`, or ``None`` if disabled.

        :class:`MemoryProfiler.__init__` starts memory history recording immediately.
        """
        cfg = self._config
        if not cfg.enable_memory_snapshot:
            return None

        snapshot_dir = os.path.join(base_folder, cfg.save_memory_snapshot_folder)
        if not os.path.exists(snapshot_dir):
            os.makedirs(snapshot_dir, exist_ok=True)
        rank = torch.distributed.get_rank()

        logger.info(f"Memory profiler active. Snapshot will be saved at {snapshot_dir}")
        return MemoryProfiler(
            global_step,
            cfg.profile_freq,
            snapshot_dir,
            leaf_folder,
            rank,
            cfg.memory_snapshot_max_entries,
        )
