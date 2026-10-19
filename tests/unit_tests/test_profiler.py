# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Modifications Copyright (c) Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the BSD-3-Clause license.

import enum
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

import torch

from torchtitan.tools.profiler import Profiler


class TestProfilerConfig(unittest.TestCase):
    def test_default_field_values(self):
        cfg = Profiler.Config()
        self.assertFalse(cfg.enable_profiling)
        self.assertEqual(cfg.save_traces_folder, "profiling/traces")
        self.assertEqual(cfg.profile_freq, 10)
        self.assertEqual(cfg.profiler_active, 1)
        self.assertEqual(cfg.profiler_warmup, 3)
        self.assertIsNone(cfg.profiler_repeat)
        self.assertIsNone(cfg.profiler_skip_first)
        self.assertIsNone(cfg.profiler_skip_first_wait)
        self.assertFalse(cfg.enable_memory_snapshot)
        self.assertEqual(cfg.save_memory_snapshot_folder, "profiling/memory_snapshot")

    def test_custom_field_values(self):
        cfg = Profiler.Config(
            enable_profiling=True,
            save_traces_folder="my_traces",
            profile_freq=50,
            profiler_repeat=2,
            profiler_skip_first=5,
            profiler_skip_first_wait=3,
        )
        self.assertTrue(cfg.enable_profiling)
        self.assertEqual(cfg.save_traces_folder, "my_traces")
        self.assertEqual(cfg.profile_freq, 50)
        self.assertEqual(cfg.profiler_repeat, 2)
        self.assertEqual(cfg.profiler_skip_first, 5)
        self.assertEqual(cfg.profiler_skip_first_wait, 3)

    def test_build_returns_profiler_instance(self):
        """Profiler.Config.build() auto-wires to Profiler via Configurable."""
        cfg = Profiler.Config()
        profiler = cfg.build()
        self.assertIsInstance(profiler, Profiler)


class TestProfilerInit(unittest.TestCase):
    def test_default_runtime_attrs(self):
        """Profiler initializes runtime attrs to safe defaults."""
        profiler = Profiler(Profiler.Config())
        self.assertEqual(profiler._global_step, 0)
        self.assertEqual(profiler._base_folder, "")
        self.assertEqual(profiler._leaf_folder, "")
        self.assertIsNone(profiler.torch_profiler)
        self.assertIsNone(profiler.memory_profiler)


class TestProfilerDisabledPaths(unittest.TestCase):
    """Tests for the no-op / disabled paths that require no GPU."""

    def test_build_torch_profiler_disabled_returns_none(self):
        """build_torch_profiler returns None when profiling is disabled."""
        profiler = Profiler(Profiler.Config(enable_profiling=False))
        result = profiler.build_torch_profiler(
            global_step=0, base_folder="/tmp", leaf_folder=""
        )
        self.assertIsNone(result)

    def test_build_memory_profiler_disabled_returns_none(self):
        """build_memory_profiler returns None when memory snapshot is disabled."""
        profiler = Profiler(Profiler.Config(enable_memory_snapshot=False))
        result = profiler.build_memory_profiler(
            global_step=0, base_folder="/tmp", leaf_folder=""
        )
        self.assertIsNone(result)

    def test_runtime_args_stored_on_init(self):
        """Runtime kwargs passed to __init__ are stored on the instance."""
        profiler = Profiler(
            Profiler.Config(), global_step=42, base_folder="/data", leaf_folder="sub"
        )
        self.assertEqual(profiler._global_step, 42)
        self.assertEqual(profiler._base_folder, "/data")
        self.assertEqual(profiler._leaf_folder, "sub")

    def test_context_manager_step_is_noop(self):
        """With everything disabled, context manager and step() don't raise."""
        profiler = Profiler(Profiler.Config())
        with profiler as prof:
            self.assertIs(prof, profiler)
            self.assertIsNone(prof.torch_profiler)
            self.assertIsNone(prof.memory_profiler)
            prof.step()
            prof.step()

    def test_default_args_context_manager(self):
        """Profiler with default runtime args works as a context manager."""
        profiler = Profiler(Profiler.Config())
        with profiler as prof:
            prof.step()

    def test_step_noop_when_both_profilers_none(self):
        """step() is a no-op when torch_profiler and memory_profiler are both None."""
        profiler = Profiler(Profiler.Config())
        profiler.step()
        profiler.step()

    def test_exit_resets_profiler_attrs(self):
        """After __exit__, torch_profiler and memory_profiler are reset to None."""
        profiler = Profiler(Profiler.Config())
        with profiler:
            pass
        self.assertIsNone(profiler.torch_profiler)
        self.assertIsNone(profiler.memory_profiler)

    def test_active_updates_runtime_args(self):
        """active() updates runtime args and returns self for context manager use."""
        profiler = Profiler(Profiler.Config())
        self.assertEqual(profiler._global_step, 0)
        self.assertEqual(profiler._base_folder, "")
        self.assertEqual(profiler._leaf_folder, "")

        result = profiler.active(
            global_step=10, base_folder="/output", leaf_folder="replica_0"
        )
        self.assertIs(result, profiler)
        self.assertEqual(profiler._global_step, 10)
        self.assertEqual(profiler._base_folder, "/output")
        self.assertEqual(profiler._leaf_folder, "replica_0")

    def test_active_as_context_manager(self):
        """active() can be used as a context manager with 'with' statement."""
        profiler = Profiler(Profiler.Config())
        with profiler.active(global_step=5, base_folder="/tmp") as prof:
            self.assertIs(prof, profiler)
            self.assertEqual(prof._global_step, 5)
            prof.step()


class TestProfilerEnabledPaths(unittest.TestCase):
    """Tests for enabled profiler paths — uses mocked distributed rank."""

    def setUp(self):
        self.patcher_rank = mock.patch("torch.distributed.get_rank", return_value=0)
        self.patcher_rank.start()

    def tearDown(self):
        self.patcher_rank.stop()

    def test_build_torch_profiler_returns_active_handle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            profiler = Profiler(
                Profiler.Config(
                    enable_profiling=True,
                    profile_freq=4,
                    profiler_warmup=1,
                    profiler_active=1,
                ),
                global_step=0,
                base_folder=tmpdir,
            )
            with profiler:
                self.assertIsNotNone(profiler.torch_profiler)

    @mock.patch("torchtitan.tools.profiler.device_module.memory._record_memory_history")
    def test_context_reuses_started_memory_profiler(self, mock_record_history):
        with tempfile.TemporaryDirectory() as tmpdir:
            profiler = Profiler(
                Profiler.Config(enable_memory_snapshot=True),
                global_step=0,
                base_folder=tmpdir,
            )
            memory_profiler = profiler.start_memory_profiler()
            with profiler as active_profiler:
                self.assertIs(active_profiler.memory_profiler, memory_profiler)

        mock_record_history.assert_called_once()


class _FakeProfileMode(enum.Enum):
    RUNTIME = 1
    CPU_UTIL = 2
    HOST_MEMORY = 3
    DEVICE = 4


class _FakeNeuronConfig:
    __slots__ = ("kwargs",)

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeNeuronProfiler:
    def __init__(self, config):
        self.config = config

    def export_trace(self, prof):  # matches torch_neuronx NeuronProfiler API
        pass


def _fake_torch_neuronx_profiling():
    mod = types.ModuleType("torch_neuronx.profiling")
    mod.NeuronConfig = _FakeNeuronConfig
    mod.NeuronProfiler = _FakeNeuronProfiler
    mod.ProfileMode = _FakeProfileMode
    return mod


class TestNeuronProfiling(unittest.TestCase):
    """Tests for the Neuron branch of build_torch_profiler.

    _enable_neuron_profiling is host-independent (it never touches real Neuron
    hardware -- it only wires torch_neuronx.profiling into the torch profiler), so
    these run on any CPU host with torch_neuronx.profiling mocked.
    """

    def _call(self, env):
        profiler = Profiler(Profiler.Config(enable_profiling=True))
        activities = [torch.profiler.ProfilerActivity.CPU]
        profile_kwargs: dict = {}
        with mock.patch.dict(
            sys.modules, {"torch_neuronx.profiling": _fake_torch_neuronx_profiling()}
        ), mock.patch.dict(os.environ, env, clear=False):
            os.environ.pop("NEURON_PROFILE_DEVICE", None)
            os.environ.pop("NEURON_PROFILE_NEFF_CACHE_DIR", None)
            os.environ.pop("NEURON_PROFILE_CAPTURE_NC", None)
            os.environ.update(env)
            handler = profiler._enable_neuron_profiling(
                activities, profile_kwargs, "/tmp/traces"
            )
        return handler, activities, profile_kwargs

    def test_system_modes_default(self):
        """Without NEURON_PROFILE_DEVICE, only system + host modes are enabled."""
        handler, activities, kwargs = self._call(env={})
        self.assertIsNotNone(handler)
        self.assertIn(torch.profiler.ProfilerActivity.PrivateUse1, activities)
        self.assertIn("experimental_config", kwargs)
        self.assertTrue(kwargs["with_stack"])
        cfg = kwargs["experimental_config"]
        self.assertEqual(
            {m.name for m in cfg.kwargs["modes"]},
            {"RUNTIME", "CPU_UTIL", "HOST_MEMORY"},
        )
        self.assertEqual(cfg.kwargs["profile_output_dir"], "/tmp/traces")
        self.assertEqual(cfg.kwargs["max_events_per_nc"], 10_000_000)
        # Defaults to torch_neuronx's compile cache so DEVICE .ntff resolves NEFFs.
        self.assertEqual(cfg.kwargs["neff_cache_dir"], "/tmp/neff_cache")
        self.assertEqual(cfg.kwargs["capture_enabled_for_nc"], "0,1")

    def test_device_mode_enabled_by_env(self):
        """NEURON_PROFILE_DEVICE=1 adds the heavy DEVICE mode."""
        _, _, kwargs = self._call(env={"NEURON_PROFILE_DEVICE": "1"})
        cfg = kwargs["experimental_config"]
        self.assertIn("DEVICE", {m.name for m in cfg.kwargs["modes"]})

    def test_neff_cache_dir_env_override(self):
        """NEURON_PROFILE_NEFF_CACHE_DIR overrides the default neff cache dir."""
        _, _, kwargs = self._call(
            env={"NEURON_PROFILE_NEFF_CACHE_DIR": "/custom/neffs"}
        )
        cfg = kwargs["experimental_config"]
        self.assertEqual(cfg.kwargs["neff_cache_dir"], "/custom/neffs")

    def test_capture_nc_env_override(self):
        """NEURON_PROFILE_CAPTURE_NC overrides which NeuronCores are captured."""
        _, _, kwargs = self._call(env={"NEURON_PROFILE_CAPTURE_NC": "0-63"})
        cfg = kwargs["experimental_config"]
        self.assertEqual(cfg.kwargs["capture_enabled_for_nc"], "0-63")

    def test_import_failure_falls_back_to_cpu_only(self):
        """When torch_neuronx.profiling is unavailable, returns None untouched."""
        profiler = Profiler(Profiler.Config(enable_profiling=True))
        activities = [torch.profiler.ProfilerActivity.CPU]
        profile_kwargs: dict = {}
        with mock.patch.dict(sys.modules, {"torch_neuronx.profiling": None}):
            handler = profiler._enable_neuron_profiling(
                activities, profile_kwargs, "/tmp/traces"
            )
        self.assertIsNone(handler)
        self.assertEqual(activities, [torch.profiler.ProfilerActivity.CPU])
        self.assertEqual(profile_kwargs, {})


if __name__ == "__main__":
    unittest.main()
