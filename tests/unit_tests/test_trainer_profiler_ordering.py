# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

import contextlib
import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

sys.modules.setdefault(
    "spmd_types",
    types.SimpleNamespace(no_typecheck=contextlib.nullcontext),
)

try:
    from torchtitan.trainer import Trainer
except ImportError as error:
    Trainer = None
    _TRAINER_IMPORT_ERROR = error
else:
    _TRAINER_IMPORT_ERROR = None


class _FakeProfiler:
    def __init__(self, events):
        self.events = events
        self.memory_profiler_started = False
        self.start_memory_profiler_call_count = 0

    def start_memory_profiler(self):
        self.start_memory_profiler_call_count += 1
        if not self.memory_profiler_started:
            self.events.append("start_memory_profiler")
            self.memory_profiler_started = True

    def set_memory_profiler_step(self, global_step):
        self.events.append(("set_memory_profiler_step", global_step))

    def dump_memory_snapshot(self, name):
        self.events.append(("dump_memory_snapshot", name))

    def active(self, *, global_step=0, base_folder="", leaf_folder=""):
        self.events.append(("active", global_step, base_folder, leaf_folder))
        return self

    def __enter__(self):
        self.events.append("enter_profiler_context")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.events.append("exit_profiler_context")
        return False

    def step(self):
        self.events.append("profiler_step")


class _FakeProfilerConfig:
    def __init__(self, events):
        self.events = events
        self.enable_memory_snapshot = True
        self.profiler = _FakeProfiler(events)

    def build(self, *, global_step=0, base_folder="", leaf_folder=""):
        self.events.append(("build_profiler", global_step, base_folder, leaf_folder))
        return self.profiler


class _FakeCheckpointer:
    def __init__(self, events, trainer, loaded_step, loaded=True):
        self.events = events
        self.trainer = trainer
        self.loaded_step = loaded_step
        self.loaded = loaded

    def load(self, *, step):
        self.events.append(("checkpoint_load", step))
        self.trainer.step = self.loaded_step
        return self.loaded


@unittest.skipIf(
    Trainer is None, f"Trainer import unavailable: {_TRAINER_IMPORT_ERROR}"
)
class TestTrainerProfilerOrdering(unittest.TestCase):
    def _trainer(self, *, checkpoint_loaded=True):
        events = []
        trainer = object.__new__(Trainer)
        trainer.step = 0
        trainer.config = SimpleNamespace(
            checkpoint=SimpleNamespace(load_step=-1),
            dump_folder="/tmp/torchtitan-test-output",
            profiler=_FakeProfilerConfig(events),
        )
        trainer.checkpointer = _FakeCheckpointer(
            events, trainer, loaded_step=123, loaded=checkpoint_loaded
        )
        trainer.profiler = trainer.config.profiler.build(
            global_step=0,
            base_folder="/tmp/torchtitan-test-output",
        )
        trainer.profiler.start_memory_profiler()
        trainer.dataloader = object()
        trainer.batch_generator = mock.Mock(return_value=iter(()))
        trainer.should_continue_training = mock.Mock(return_value=False)
        return trainer, events

    @mock.patch("torch.distributed.get_rank", return_value=1)
    @mock.patch("torchtitan.trainer.sl.log_trace_instant")
    def test_memory_history_starts_before_checkpoint_load(self, *_):
        trainer, events = self._trainer(checkpoint_loaded=True)

        Trainer.train(trainer)

        self.assertEqual(trainer.profiler.start_memory_profiler_call_count, 1)
        self.assertEqual(
            events,
            [
                ("build_profiler", 0, "/tmp/torchtitan-test-output", ""),
                "start_memory_profiler",
                ("active", 0, "/tmp/torchtitan-test-output", ""),
                ("checkpoint_load", -1),
                ("set_memory_profiler_step", 123),
                ("active", 123, "/tmp/torchtitan-test-output", ""),
                "enter_profiler_context",
                "exit_profiler_context",
            ],
        )

    @mock.patch("torch.distributed.get_rank", return_value=1)
    @mock.patch("torchtitan.trainer.sl.log_trace_instant")
    def test_train_uses_existing_profiler_when_checkpoint_not_loaded(self, *_):
        trainer, events = self._trainer(checkpoint_loaded=False)

        Trainer.train(trainer)

        self.assertEqual(
            [event for event in events if event[0] == "build_profiler"],
            [("build_profiler", 0, "/tmp/torchtitan-test-output", "")],
        )


if __name__ == "__main__":
    unittest.main()
