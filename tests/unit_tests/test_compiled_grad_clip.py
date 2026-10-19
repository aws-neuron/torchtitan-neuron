# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

from unittest import mock

import torch

from torchtitan.config import CompileConfig
from torchtitan.distributed import utils as dist_utils
from torchtitan.tools import logging as logging_utils


def test_compiled_grad_clip_uses_configured_backend(monkeypatch):
    compile_calls = []

    def compile_fn(fn, **kwargs):
        compile_calls.append((fn, kwargs))
        return fn

    monkeypatch.setattr(torch, "compile", compile_fn)
    monkeypatch.setattr(dist_utils.dist, "all_reduce", mock.Mock())

    parameter = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    parameter.grad = torch.tensor([3.0, 4.0])
    clipper = dist_utils.CompiledGradClip(backend="aot_eager")

    total_norm = dist_utils.clip_grad_norm_(
        [parameter],
        max_norm=1.0,
        compiled_grad_clip=clipper,
    )

    torch.testing.assert_close(total_norm, torch.tensor(5.0))
    torch.testing.assert_close(parameter.grad, torch.tensor([0.6, 0.8]))
    assert compile_calls == [
        (
            dist_utils._compute_grouped_norm_sq,
            {"backend": "aot_eager", "dynamic": False},
        ),
        (
            dist_utils._compute_clip_coef,
            {"backend": "aot_eager", "dynamic": False},
        ),
    ]


def test_compiled_grad_clip_is_opt_in():
    parameter = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    parameter.grad = torch.tensor([3.0, 4.0])
    compiled_clip = mock.Mock(return_value=torch.tensor(7.0))

    eager_total_norm = dist_utils.clip_grad_norm_([parameter], max_norm=1.0)

    torch.testing.assert_close(eager_total_norm, torch.tensor(5.0))
    compiled_clip.assert_not_called()

    parameter.grad = torch.tensor([3.0, 4.0])
    compiled_total_norm = dist_utils.clip_grad_norm_(
        [parameter],
        max_norm=1.0,
        compiled_grad_clip=compiled_clip,
    )

    torch.testing.assert_close(compiled_total_norm, torch.tensor(7.0))
    compiled_clip.assert_called_once_with([parameter], 1.0)


def test_build_compiled_grad_clip_is_opt_in():
    assert (
        dist_utils.build_compiled_grad_clip(
            CompileConfig(enable=True, backend="aot_eager")
        )
        is None
    )

    clipper = dist_utils.build_compiled_grad_clip(
        CompileConfig(
            enable=True,
            components=["model", "loss", "grad_clip"],
            backend="aot_eager",
        )
    )

    assert isinstance(clipper, dist_utils.CompiledGradClip)
    assert clipper._backend == "aot_eager"


def test_compiled_grad_clip_warns_once_when_falling_back(
    monkeypatch,
    caplog,
):
    monkeypatch.setattr(logging_utils, "_logged", set())
    parameter = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    compiled_clip = mock.Mock()

    with caplog.at_level("WARNING"):
        for _ in range(2):
            parameter.grad = torch.tensor([3.0, 4.0])
            total_norm = dist_utils.clip_grad_norm_(
                [parameter],
                max_norm=1.0,
                norm_type=1.0,
                compiled_grad_clip=compiled_clip,
            )
            torch.testing.assert_close(total_norm, torch.tensor(7.0))

    compiled_clip.assert_not_called()
    warnings = [
        record.message
        for record in caplog.records
        if "Compiled gradient clipping was requested" in record.message
    ]
    assert len(warnings) == 1
    assert "norm_type is 1.0, not 2.0" in warnings[0]
    assert "falling back to eager clipping" in warnings[0]


def test_compiled_grad_clip_without_local_grads_still_reduces(monkeypatch):
    all_reduce = mock.Mock(side_effect=lambda tensor, **_: tensor.fill_(25.0))
    monkeypatch.setattr(dist_utils.dist, "all_reduce", all_reduce)

    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    clipper = dist_utils.CompiledGradClip(backend="aot_eager")

    total_norm = clipper([parameter], max_norm=1.0)

    torch.testing.assert_close(total_norm, torch.tensor(5.0))
    assert total_norm.device == parameter.device
    all_reduce.assert_called_once()
