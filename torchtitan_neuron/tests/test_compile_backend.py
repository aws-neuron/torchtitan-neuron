# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

from typing import Any, cast
from unittest import mock

import torch

from torchtitan_neuron.compile_backend import install_neuron_compile_backend


def test_install_neuron_compile_backend_routes_only_backendless_calls(monkeypatch):
    original_compile = mock.Mock(return_value=object())
    original_compile._neuron_wrapped = False
    monkeypatch.setattr(torch, "compile", original_compile)

    install_neuron_compile_backend()

    def model(value):
        return value

    options = {"example": True}
    wrapped_compile = cast(Any, torch.compile)
    wrapped_compile(model, options=options)
    original_compile.assert_called_once_with(
        model,
        options=options,
        backend="neuron",
    )

    original_compile.reset_mock()
    wrapped_compile(model, backend="custom")
    original_compile.assert_called_once_with(model, backend="custom")


def test_install_neuron_compile_backend_is_idempotent(monkeypatch):
    original_compile = mock.Mock(return_value=object())
    original_compile._neuron_wrapped = False
    monkeypatch.setattr(torch, "compile", original_compile)

    install_neuron_compile_backend()
    wrapped_compile = torch.compile
    install_neuron_compile_backend()

    assert torch.compile is wrapped_compile
