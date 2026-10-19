# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

import functools
from typing import Any, cast

import torch


def install_neuron_compile_backend() -> None:
    """Route bare ``torch.compile`` calls to the Neuron dynamo backend.

    Several call sites (flex attention, block-mask compilation) call
    ``torch.compile`` with no backend, so they default to inductor, which
    cannot lower to the neuron device ("No interface for device neuron").
    Rather than thread a backend through every site, wrap ``torch.compile``
    once: on the neuron device, and only when the caller did not pass a
    ``backend``, use the registered "neuron" backend. Any ``options`` the caller
    passed are left intact -- the neuron backend accepts them, and the flex call
    sites pass their inductor autotune configs there. Calls that pass an explicit
    ``backend`` (e.g. via ``config.compile.backend``) are untouched.

    The caller is responsible for invoking this only on the Neuron device.
    """
    if getattr(torch.compile, "_neuron_wrapped", False):
        return

    _original_compile = torch.compile

    @functools.wraps(_original_compile)
    def _compile(model=None, **kwargs):
        if "backend" not in kwargs:
            kwargs["backend"] = "neuron"
        return _original_compile(model, **kwargs)

    compile_wrapper = cast(Any, _compile)
    compile_wrapper._neuron_wrapped = True
    torch.compile = compile_wrapper
