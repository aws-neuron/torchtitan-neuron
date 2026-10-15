# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Function-scoped torch.compile registration."""

import functools
from collections.abc import Callable
from typing import Any

import torch

from torchtitan.distributed.batch_invariant import is_in_batch_invariant_mode


_BACKEND: str = "inductor"


def apply_local_compile(regions: list[str], *, backend: str = "inductor") -> None:
    """Bind registered functions to eager or compiled implementations.

    Process-wide: a later call replaces these choices for every model in the process.
    """
    unknown = [name for name in regions if name not in _LOCAL_COMPILE_CALLBACKS]
    if unknown:
        raise ValueError(
            f"Unknown local_compile_regions entries {unknown}; "
            f"registered values are {sorted(_LOCAL_COMPILE_CALLBACKS)}"
        )

    global _BACKEND
    _BACKEND = backend
    for callbacks in _LOCAL_COMPILE_CALLBACKS.values():
        for bind_local_compile_fn in callbacks:
            bind_local_compile_fn(regions)
    for bind_compiled_helper in _COMPILED_HELPERS:
        bind_compiled_helper()


_LOCAL_COMPILE_CALLBACKS: dict[str, list[Callable[[list[str]], None]]] = {}
_COMPILED_HELPERS: list[Callable[[], None]] = []


def compiled(
    reference: Callable[..., Any],
    *,
    inductor_options: dict[str, Any] | None = None,
) -> Callable[..., Any]:
    """Return ``reference`` compiled with the configured backend.

    The function binds on first use and rebinds whenever ``apply_local_compile``
    changes the process-wide backend. ``inductor_options`` are passed only when
    that backend is inductor.
    """
    fn: Callable[..., Any] | None = None

    def bind_compiled_helper() -> None:
        nonlocal fn
        compile_kwargs: dict[str, Any] = {"backend": _BACKEND}
        if _BACKEND == "inductor" and inductor_options is not None:
            compile_kwargs["options"] = inductor_options
        fn = torch.compile(reference, **compile_kwargs)

    @functools.wraps(reference)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if fn is None:
            bind_compiled_helper()
        assert fn is not None
        return fn(*args, **kwargs)

    _COMPILED_HELPERS.append(bind_compiled_helper)
    return wrapped


def local_compile(
    name: str,
    *,
    batch_invariant: bool,
    **compile_kwargs: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a function that can be compiled independently.

    Args:
        name: Name used to enable the function in a model's ``local_compile_regions``.
        batch_invariant: Whether the compiled function preserves batch invariance.
        **compile_kwargs: Additional ``torch.compile`` keyword arguments;
            ``fullgraph`` is fixed to ``True`` so each function forms one complete
            compile region.
    """
    if compile_kwargs.pop("fullgraph", True) is not True:
        raise ValueError("local_compile requires fullgraph=True.")
    if "backend" in compile_kwargs:
        raise ValueError("local_compile backend is configured by apply_local_compile.")

    def decorate(reference: Callable[..., Any]) -> Callable[..., Any]:
        fn = reference

        def bind_local_compile(regions: list[str]) -> None:
            nonlocal fn
            enabled = name in regions
            batch_invariant_mode = is_in_batch_invariant_mode()
            if enabled and batch_invariant_mode and not batch_invariant:
                raise ValueError(
                    f"Local compile region {name!r} does not support "
                    "batch-invariant mode; remove it from local_compile_regions."
                )
            if enabled:
                fn = torch.compile(
                    reference,
                    fullgraph=True,
                    backend=_BACKEND,
                    **compile_kwargs,
                )
            else:
                fn = reference

        @functools.wraps(reference)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            return fn(*args, **kwargs)

        _LOCAL_COMPILE_CALLBACKS.setdefault(name, []).append(bind_local_compile)
        return wrapped

    return decorate


__all__ = ["apply_local_compile", "compiled", "local_compile"]
