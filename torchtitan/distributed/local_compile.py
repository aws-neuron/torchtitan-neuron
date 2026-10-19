# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Function-scoped torch.compile registration and configuration."""

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from torchtitan.distributed.batch_invariant import is_in_batch_invariant_mode


_COMPILED_HELPER_RESETS: list[Callable[[], None]] = []
_LOCAL_COMPILE_CALLBACKS: dict[
    str, list[Callable[["LocalCompileConfig", list[str]], None]]
] = {}


def _reset_compiled_helpers() -> None:
    for reset in _COMPILED_HELPER_RESETS:
        reset()


@dataclass(kw_only=True, slots=True)
class LocalCompileConfig:
    """Policy for TorchTitan-owned local ``torch.compile`` boundaries."""

    regions: list[str] | None = None
    """Optional region override.

    ``None`` preserves the model's existing ``local_compile_regions`` value.
    """

    def compile(
        self,
        reference: Callable[..., Any],
        *,
        inductor_options: dict[str, Any] | None = None,
        **compile_kwargs: Any,
    ) -> Callable[..., Any]:
        """Compile one TorchTitan-owned region.

        Subclasses can override this method to select an out-of-tree backend.
        ``inductor_options`` are forwarded only when the effective backend is
        Inductor.
        """
        backend = compile_kwargs.setdefault("backend", "inductor")
        if backend == "inductor" and inductor_options is not None:
            compile_kwargs.setdefault("options", inductor_options)
        return torch.compile(reference, **compile_kwargs)

    def apply(self, fallback_regions: list[str] | None = None) -> None:
        """Activate this policy and bind all registered named regions.

        The active policy is process-wide. A later call replaces it for lazy
        helpers such as FlexAttention and ``create_block_mask``.
        """
        global _ACTIVE_LOCAL_COMPILE_CONFIG

        regions = self.regions if self.regions is not None else fallback_regions
        if regions is None:
            regions = []

        unknown = [name for name in regions if name not in _LOCAL_COMPILE_CALLBACKS]
        if unknown:
            raise ValueError(
                f"Unknown local_compile_regions entries {unknown}; "
                f"registered values are {sorted(_LOCAL_COMPILE_CALLBACKS)}"
            )

        _ACTIVE_LOCAL_COMPILE_CONFIG = self
        _reset_compiled_helpers()
        for callbacks in _LOCAL_COMPILE_CALLBACKS.values():
            for bind_local_compile_fn in callbacks:
                bind_local_compile_fn(self, regions)


_ACTIVE_LOCAL_COMPILE_CONFIG = LocalCompileConfig()


def apply_local_compile(regions: list[str]) -> None:
    """Bind regions using the active local compile policy."""
    _ACTIVE_LOCAL_COMPILE_CONFIG.apply(regions)


def local_compile(
    name: str,
    *,
    batch_invariant: bool,
    **compile_kwargs: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a function that can be compiled independently.

    Args:
        name: Name used to enable the function in a model's
            ``local_compile_regions`` or the policy's ``regions`` override.
        batch_invariant: Whether the compiled function preserves batch invariance.
        **compile_kwargs: Additional ``torch.compile`` keyword arguments;
            ``fullgraph`` is fixed to ``True`` so each function forms one complete
            compile region.
    """
    if compile_kwargs.pop("fullgraph", True) is not True:
        raise ValueError("local_compile requires fullgraph=True.")

    def decorate(reference: Callable[..., Any]) -> Callable[..., Any]:
        fn = reference

        def bind_local_compile(
            config: LocalCompileConfig,
            regions: list[str],
        ) -> None:
            nonlocal fn
            enabled = name in regions
            batch_invariant_mode = is_in_batch_invariant_mode()
            if enabled and batch_invariant_mode and not batch_invariant:
                raise ValueError(
                    f"Local compile region {name!r} does not support "
                    "batch-invariant mode; remove it from local_compile_regions."
                )
            if enabled:
                fn = config.compile(
                    reference,
                    fullgraph=True,
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


def compiled(
    reference: Callable[..., Any],
    *,
    inductor_options: dict[str, Any] | None = None,
    **compile_kwargs: Any,
) -> Callable[..., Any]:
    """Compile ``reference`` lazily with the configured local backend.

    ``inductor_options`` are passed as ``torch.compile(options=...)`` only when
    the effective backend is Inductor.
    """
    fn: Callable[..., Any] | None = None

    def reset() -> None:
        nonlocal fn
        fn = None

    @functools.wraps(reference)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        nonlocal fn
        if fn is None:
            fn = _ACTIVE_LOCAL_COMPILE_CONFIG.compile(
                reference,
                inductor_options=inductor_options,
                **compile_kwargs,
            )
        return fn(*args, **kwargs)

    _COMPILED_HELPER_RESETS.append(reset)
    return wrapped


__all__ = [
    "apply_local_compile",
    "compiled",
    "LocalCompileConfig",
    "local_compile",
]
