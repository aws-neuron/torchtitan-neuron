# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Function-scoped torch.compile registration and configuration."""

import functools
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from torchtitan.distributed.batch_invariant import is_in_batch_invariant_mode


_LOCAL_COMPILE_CALLBACKS: dict[str, list[Callable[["LocalCompileConfig"], None]]] = {}
_LOCAL_COMPILE_GENERATION = 0


@dataclass(kw_only=True, slots=True)
class LocalCompileConfig:
    """Configuration for TorchTitan-owned local ``torch.compile`` boundaries."""

    regions: list[str] = field(default_factory=list)
    backend: str = "inductor"

    def __bool__(self) -> bool:
        return bool(self.regions)

    def __contains__(self, region: str) -> bool:
        return region in self.regions

    def append(self, region: str) -> None:
        self.regions.append(region)

    def compile(
        self,
        reference: Callable[..., Any],
        *,
        inductor_options: dict[str, Any] | None = None,
        **compile_kwargs: Any,
    ) -> Callable[..., Any]:
        backend = compile_kwargs.setdefault("backend", self.backend)
        if backend == "inductor" and inductor_options is not None:
            compile_kwargs.setdefault("options", inductor_options)
        return torch.compile(reference, **compile_kwargs)

    def apply(self) -> None:
        """Activate this config and bind all registered named regions."""
        global _ACTIVE_LOCAL_COMPILE_CONFIG, _LOCAL_COMPILE_GENERATION

        unknown = [
            name for name in self.regions if name not in _LOCAL_COMPILE_CALLBACKS
        ]
        if unknown:
            raise ValueError(
                f"Unknown local_compile_regions entries {unknown}; "
                f"registered values are {sorted(_LOCAL_COMPILE_CALLBACKS)}"
            )

        _ACTIVE_LOCAL_COMPILE_CONFIG = self
        _LOCAL_COMPILE_GENERATION += 1
        for callbacks in _LOCAL_COMPILE_CALLBACKS.values():
            for bind_local_compile_fn in callbacks:
                bind_local_compile_fn(self)


_ACTIVE_LOCAL_COMPILE_CONFIG = LocalCompileConfig()


def _as_local_compile_config(
    value: list[str] | LocalCompileConfig,
) -> LocalCompileConfig:
    """Normalize the existing region-list setting to one local compile config."""
    if isinstance(value, LocalCompileConfig):
        return value
    return LocalCompileConfig(regions=value)


def apply_local_compile(value: list[str] | LocalCompileConfig) -> None:
    """Bind local compile boundaries from a region list or full config."""
    _as_local_compile_config(value).apply()


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

    def decorate(reference: Callable[..., Any]) -> Callable[..., Any]:
        fn = reference

        def bind_local_compile(config: LocalCompileConfig) -> None:
            nonlocal fn
            enabled = name in config.regions
            batch_invariant_mode = is_in_batch_invariant_mode()
            if enabled and batch_invariant_mode and not batch_invariant:
                raise ValueError(
                    f"Local compile region {name!r} does not support "
                    "batch-invariant mode; remove it from local_compile_regions."
                )
            if enabled:
                fn = config.compile(reference, fullgraph=True, **compile_kwargs)
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
    """Compile lazily with the active local compile backend."""
    fn: Callable[..., Any] | None = None
    applied_generation = -1

    @functools.wraps(reference)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        nonlocal applied_generation, fn
        if applied_generation != _LOCAL_COMPILE_GENERATION:
            fn = _ACTIVE_LOCAL_COMPILE_CONFIG.compile(
                reference,
                inductor_options=inductor_options,
                **compile_kwargs,
            )
            applied_generation = _LOCAL_COMPILE_GENERATION
        assert fn is not None
        return fn(*args, **kwargs)

    return wrapped


__all__ = [
    "apply_local_compile",
    "compiled",
    "local_compile",
    "LocalCompileConfig",
]
