# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal

import torch


_fused_qkv_load_planning: ContextVar[bool] = ContextVar(
    "fused_qkv_load_planning", default=False
)


@dataclass(frozen=True, slots=True)
class FusedQKVLoadTarget:
    """Metadata-only checkpoint target for one projection of a fused QKV tensor."""

    fused_tensor: torch.Tensor
    projection: Literal["q", "k", "v"]
    shape: torch.Size
    dtype: torch.dtype
    n_kv_heads: int
    heads_per_kv: int
    head_dim: int


@contextmanager
def plan_fused_qkv_load() -> Iterator[None]:
    """Make FusedQKVLinear state-dict hooks emit metadata-only load targets."""

    token = _fused_qkv_load_planning.set(True)
    try:
        yield
    finally:
        _fused_qkv_load_planning.reset(token)


def is_fused_qkv_load_planning() -> bool:
    return _fused_qkv_load_planning.get()
