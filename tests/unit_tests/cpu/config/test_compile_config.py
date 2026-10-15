# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.components.loss import compute_logprobs, cross_entropy_loss, mse_loss
from torchtitan.distributed.local_compile import (
    apply_local_compile,
    compiled,
    local_compile,
)
from torchtitan.models.common.moe import _microbatch_load_balance_local_stats


@pytest.fixture(autouse=True)
def reset_local_compile():
    apply_local_compile([])
    yield
    apply_local_compile([])


def test_apply_local_compile_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match=r"foo.*registered values"):
        apply_local_compile(["foo"])


def test_local_compile_uses_default_backend(monkeypatch) -> None:
    compiled_calls = []

    @local_compile("test_local_compile", batch_invariant=False)
    def fn(value: int) -> tuple[str, int]:
        return "eager", value

    def fake_compile(reference, **kwargs):
        compiled_calls.append((reference, kwargs))

        def compiled(value: int) -> tuple[str, int]:
            return "compiled", value

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    apply_local_compile(["test_local_compile"])

    assert fn(3) == ("compiled", 3)
    assert compiled_calls[0][1]["backend"] == "inductor"
    assert compiled_calls[0][1]["fullgraph"] is True


def test_local_compile_forwards_backend_and_compile_kwargs(monkeypatch) -> None:
    compiled_calls = []
    options = {"max_autotune": True}

    @local_compile(
        "test_compile_kwargs",
        batch_invariant=False,
        dynamic=True,
        fullgraph=True,
        options=options,
    )
    def fn(value: int) -> int:
        return value

    def fake_compile(reference, **kwargs):
        compiled_calls.append((reference, kwargs))
        return reference

    monkeypatch.setattr(torch, "compile", fake_compile)
    apply_local_compile(["test_compile_kwargs"], backend="aot_eager")

    assert fn(3) == 3
    assert compiled_calls[0][1] == {
        "backend": "aot_eager",
        "fullgraph": True,
        "dynamic": True,
        "options": options,
    }


def test_local_compile_rejects_fullgraph_false() -> None:
    with pytest.raises(ValueError, match="fullgraph=True"):
        local_compile(
            "test_fullgraph",
            batch_invariant=False,
            fullgraph=False,
        )


def test_local_compile_rejects_per_region_backend() -> None:
    with pytest.raises(ValueError, match="apply_local_compile"):
        local_compile(
            "test_backend",
            batch_invariant=False,
            backend="aot_eager",
        )


def test_compiled_helper_binds_lazily_and_rebinds_backend(monkeypatch) -> None:
    compiled_calls = []
    inductor_options = {"max_autotune": True}

    def reference(value: int) -> tuple[str, int]:
        return "eager", value

    def fake_compile(reference, **kwargs):
        compiled_calls.append((reference, kwargs))
        backend = kwargs["backend"]

        def compiled_fn(value: int) -> tuple[str, int]:
            return backend, value

        return compiled_fn

    monkeypatch.setattr(torch, "compile", fake_compile)
    helper = compiled(reference, inductor_options=inductor_options)

    assert compiled_calls == []
    assert helper(3) == ("inductor", 3)
    assert compiled_calls == [
        (
            reference,
            {"backend": "inductor", "options": inductor_options},
        )
    ]

    inductor_options["max_autotune"] = False
    apply_local_compile([])

    assert compiled_calls[-1] == (
        reference,
        {"backend": "inductor", "options": {"max_autotune": False}},
    )

    apply_local_compile([], backend="custom_device_backend")

    assert helper(4) == ("custom_device_backend", 4)
    assert compiled_calls[-1] == (
        reference,
        {"backend": "custom_device_backend"},
    )


def test_torchtitan_helpers_use_configured_backend(monkeypatch) -> None:
    compiled_calls = []

    def fake_compile(reference, **kwargs):
        compiled_calls.append((reference, kwargs))
        return reference

    monkeypatch.setattr(torch, "compile", fake_compile)

    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    from torchtitan.models.common import vision_encoder  # noqa: F401

    from torchtitan.models.common.attention import attention

    assert compiled_calls == []

    apply_local_compile([], backend="custom_device_backend")

    helper_calls = [
        call
        for call in compiled_calls
        if call[0] in (flex_attention, create_block_mask)
    ]
    assert helper_calls == [
        (flex_attention, {"backend": "custom_device_backend"}),
        (create_block_mask, {"backend": "custom_device_backend"}),
        (create_block_mask, {"backend": "custom_device_backend"}),
    ]
    assert attention.FlexInnerAttention._compiled_flex_attn is not flex_attention


def test_local_compile_rejects_non_batch_invariant_region(monkeypatch) -> None:
    @local_compile("test_non_batch_invariant", batch_invariant=False)
    def fn(value: int) -> int:
        return value

    monkeypatch.setattr(
        "torchtitan.distributed.local_compile.is_in_batch_invariant_mode",
        lambda: True,
    )
    with pytest.raises(
        ValueError, match=r"test_non_batch_invariant.*local_compile_regions"
    ):
        apply_local_compile(["test_non_batch_invariant"])


def test_loss_functions_use_local_compile(monkeypatch) -> None:
    compiled_names = []

    def fake_compile(reference, **kwargs):
        del kwargs
        compiled_names.append(reference.__name__)
        return reference

    monkeypatch.setattr(torch, "compile", fake_compile)
    apply_local_compile(["loss"])

    assert compiled_names[:3] == [
        cross_entropy_loss.__name__,
        mse_loss.__name__,
        compute_logprobs.__name__,
    ]


def test_moe_aux_loss_uses_local_compile(monkeypatch) -> None:
    compiled = []

    def fake_compile(reference, **kwargs):
        compiled.append((reference, kwargs))
        return reference

    monkeypatch.setattr(torch, "compile", fake_compile)
    apply_local_compile(["moe_aux_loss"])

    assert compiled[0] == (
        _microbatch_load_balance_local_stats.__wrapped__,
        {"backend": "inductor", "fullgraph": True},
    )
