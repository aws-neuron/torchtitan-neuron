# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

"""MXFP8-vs-bf16 accuracy tests for the titan MXFP8 linear on Neuron trn3.

Builds the torchtitan ``MXFP8Linear`` (which wraps torchao's ``MXFP8Linear``
and, on trn3, selects the ``neuron`` GEMM kernel), runs a forward
+ backward on device, and checks the output and both gradients track a bf16 stock
reference within an MXFP8 quantization-error bar (eager: the bare linear vs
``F.linear``).

Gated on a real trn3 device: the ``neuron`` kernel + backend only dispatch there.
Shapes are kernel-valid (features multiples of 128; tokens multiple of 128).
"""

import unittest

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Replicate, Shard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.tools.utils import has_neuron_capability

from torchtitan_neuron.components.quantization import torchao_supports_neuron

pytestmark = pytest.mark.skipif(
    not torchao_supports_neuron(),
    reason="requires torchao with Neuron MXFP8 support",
)

# Kernel-valid shapes: in/out features multiples of 128, tokens multiple of 128.
_IN, _OUT, _TOKENS = 512, 512, 128

# MXFP8-vs-bf16 relative-error bars. Measured on trn3: eager fwd ~0.075,
# dgrad ~0.052, wgrad ~0.057. Bars carry headroom for seed variation.
_FWD_BAR = 0.10
_EAGER_GRAD_BAR = 0.10


def _rel_error(actual: torch.Tensor, ref: torch.Tensor) -> float:
    """Relative L2 error ``||actual - ref|| / ||ref||`` (both as f32 on CPU)."""
    a = actual.detach().to("cpu").float()
    r = ref.detach().to("cpu").float()
    return (torch.linalg.norm(a - r) / torch.linalg.norm(r)).item()


def _build_mxfp8_linear() -> tuple["nn.Module", torch.Tensor]:
    """Build a Neuron MXFP8 linear with a known bf16 weight (the reference ground
    truth). Cast to bf16 up front: the Neuron MXFP8 GEMM takes bf16 operands (an
    fp32 module, the build default, is rejected by the fused op)."""
    from torchtitan.components.quantization.mx import MXFP8Linear

    lin = MXFP8Linear.Config(
        in_features=_IN,
        out_features=_OUT,
        bias=False,
        kernel_preference="neuron",
    ).build()
    weight = torch.randn(_OUT, _IN, dtype=torch.bfloat16)
    with torch.no_grad():
        lin.weight.copy_(weight)
    return lin, weight


@unittest.skipUnless(
    has_neuron_capability("mxfp8"), "requires a real Neuron trn3 device"
)
class TestMXFP8LinearNumerics(unittest.TestCase):
    def _run_numerics(self, *, module, linear, weight, ref_fn, compiled, grad_bar):
        torch.manual_seed(0)
        runner = (
            torch.compile(module, backend="neuron", fullgraph=True)
            if compiled
            else module
        )

        x_cpu = torch.randn(_TOKENS, _IN, dtype=torch.bfloat16)
        x_dev = x_cpu.detach().to("neuron").requires_grad_(True)
        out = runner(x_dev)
        self.assertEqual(tuple(out.shape), (_TOKENS, _OUT))
        self.assertTrue(torch.isfinite(out.to("cpu")).all())
        out.sum().backward()

        # Gradients must propagate to the input and the weight (finite, right shape).
        self.assertIsNotNone(x_dev.grad)
        self.assertIsNotNone(linear.weight.grad)
        self.assertEqual(tuple(linear.weight.grad.shape), (_OUT, _IN))
        self.assertTrue(torch.isfinite(x_dev.grad.to("cpu")).all())
        self.assertTrue(torch.isfinite(linear.weight.grad.to("cpu")).all())

        # bf16 float reference from the same bf16-rounded weight.
        x_ref = x_cpu.detach().float().requires_grad_(True)
        w_ref = weight.detach().float().requires_grad_(True)
        out_ref = ref_fn(x_ref, w_ref)
        out_ref.sum().backward()

        fwd_rel = _rel_error(out, out_ref)
        dgrad_rel = _rel_error(x_dev.grad, x_ref.grad)
        wgrad_rel = _rel_error(linear.weight.grad, w_ref.grad)
        self.assertLess(fwd_rel, _FWD_BAR, f"fwd rel error {fwd_rel:.4f}")
        self.assertLess(dgrad_rel, grad_bar, f"input-grad rel error {dgrad_rel:.4f}")
        self.assertLess(wgrad_rel, grad_bar, f"weight-grad rel error {wgrad_rel:.4f}")

    def test_forward_backward_vs_bf16_linear(self):
        lin, weight = _build_mxfp8_linear()
        lin = lin.to(torch.bfloat16).to("neuron")
        self._run_numerics(
            module=lin,
            linear=lin,
            weight=weight,
            ref_fn=F.linear,
            compiled=False,
            grad_bar=_EAGER_GRAD_BAR,
        )


@unittest.skip(
    "DTensor + compile spawns worker processes (MultiProcessTestCase); the Neuron "
    "runtime cannot re-initialize in the children after the parent used it in a "
    "shared test session (Failed to initialize Neuron Runtime: status code 1). "
    "The DTensor + compile + NKI path is covered by the qwen3_8b_mxfp8 (tp4 x "
    "fsdp2) E2E compile run instead."
)
@unittest.skipUnless(
    has_neuron_capability("mxfp8"), "requires a real Neuron trn3 device"
)
class TestMXFP8LinearCompileDTensor(DTensorTestBase):
    """MXFP8 linear with a DTensor (TP-sharded) weight under torch.compile."""

    @property
    def world_size(self) -> int:
        return 2

    @with_comms
    def test_tp_sharded_linear_compiles_and_runs(self):
        from torchtitan.components.quantization.mx import MXFP8Linear

        torch.manual_seed(0)
        mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("tp",)
        )

        lin = (
            MXFP8Linear.Config(
                in_features=_IN,
                out_features=_OUT,
                bias=False,
                kernel_preference="neuron",
            )
            .build()
            .to(torch.bfloat16)
            .to(self.device_type)
        )

        # Column-parallel linear: shard the weight rows (out_features) across TP,
        # replicate the input. Output is sharded on the last dim.
        with torch.no_grad():
            w_dt = distribute_tensor(lin.weight.detach(), mesh, [Shard(0)])
        lin.weight = nn.Parameter(w_dt)

        x = distribute_tensor(
            torch.randn(_TOKENS, _IN, dtype=torch.bfloat16, device=self.device_type),
            mesh,
            [Replicate()],
        ).requires_grad_(True)

        compiled = torch.compile(lin, backend="neuron", fullgraph=True)
        out = compiled(x)

        full = out.full_tensor()
        self.assertEqual(tuple(full.shape), (_TOKENS, _OUT))
        self.assertTrue(torch.isfinite(full.to("cpu")).all())

        out.sum().backward()
        self.assertIsNotNone(lin.weight.grad)
        wgrad_full = lin.weight.grad.full_tensor()
        self.assertEqual(tuple(wgrad_full.shape), (_OUT, _IN))
        self.assertTrue(torch.isfinite(wgrad_full.to("cpu")).all())


# --- Neuron trn3 MXFP8 capability gating + kernel_preference flow ---


def test_mxfp8_linear_converter_accepts_neuron_trn3(monkeypatch):
    """MXFP8LinearConverter constructs on Neuron trn3 (no CUDA SM100)."""
    pytest.importorskip("torchao")
    from torchtitan.components.quantization import mx

    monkeypatch.setattr(mx, "has_cuda_capability", lambda *a: False)
    monkeypatch.setattr(mx, "has_neuron_capability", lambda feature: True)
    # Should not raise: Neuron trn3 satisfies the capability gate.
    mx.MXFP8LinearConverter(mx.MXFP8LinearConverter.Config(model_compile_enabled=False))


def test_mxfp8_linear_converter_rejects_unsupported_device(monkeypatch):
    """Neither CUDA SM100 nor Neuron trn3 -> ValueError naming trn3."""
    pytest.importorskip("torchao")
    from torchtitan.components.quantization import mx

    monkeypatch.setattr(mx, "has_cuda_capability", lambda *a: False)
    monkeypatch.setattr(mx, "has_neuron_capability", lambda feature: False)
    with pytest.raises(ValueError, match="trn3"):
        mx.MXFP8LinearConverter(
            mx.MXFP8LinearConverter.Config(model_compile_enabled=False)
        )


def test_kernel_preference_flows_to_torchao():
    """An explicit kernel_preference="neuron" reaches the torchao MXFP8Linear.

    Skips unless the installed torchao exposes KernelPreference.NEURON;
    upstream torchao has no NEURON member.
    """
    pytest.importorskip("torchao")
    kp = pytest.importorskip("torchao.quantization.quantize_.common.kernel_preference")
    if not hasattr(kp.KernelPreference, "NEURON"):
        pytest.skip("installed torchao has no KernelPreference.NEURON")

    from torchtitan.components.quantization.mx import MXFP8Linear

    module = MXFP8Linear.Config(
        in_features=16, out_features=32, bias=False, kernel_preference="neuron"
    ).build()
    assert module.kernel_preference == kp.KernelPreference.NEURON


if __name__ == "__main__":
    unittest.main()
