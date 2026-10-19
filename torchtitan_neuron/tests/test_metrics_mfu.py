# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

"""Unit tests for MFU computation, including the Neuron MXFP8 4x-peak path.

These test the extracted ``compute_mfu`` helper directly (pure math), so they
run on CPU without real Neuron hardware: the Neuron branch is exercised by
patching ``has_neuron_capability`` rather than requiring a trn3 device.
"""

from unittest import mock

from torchtitan.components import metrics
from torchtitan.components.metrics import compute_mfu

# Simple round numbers so the expected values are obvious.
_FLOPS_PER_TOKEN = 100
_TPS = 10.0
_PEAK = 1000.0


def test_unquantized_mfu_uses_bf16_peak():
    # 100 * 100 * 10 / 1000 = 100.0
    mfu = compute_mfu(_FLOPS_PER_TOKEN, _TPS, _PEAK, has_quantization=False)
    assert mfu == 100 * _FLOPS_PER_TOKEN * _TPS / _PEAK


def test_quantized_non_neuron_mfu_is_none():
    # Off Neuron, a quantized run cannot report a meaningful MFU.
    with mock.patch.object(metrics.utils, "has_neuron_capability", return_value=False):
        mfu = compute_mfu(_FLOPS_PER_TOKEN, _TPS, _PEAK, has_quantization=True)
    assert mfu is None


def test_quantized_neuron_mxfp8_uses_4x_peak():
    # On trn3 MXFP8, peak is 4x bf16, so MFU is 1/4 of the naive value.
    with mock.patch.object(metrics.utils, "has_neuron_capability", return_value=True):
        mfu = compute_mfu(_FLOPS_PER_TOKEN, _TPS, _PEAK, has_quantization=True)
    assert mfu == 100 * _FLOPS_PER_TOKEN * _TPS / (4 * _PEAK)


def test_neuron_mxfp8_mfu_is_quarter_of_unquantized():
    # Cross-check the 4x relationship between the two branches directly.
    unquant = compute_mfu(_FLOPS_PER_TOKEN, _TPS, _PEAK, has_quantization=False)
    with mock.patch.object(metrics.utils, "has_neuron_capability", return_value=True):
        neuron = compute_mfu(_FLOPS_PER_TOKEN, _TPS, _PEAK, has_quantization=True)
    assert neuron == unquant / 4
