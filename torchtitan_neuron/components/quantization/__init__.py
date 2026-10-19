# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

"""Neuron-only quantization extensions kept out of core torchtitan.

Holds the MXFP8 pre-quantized SimpleFSDP weight path (``mx_prequant``), which
depends on torchao's ``neuron::`` custom ops. Opt in per recipe with
``NeuronMXFP8LinearConverter(prequantize_weights=True)``; the trainer refreshes
the encodings via the generic ``refresh_weight_derived_state`` hook.
"""

import importlib.util


def torchao_supports_neuron() -> bool:
    """True iff torchao is installed and exposes Neuron MXFP8 support.

    The Neuron MXFP8 path needs the torchao build that adds
    ``KernelPreference.NEURON`` and the ``neuron::`` custom ops. Public torchao (a
    plain ``pip install torchao``) lacks it; callers skip or raise rather than
    crash. The fork is not vended publicly yet.
    """
    if importlib.util.find_spec("torchao") is None:
        return False
    try:
        from torchao.quantization.quantize_.common.kernel_preference import (
            KernelPreference,
        )
    except Exception:
        return False
    return hasattr(KernelPreference, "NEURON")
