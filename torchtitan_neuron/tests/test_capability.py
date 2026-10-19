# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

"""Unit tests for torchtitan.tools.utils.has_neuron_capability.

The off-device and simulated-trn3 cases run anywhere (they do not need real
Neuron hardware or the neuron custom ops). The real-hardware case is gated on
trn3 because MXFP8 is a trn3-only capability.
"""

import sys
import types
import unittest
from unittest import mock

from torchtitan.tools.utils import has_neuron_capability


class TestHasNeuronCapability(unittest.TestCase):
    def test_returns_false_without_torch_neuronx(self):
        # When torch_neuronx can't be imported, mxfp8 is unsupported -- forced
        # here (sys.modules entry -> None makes the import raise) so it holds on
        # any platform, including a real trn3 pod where it would otherwise be True.
        with mock.patch.dict(sys.modules, {"torch_neuronx": None}):
            self.assertFalse(has_neuron_capability("mxfp8"))

    def test_unknown_feature_returns_false(self):
        # Only "mxfp8" is a real capability today; anything else is False,
        # including on hardware.
        self.assertFalse(has_neuron_capability("fp4"))
        self.assertFalse(has_neuron_capability("nvfp4"))

    def test_feature_name_is_case_insensitive(self):
        # "MXFP8" and "mxfp8" resolve identically (the .lower() path) on any
        # platform -- both False off-device, both True on trn3.
        self.assertEqual(has_neuron_capability("MXFP8"), has_neuron_capability("mxfp8"))

    def _fake_torch_neuronx(self, platform_target: str) -> types.ModuleType:
        mod = types.ModuleType("torch_neuronx")
        utils_mod = types.ModuleType("torch_neuronx.utils")
        utils_mod.get_platform_target = lambda: platform_target
        mod.utils = utils_mod
        return mod

    def test_simulated_trn3_returns_true(self):
        fake = self._fake_torch_neuronx("trn3")
        with mock.patch.dict(
            sys.modules,
            {"torch_neuronx": fake, "torch_neuronx.utils": fake.utils},
        ):
            self.assertTrue(has_neuron_capability("mxfp8"))

    def test_simulated_trn3u_returns_true(self):
        # startswith("trn3") also matches later trn3 variants.
        fake = self._fake_torch_neuronx("trn3u")
        with mock.patch.dict(
            sys.modules,
            {"torch_neuronx": fake, "torch_neuronx.utils": fake.utils},
        ):
            self.assertTrue(has_neuron_capability("mxfp8"))

    def test_simulated_trn2_returns_false(self):
        fake = self._fake_torch_neuronx("trn2")
        with mock.patch.dict(
            sys.modules,
            {"torch_neuronx": fake, "torch_neuronx.utils": fake.utils},
        ):
            self.assertFalse(has_neuron_capability("mxfp8"))

    @unittest.skipUnless(
        has_neuron_capability("mxfp8"), "requires a real Neuron trn3 device"
    )
    def test_real_trn3_returns_true(self):
        self.assertTrue(has_neuron_capability("mxfp8"))


if __name__ == "__main__":
    unittest.main()
