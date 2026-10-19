# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Modifications Copyright (c) Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the BSD-3-Clause license.

import importlib.util
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "OverrideDefinitions",
    "_ON_NEURON",
]

_ON_NEURON = importlib.util.find_spec("torch_neuronx") is not None


@dataclass
class OverrideDefinitions:
    """
    This class is used to define the override definitions for the integration tests.
    """

    override_args: Sequence[Sequence[str]] = tuple(tuple(" "))
    test_descr: str = "default"
    test_name: str = "default"
    ngpu: int = 4
    disabled: bool = False
    skip_rocm_test: bool = False
    skip_neuron: bool = False
    timeout: int | None = None

    def __repr__(self):
        return self.test_descr
