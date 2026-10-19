# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Modifications Copyright (c) Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the BSD-3-Clause license.

"""Neuron enablement shim for PyTorch's DTensorTestBase.

This runs at import of the ``tests.unit_tests`` package. It must live here (not
in ``conftest.py``) because ``MultiProcessTestCase`` launches each rank in a
fresh spawned interpreter that never loads ``conftest.py`` -- but every spawned
rank re-imports the test module, and therefore this package ``__init__``. This
is the only place a patch reaches the ranks where the process group is built.

Problem: torch_neuronx registers ``neuron`` as a PrivateUse1 accelerator with a
``neuron`` c10d backend, so ``DTensorTestBase.backend`` resolves to the bare
string ``"neuron"``. PyTorch's ``common_dtensor.init_pg`` only accepts a fixed
set of backend strings; the bare device backend is rejected with "Backend
neuron not supported!". The compound ``"cpu:gloo,<device>:<backend>"`` form IS
in that accepted set, so we return that instead.

UPSTREAM FIX: recent PyTorch widens ``init_pg``'s accepted-backend list to also
accept a device's registered backend (bare and compound) for any PrivateUse1
accelerator. Once the torch build under test carries that change, the bare
``"neuron"`` backend is accepted directly and this shim is unnecessary -- it can
be removed (and the enabled tests dropped from the conftest skip list).
"""

import torch
import torch.distributed as dist
from torch.testing._internal.distributed._tensor import common_dtensor

# Name of the registered PrivateUse1 backend (e.g. "neuron"); "privateuseone"
# when none is registered.
_PRIVATEUSE1_NAME = torch._C._get_privateuse1_backend_name()

_original_backend = common_dtensor.DTensorTestBase.backend


def _backend(self):
    # Only reshape the backend string for the PrivateUse1 accelerator (neuron);
    # cpu (gloo) and cuda (nccl) already resolve to an accepted bare backend.
    if self.device_type == _PRIVATEUSE1_NAME:
        curr = dist.get_default_backend_for_device(self.device_type)
        return f"cpu:gloo,{self.device_type}:{curr}"
    return _original_backend.fget(self)


common_dtensor.DTensorTestBase.backend = property(_backend)
