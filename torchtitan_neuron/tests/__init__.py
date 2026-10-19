# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

"""Neuron enablement shim for PyTorch's DTensorTestBase.

Runs at import of the ``torchtitan_neuron.tests`` package. It must live in the
package ``__init__`` (not ``conftest.py``) because ``MultiProcessTestCase``
spawns each rank in a fresh interpreter that re-imports the test module and its
package ``__init__`` but never ``conftest.py`` -- this is the only place the
patch reaches the ranks where the process group is built.

torch_neuronx registers ``neuron`` as a PrivateUse1 accelerator with a ``neuron``
c10d backend, so ``DTensorTestBase.backend`` resolves to the bare string
``"neuron"``, which ``common_dtensor.init_pg`` rejects. The compound
``"cpu:gloo,<device>:<backend>"`` form is accepted, so return that instead.
Mirrors ``tests/unit_tests/__init__.py``; both are unnecessary once the torch
build under test accepts a PrivateUse1 device's registered backend directly.
"""

import torch
import torch.distributed as dist
from torch.testing._internal.distributed._tensor import common_dtensor

_PRIVATEUSE1_NAME = torch._C._get_privateuse1_backend_name()

if not getattr(common_dtensor.DTensorTestBase, "_neuron_backend_patched", False):
    _original_backend = common_dtensor.DTensorTestBase.backend

    def _backend(self):
        if self.device_type == _PRIVATEUSE1_NAME:
            curr = dist.get_default_backend_for_device(self.device_type)
            return f"cpu:gloo,{self.device_type}:{curr}"
        return _original_backend.fget(self)

    common_dtensor.DTensorTestBase.backend = property(_backend)
    common_dtensor.DTensorTestBase._neuron_backend_patched = True
