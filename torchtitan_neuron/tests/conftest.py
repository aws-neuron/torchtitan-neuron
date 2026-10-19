# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

import os


# Import torch_neuronx explicitly from tests after torch has finished importing.
# Backend autoload can otherwise recurse through torch_neuronx while torch is
# partially initialized and duplicate-register core operator namespaces.
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
