# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

"""torchtitan_neuron: Neuron (AWS Trainium) specific extensions for torchtitan.

Holds Neuron-specific kernels, ops, and configs (e.g. NKI MoE) kept out of
core torchtitan. Neuron recipes select them by converting the stock model
config (for example ``to_nki_moe_config`` for each model's NKI MoE).
"""
