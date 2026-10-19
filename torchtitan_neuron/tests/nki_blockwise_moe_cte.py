#!/usr/bin/env python3
# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

"""Shared handle for the pure blockwise MoE-CTE NKI kernel (no router).

``moe_cte`` (nkilib.core.moe.moe_cte) is the standalone blockwise expert matmul: it takes
block metadata (token_position_to_id, block_to_expert, block_size) as inputs and tiles
internally by block_size, so it compiles where a single large ``aten::_grouped_mm`` over
the same rows overflows the neuronx-cc 5M-instruction limit (NCC_EBVF030).

Exposes one wrapped kernel and a default spec so the standalone blockwise bench
and the neuron_ep dispatcher bench share one expert-compute definition.
"""

from __future__ import annotations

from nkilib.core.moe.moe_cte import (
    moe_cte,
    MoECTEImplementation,
    MoECTESpec,
    ShardOnIConfig,
)

from torch_neuronx.nki_hop import wrap_nki

# wrap_nki handle; index with [(lnc,)] at the callsite.
wrapped_moe_cte = wrap_nki(moe_cte)


def default_spec():
    """Simplest working spec: I-sharding, defaults. bf16 TRN2/TRN3 path."""
    return MoECTESpec(
        implementation=MoECTEImplementation.shard_on_i,
        shard_on_I=ShardOnIConfig(),
    )
