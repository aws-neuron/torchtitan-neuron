# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

"""Neuron bring-up skip configuration for the torchtitan unit test suite.

This conftest catalogs unit tests that cannot currently run on the Neuron
CPU-eager stack (torch 2.12.x + torch_neuronx 2.12.x) and skips them with a
specific reason so `pytest tests/unit_tests/` finishes green on a Neuron node.
Each skip documents *why* the test is incompatible; the buckets are:

  1. Gated HuggingFace repos that need auth/network unavailable on the node
     (meta-llama, deepseek-ai) -- these fail with 401 GatedRepoError.
  2. spmd_types operations that neuron does not yet support (copy-on-write /
     lazy clone), which fail deep inside the collective, not in setup.
  3. loss_parallel-path tests that assert *bitwise* parity between a custom
     cross-entropy decomposition and torch's loss_parallel(); the two kernels
     diverge at the bit level on the neuron device.

Optional-dep skips (e.g. `fla`/flash-linear-attention for the qwen3_5 model)
are handled at their own import site with unittest.skipIf, so only the
dependent test is skipped rather than the whole module -- see test_rope.py.

NOTE: torchvision must match torch. torchvision was upgraded to 0.27.x
alongside torch 2.12 (the stock 0.26 build was compiled for torch 2.11 and
failed to register `torchvision::nms` under 2.12). With 0.27 the flux model
and the multimodal dataset test import and run, so they are no longer skipped.

The DTensorTestBase distributed tests now run on the neuron device via the
backend shim in this package's __init__.py; only the spmd_types case below
remains skipped.
"""

import pytest

# Gated HuggingFace repos: need HF auth/network unavailable on the Neuron node
# (401 GatedRepoError).
_GATED_HF_SKIPS = {
    "test_tokenizer.py::TestTokenizerIntegration::test_download_and_build_tokenizer_test_repo_id_meta-llama",
    "test_tokenizer.py::TestTokenizerIntegration::test_download_and_build_tokenizer_test_repo_id_deepseek-ai",
}

# spmd_types operations unsupported on neuron: this test drives
# spmd_types redistribution whose `_replicate_to_varying` calls
# `Tensor._lazy_clone()` (copy-on-write), which returns null storage on the
# neuron device ("Expected storage != nullptr to be true, but got false").
_SPMD_TYPES_SKIPS = {
    "test_parallel_dims.py::TestSpmdLayout::test_partition_spec_order_controls_state_shard",
}

# loss_parallel path asserts bitwise parity (torch.equal) between a custom
# cross-entropy decomposition and torch's loss_parallel(); the neuron kernels
# are not bitwise-identical to the reference, so these fail on device. Not
# loosened to a tolerance on purpose -- the divergence is real and should be
# quantified before these are enabled.
_LOSS_PARALLEL_SKIPS = {
    "test_loss.py::TestLossParallelCrossEntropy::test_loss_parallel_cross_entropy_parity",
    "test_loss_dtensor.py::TestCrossEntropyDTensor::test_legacy_tp_loss_parallel",
}

# Reason strings shared by every entry in each bucket.
_SKIP_BUCKETS = [
    (
        _GATED_HF_SKIPS,
        "Gated HuggingFace repo requires auth/network unavailable on the Neuron "
        "node (401 GatedRepoError).",
    ),
    (
        _SPMD_TYPES_SKIPS,
        "spmd_types redistribution calls Tensor._lazy_clone() (copy-on-write), "
        "unsupported on the neuron device: 'Expected storage != nullptr to be "
        "true, but got false'.",
    ),
    (
        _LOSS_PARALLEL_SKIPS,
        "loss_parallel path asserts bitwise parity (torch.equal) with torch's "
        "loss_parallel(); the neuron kernels are not bitwise-identical to the "
        "reference. Divergence should be quantified before enabling.",
    ),
]


def pytest_collection_modifyitems(config, items):
    """Attach skip markers (with reasons) to known-incompatible unit tests.

    Matching is by nodeid substring so it is stable regardless of the
    directory pytest is invoked from (nodeids are prefixed with the
    rootdir-relative path, e.g. "tests/unit_tests/test_loss.py::...") and
    still matches parametrized tests whose param id is appended to the name
    (e.g. the tokenizer repo-id cases).
    """
    for item in items:
        for nodeids, reason in _SKIP_BUCKETS:
            if any(n in item.nodeid for n in nodeids):
                item.add_marker(pytest.mark.skip(reason=reason))
                break
