# Neuron Expert Parallelism

This package integrates the NKI Library (`nkilib`) `permute_a2av` and
`unpermute_a2av` kernels with TorchTitan's token dispatcher and autograd.

The packed dispatch and combine kernels are imported from the NKI Library.
Their autograd wrappers and token-dispatcher integration live here.

## Scope and known limitation

The `neuron_ep` dispatcher currently supports the published packed-kernel contract:
Trn3 LNC=2, EP >= 8, and one local expert per EP rank. Its receive capacity is
shape-static and grouped-mm aligned. Standard and local dispatch do not add
Neuron-specific grouped-mm padding in core. A follow-up is required for
multi-local-expert regrouping.
