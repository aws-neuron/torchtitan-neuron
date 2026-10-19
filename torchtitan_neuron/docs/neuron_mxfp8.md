# Neuron MXFP8 (trn3)

MXFP8 dense-Linear training for qwen3 on AWS Trainium trn3 (NeuronCore-v4), on
torchao with Neuron MXFP8 support. titan builds `MXFP8Linear` with
`KernelPreference.NEURON`; torchao selects the GEMM backend internally.

## Requirements
Trn3 hardware, torchao with Neuron MXFP8 support

## Usage

Torchtitan swaps each in-layer `Linear` module (attention + FFN; the LM head stays
bf16) for torchao's `MXFP8Linear` with `kernel_preference="neuron"`. MXFP8 is gated
on trn3 (`has_neuron_capability("mxfp8")`); off trn3 the converter/linear raises a
clear `ValueError`. Enable it with the named config:

### Named config

`torchtitan_neuron/models/qwen3/config_registry.py` ships `qwen3_8b_mxfp8`, which
attaches `MXFP8LinearConverter(fqns=["layers"], kernel_preference="neuron")` to the
model spec:

```bash
NGPU=64 MODULE=torchtitan_neuron.models.qwen3 CONFIG=qwen3_8b_mxfp8 ./run_train.sh
```

## Tests

trn3-gated unit tests:

```bash
python -m pytest \
  torchtitan_neuron/tests/test_mxfp8_numerics.py \
  torchtitan_neuron/tests/test_capability.py \
  torchtitan_neuron/tests/test_metrics_mfu.py
```
