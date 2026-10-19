# Testing `torchtitan_neuron`

Tests in this directory mirror the `torchtitan_neuron` source layout. Keep
hardware-specific tests device-guarded and prefer CPU-runnable config and
registry tests when validating extension contracts.

The test configuration disables PyTorch device-backend autoload during
collection. Tests that require Neuron packages import them explicitly after
PyTorch initialization and skip when their dependencies are unavailable.

Run the suite from the repository root:

```bash
pytest -s torchtitan_neuron/tests/
```

`test_flex_attention.py` checks Neuron kernel lowering for static causal,
static sliding-window, and dynamic document masks. It requires `torch_neuronx`
and a Neuron device:

```bash
pytest -s torchtitan_neuron/tests/test_flex_attention.py
```
