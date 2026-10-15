# Custom device plugins

TorchTitan loads an installed compile-backend plugin before model modules bind
backend-less `torch.compile` calls. A custom device package can use this hook
to select its registered backend without patching TorchTitan.

## Plugin package

Register an entry point whose name matches the device type returned by PyTorch:

```text
torchtitan-custom-device/
  pyproject.toml
  torchtitan_custom_device/
    __init__.py
    compile_backend.py
```

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "torchtitan-custom-device"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["torch"]

[project.entry-points."torchtitan.compile_backends"]
custom_device = "torchtitan_custom_device.compile_backend:BACKEND"
```

The plugin provides the device-specific backend name:

```python
# torchtitan_custom_device/compile_backend.py
BACKEND = "custom_device_backend"
```

Replace `custom_device` with the device type registered with PyTorch and
`custom_device_backend` with the registered Dynamo backend name.

## Install and run

Install the plugin into the same environment as TorchTitan:

```bash
python -m pip install -e /path/to/torchtitan-custom-device
```

Then launch TorchTitan normally. When PyTorch reports `custom_device` as the
active device, TorchTitan discovers the matching entry point and calls
`torch.compiler.set_default_backend(BACKEND)` before importing FlexAttention.
Backend-less `torch.compile` calls therefore use `custom_device_backend`; calls
that pass `backend=` explicitly keep their requested backend.
