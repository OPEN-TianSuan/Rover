"""Load optional native kernels only when a quantized operation needs them."""

from importlib import import_module


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    try:
        native = import_module("rover.kernels._kernels")
    except ImportError as exc:
        raise RuntimeError(
            "Quantized KV cache requires CUDA kernels. Run: uv run python scripts/build_kernels.py"
        ) from exc
    return getattr(native, name)
