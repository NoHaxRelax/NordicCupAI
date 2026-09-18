"""Select the native engine while retaining the original implementation as an oracle."""
from functools import wraps
import os

try:
    import _survival_native as native
except ModuleNotFoundError as exc:
    if exc.name != "_survival_native":
        raise
    native = None


def resolve_backend(backend=None):
    backend = backend or os.environ.get("SURVIVAL_BACKEND", "auto")
    if backend not in ("auto", "python", "rust"):
        raise ValueError("backend must be 'auto', 'python', or 'rust'")
    if backend == "auto":
        return "rust" if native is not None else "python"
    if backend == "rust" and native is None:
        raise RuntimeError(
            "Rust engine is not installed. Run: "
            "uv pip install ./survival-simulator/rust"
        )
    return backend


def native_method(method):
    @wraps(method)
    def dispatch(self, *args, **kwargs):
        if getattr(self, "_backend", "python") == "rust":
            return getattr(native, method.__name__)(self, *args, **kwargs)
        return method(self, *args, **kwargs)
    return dispatch
