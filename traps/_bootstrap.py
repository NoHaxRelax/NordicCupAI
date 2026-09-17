"""Locate the survival-simulator root so `src.*` imports resolve.

Lets this folder be dropped anywhere inside the simulator checkout without
every script needing PYTHONPATH set by hand. Walks up from this file looking
for the directory that contains `src/elements/environment.py`.
"""

import os
import sys

_MARKER = os.path.join("src", "elements", "environment.py")


def simulator_root(start=None):
    """The directory holding `src/`, or None if this is not inside a checkout."""
    here = os.path.abspath(start or os.path.dirname(__file__))
    for _ in range(8):
        if os.path.isfile(os.path.join(here, _MARKER)):
            return here
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent
    return None


def ensure_on_path():
    """Put the simulator root on sys.path. Returns it, or None if not found."""
    root = simulator_root()
    if root and root not in sys.path:
        sys.path.insert(0, root)
    return root


def require_simulator():
    """Same, but fail with a useful message instead of an ImportError later."""
    root = ensure_on_path()
    if root is None:
        raise SystemExit(
            "Could not find the survival-simulator root (the directory that "
            "contains src/elements/environment.py).\n"
            "Copy this 'traps' folder inside your survival-simulator checkout, "
            "or set PYTHONPATH to point at it."
        )
    return root


def headless():
    """Stop pygame opening a window; the simulator imports it at module load."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
