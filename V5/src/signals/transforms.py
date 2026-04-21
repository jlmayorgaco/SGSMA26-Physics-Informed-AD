"""Small angle utilities preserved for parity with legacy behavior."""

from __future__ import annotations

import numpy as np


def wrap_deg(values: np.ndarray | float) -> np.ndarray:
    """Wrap degree values into [-180, 180)."""
    arr = np.asarray(values, dtype=float)
    return ((arr + 180.0) % 360.0) - 180.0


def angle_diff_deg(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    """Compute wrapped angular difference a - b in degrees."""
    return wrap_deg(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))
