"""ROCOF helpers aligned with current legacy behavior."""

from __future__ import annotations

import numpy as np
import pandas as pd


def robust_rocof(
    freq_hz: np.ndarray,
    timestamps_s: np.ndarray,
    median_window: int = 5,
    clip_limits: tuple[float, float] = (-20.0, 20.0),
) -> np.ndarray:
    """Compute robust ROCOF using gradient, median smoothing, and clipping."""
    freq = np.asarray(freq_hz, dtype=float)
    t = np.asarray(timestamps_s, dtype=float)

    if freq.shape != t.shape:
        raise ValueError("freq_hz and timestamps_s must have the same shape.")

    if freq.size < 3:
        return np.zeros_like(freq)

    rocof = np.gradient(freq, t)
    rocof = pd.Series(rocof).rolling(window=median_window, center=True, min_periods=1).median().to_numpy()
    low, high = clip_limits
    return np.clip(rocof, low, high)
