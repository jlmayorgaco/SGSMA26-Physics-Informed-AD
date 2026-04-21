"""Trend estimation helpers for raw-signal noise profiling."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def safe_odd_window(n: int, target: int) -> int:
    """Return a safe odd window length bounded by series length."""
    if n < 5:
        return max(1, n)
    win = min(target, n)
    if win % 2 == 0:
        win -= 1
    return max(5, win)


def estimate_trend(y: np.ndarray, family: str) -> tuple[np.ndarray, str]:
    """Estimate signal trend using legacy-compatible family rules."""
    if len(y) < 7:
        return np.full_like(y, np.nanmedian(y)), "median_short_series"

    if family in {"current_mag", "voltage_mag", "frequency"}:
        win = safe_odd_window(len(y), 301)
        trend = (
            pd.Series(y)
            .rolling(window=win, center=True, min_periods=max(5, win // 5))
            .median()
            .bfill()
            .ffill()
            .to_numpy(dtype=float)
        )
        return trend, f"rolling_median_w{win}"

    win = safe_odd_window(len(y), 101)
    poly = 3 if win >= 7 else 2
    return savgol_filter(y, win, polyorder=poly), f"savgol_w{win}_p{poly}"
