"""Baseline estimators for m1 normalization (parity-first)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def robust_trimmed_mean(values: pd.Series, trim_frac: float = 0.1) -> tuple[float, str]:
    """Return legacy-compatible baseline for magnitude/frequency series.

    Notes:
    - `trim_frac` is currently unused by design to preserve legacy behavior.
    """
    _ = trim_frac
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(clean) == 0:
        return 1.0, "fallback_constant"
    return float(np.mean(clean)), "mean"


def robust_center(values: pd.Series) -> tuple[float, str]:
    """Return legacy-compatible centering value for ROCOF."""
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(clean) == 0:
        return 0.0, "fallback_zero"
    if len(clean) >= 50:
        return float(np.median(clean)), "median"
    return float(np.mean(clean)), "mean"
