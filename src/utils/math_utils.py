from __future__ import annotations

import numpy as np


EPS: float = 1e-9


def robust_median(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    return float(np.nanmedian(finite))


def robust_scale(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    med = float(np.nanmedian(finite))
    mad = float(np.nanmedian(np.abs(finite - med)))
    return max(1.4826 * mad, EPS)


def safe_quantile(values: np.ndarray, q: float, fallback: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return fallback
    return float(np.nanquantile(finite, q))


def safe_mean(values: np.ndarray, fallback: float = 0.0) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return fallback
    return float(np.nanmean(finite))

