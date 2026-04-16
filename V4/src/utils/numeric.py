from __future__ import annotations

from typing import Optional

import numpy as np


def mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")

    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def robust_zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    scale = mad(x)

    if not np.isfinite(scale) or scale < 1e-12:
        std = np.nanstd(x)
        if not np.isfinite(std) or std < 1e-12:
            return np.zeros_like(x, dtype=float)
        return (x - med) / std

    return 0.6744897501960817 * (x - med) / scale


def safe_corrcoef(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return None

    x2, y2 = x[mask], y[mask]
    sx, sy = np.std(x2), np.std(y2)

    if sx < 1e-12 or sy < 1e-12:
        return None

    return float(np.corrcoef(x2, y2)[0, 1])


def autocorr_lag1(x: np.ndarray) -> Optional[float]:
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(x)
    x = x[mask]

    if x.size < 3:
        return None

    return safe_corrcoef(x[:-1], x[1:])