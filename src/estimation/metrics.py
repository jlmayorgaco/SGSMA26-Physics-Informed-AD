"""Pure estimation metric helpers mirrored from legacy behavior."""

from __future__ import annotations

import numpy as np


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Root mean squared error."""
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a_arr - b_arr) ** 2)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute error."""
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    return float(np.mean(np.abs(a_arr - b_arr)))


def rel_rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Relative RMSE normalized by reference std with epsilon guard."""
    denom = float(np.std(np.asarray(a, dtype=float)) + 1e-12)
    return float(rmse(a, b) / denom)


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation with constant-vector safety checks."""
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)

    if a_arr.size < 2 or b_arr.size < 2:
        return float("nan")

    if np.std(a_arr) < 1e-12 or np.std(b_arr) < 1e-12:
        return float("nan")

    return float(np.corrcoef(a_arr, b_arr)[0, 1])
