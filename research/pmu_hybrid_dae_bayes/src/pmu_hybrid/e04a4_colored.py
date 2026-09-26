"""Physics-only low-rank colored discrepancy primitives for E04-A4."""
from __future__ import annotations
import numpy as np


def fit_ar1(series: np.ndarray) -> tuple[float, float]:
    x = np.asarray(series, dtype=float).ravel(); a, b = x[:-1], x[1:]
    f = float((a @ b) / max(a @ a, 1e-30)); q = float(np.mean((b - f * a) ** 2)); return f, q


def stationary_covariance(f: np.ndarray, q: np.ndarray) -> np.ndarray:
    f = np.asarray(f, dtype=float); q = np.asarray(q, dtype=float)
    if np.max(np.abs(f)) >= 1: raise ValueError("unstable AR coefficients")
    return np.diag(q / np.maximum(1 - f * f, 1e-12))


def pca_fit(values: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=float); mean = x.mean(axis=0); xc = x - mean
    u, s, vt = np.linalg.svd(xc, full_matrices=False); basis = vt[:rank].T
    var = s * s / max(len(x) - 1, 1); explained = np.cumsum(var) / max(var.sum(), 1e-30)
    return mean, basis, s, explained


def acf_values(series: np.ndarray, lags=(1, 2, 3, 5, 10)) -> dict[int, float]:
    x = np.asarray(series, dtype=float).ravel(); x = x - x.mean(); den = float(x @ x)
    return {int(k): float(x[:-k] @ x[k:] / den) if den > 0 and k < len(x) else 0.0 for k in lags}


def augmented_state_dimension(physical_dim: int, rank: int) -> int:
    return int(physical_dim + rank)
