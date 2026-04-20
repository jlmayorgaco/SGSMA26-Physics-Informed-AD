"""Residual/noise statistics helpers for raw PMU profiling."""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew

from src.calibration.profile_schema import RESIDUAL_SAMPLE_LIMIT


def robust_sigma(x: np.ndarray) -> float:
    """Estimate robust sigma via MAD with std fallback."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 0.0
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    sigma = 1.4826 * mad
    return float(sigma if sigma > 1e-12 else np.std(x))


def autocorr(x: np.ndarray, lag: int) -> float:
    """Return lag autocorrelation with low-variance and short-series guards."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) <= lag or np.std(x) < 1e-15:
        return 0.0
    return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])


def quantile_summary(x: np.ndarray) -> dict:
    """Return p01/p05/p50/p95/p99 summary."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {f"p{p:02d}": 0.0 for p in [1, 5, 50, 95, 99]}
    vals = np.percentile(x, [1, 5, 50, 95, 99])
    return {name: float(val) for name, val in zip(["p01", "p05", "p50", "p95", "p99"], vals)}


def choose_distribution_type(residual: np.ndarray, family: str) -> str:
    """Choose fitted residual distribution family using legacy heuristics."""
    residual = residual[np.isfinite(residual)]
    if len(residual) < 32:
        return "gaussian"
    sk = abs(float(skew(residual, bias=False)))
    ku = float(kurtosis(residual, fisher=True, bias=False))
    tail_ratio = np.quantile(np.abs(residual), 0.99) / max(np.quantile(np.abs(residual), 0.75), 1e-12)
    if family in {"current_mag", "frequency"} and (ku > 4.0 or tail_ratio > 4.0):
        return "bootstrap"
    if ku > 2.0:
        return "student_t"
    if sk > 0.75:
        return "gmm"
    return "gaussian"


def sample_residuals(residual: np.ndarray, limit: int = RESIDUAL_SAMPLE_LIMIT) -> list[float]:
    """Sample deterministic residual subset up to `limit` entries."""
    residual = np.asarray(residual, dtype=float)
    residual = residual[np.isfinite(residual)]
    if len(residual) == 0:
        return []
    if len(residual) <= limit:
        sample = residual
    else:
        idx = np.linspace(0, len(residual) - 1, limit).astype(int)
        sample = residual[idx]
    return [float(x) for x in sample]
