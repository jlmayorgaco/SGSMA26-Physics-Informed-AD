from __future__ import annotations

import numpy as np

from src.calibration.residual_models import autocorr, quantile_summary, robust_sigma, sample_residuals


def test_robust_sigma_empty_returns_zero() -> None:
    assert robust_sigma(np.array([])) == 0.0


def test_robust_sigma_constant_series_returns_zero_or_std_fallback() -> None:
    val = robust_sigma(np.ones(20))
    assert np.isclose(val, 0.0)


def test_autocorr_short_series_returns_zero() -> None:
    assert autocorr(np.array([1.0, 2.0]), lag=5) == 0.0


def test_autocorr_constant_series_returns_zero() -> None:
    assert autocorr(np.ones(50), lag=1) == 0.0


def test_quantile_summary_empty_returns_zero_quantiles() -> None:
    qs = quantile_summary(np.array([]))
    assert qs == {"p01": 0.0, "p05": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}


def test_sample_residuals_empty_returns_empty() -> None:
    assert sample_residuals(np.array([])) == []


def test_sample_residuals_respects_limit() -> None:
    x = np.arange(5000, dtype=float)
    out = sample_residuals(x, limit=100)
    assert len(out) == 100
