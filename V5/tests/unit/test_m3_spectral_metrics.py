from __future__ import annotations

import numpy as np

from src.calibration.calibration_metrics import (
    compute_basic_metrics,
    full_metrics,
    percentile_mismatch,
    quality_bucket,
    sample_rate_from_t,
    spectral_features,
    spectral_mismatch,
)


def test_sample_rate_from_t_short_returns_30() -> None:
    assert sample_rate_from_t(np.array([0.0, 0.1])) == 30.0


def test_spectral_features_short_returns_zero_structure() -> None:
    out = spectral_features(np.array([1.0, 2.0, 3.0]))
    assert out["total_power"] == 0.0


def test_spectral_features_valid_signal_returns_expected_keys() -> None:
    t = np.arange(0, 2, 1 / 30.0)
    x = np.sin(2 * np.pi * 1.0 * t)
    out = spectral_features(x)
    assert "dominant_frequency_hz" in out


def test_spectral_mismatch_returns_expected_keys() -> None:
    x = np.sin(np.linspace(0, 10, 200))
    out = spectral_mismatch(x, x * 0.9)
    assert "psd_lowfreq_mismatch" in out


def test_percentile_mismatch_finite() -> None:
    x = np.linspace(0, 1, 100)
    y = np.linspace(0.1, 1.1, 100)
    assert np.isfinite(percentile_mismatch(x, y))


def test_compute_basic_metrics_returns_expected_keys() -> None:
    x = np.linspace(0, 1, 100)
    y = np.linspace(0.1, 1.1, 100)
    out = compute_basic_metrics(x, y, family="current_mag")
    assert "composite_score" in out


def test_full_metrics_returns_expected_keys() -> None:
    x = np.linspace(0, 1, 100)
    out = full_metrics(x, x, x, x, family="current_mag")
    assert out is not None
    assert "ks_stat" in out


def test_quality_bucket_thresholds() -> None:
    assert quality_bucket(0.02) == "very_good"
    assert quality_bucket(0.2) == "poor"
