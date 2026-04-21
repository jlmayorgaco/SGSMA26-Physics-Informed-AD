from __future__ import annotations

import numpy as np

from src.calibration.drift_synthesis import (
    apply_operating_drift,
    estimate_drift_targets_from_real,
    estimate_real_spectral_targets,
    rank_shape_to_empirical,
    synthesize_low_frequency_drift_from_fft,
)


def test_estimate_drift_targets_from_real_returns_expected_keys() -> None:
    real = np.sin(np.linspace(0, 10, 300))
    out = estimate_drift_targets_from_real(real, {"std_dev_abs": 0.1}, "current_mag")
    assert {"total_std", "noise_std", "drift_std", "spectral"}.issubset(out.keys())


def test_estimate_real_spectral_targets_returns_expected_keys() -> None:
    out = estimate_real_spectral_targets(np.sin(np.linspace(0, 10, 300)))
    assert "top_frequencies_hz" in out


def test_synthesize_low_frequency_drift_from_fft_same_length() -> None:
    t = np.linspace(0, 60, 1800)
    target = {"spectral": {"top_frequencies_hz": [0.1], "top_amplitudes": [1.0], "lowfreq_power_ratio": 0.5}}
    y = synthesize_low_frequency_drift_from_fft(t, target, np.random.default_rng(0), 0.1)
    assert len(y) == len(t)


def test_rank_shape_to_empirical_preserves_length() -> None:
    source = np.random.default_rng(0).normal(size=100)
    empirical = np.random.default_rng(1).normal(size=200)
    out = rank_shape_to_empirical(source, empirical)
    assert len(out) == len(source)


def test_apply_operating_drift_same_length() -> None:
    t = np.linspace(0, 60, 1800)
    x = np.sin(np.linspace(0, 10, 1800))
    target = estimate_drift_targets_from_real(x, {"std_dev_abs": 0.05}, "current_mag")
    out = apply_operating_drift(x, t, "current_mag", "10", target, np.random.default_rng(2))
    assert len(out) == len(x)


def test_apply_operating_drift_noop_when_none_mode_or_zero_std() -> None:
    x = np.arange(20, dtype=float)
    t = np.arange(20, dtype=float) / 30.0
    out = apply_operating_drift(x, t, "other", "10", {"drift_std": 0.0}, np.random.default_rng(0), drift_mode="none")
    assert np.allclose(out, x)
