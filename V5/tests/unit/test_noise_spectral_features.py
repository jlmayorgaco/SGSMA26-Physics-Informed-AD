from __future__ import annotations

import numpy as np

from src.calibration.spectral_features import infer_sample_rate_hz, spectral_summary


def test_infer_sample_rate_hz_none_returns_30() -> None:
    assert infer_sample_rate_hz(None) == 30.0


def test_infer_sample_rate_hz_short_returns_30() -> None:
    assert infer_sample_rate_hz(np.array([0.0, 0.033])) == 30.0


def test_infer_sample_rate_hz_valid_timestamps() -> None:
    fs = infer_sample_rate_hz(np.array([0.0, 0.0333333, 0.0666666, 0.1]))
    assert np.isclose(fs, 30.0, atol=0.5)


def test_spectral_summary_short_signal_returns_empty_structure() -> None:
    out = spectral_summary(np.array([1.0, 2.0, 3.0]), fs=30.0)
    assert out["total_power"] == 0.0
    assert "top_frequencies_hz" in out


def test_spectral_summary_constant_signal_returns_zero_structure() -> None:
    out = spectral_summary(np.ones(30), fs=30.0)
    assert out["lowfreq_power_ratio"] == 0.0
    assert out["dominant_frequency_hz"] == 0.0


def test_spectral_summary_valid_signal_returns_expected_keys() -> None:
    t = np.arange(0, 2, 1 / 30.0)
    x = np.sin(2 * np.pi * 1.0 * t)
    out = spectral_summary(x, fs=30.0)
    expected = {
        "sample_rate_hz",
        "lowfreq_power",
        "total_power",
        "lowfreq_power_ratio",
        "spectral_centroid_hz",
        "dominant_frequency_hz",
        "dominant_amplitude",
        "top_frequencies_hz",
        "top_amplitudes",
    }
    assert expected.issubset(set(out.keys()))
