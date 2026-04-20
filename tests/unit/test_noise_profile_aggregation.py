from __future__ import annotations

import numpy as np

from src.calibration.profile_aggregation import aggregate_entries, merge_spectral, weighted_average


def _entry(sample_count: int) -> dict:
    return {
        "eda_stats": {"mean": 1.0, "std": 0.5, "median": 1.0, "min": 0.0, "max": 2.0, "n_samples": 10, "p01": 0.1, "p05": 0.2, "p50": 1.0, "p95": 1.8, "p99": 1.9},
        "noise_model": {
            "std_dev_abs": 0.1,
            "std_dev_raw": 0.1,
            "ar1_rho": 0.2,
            "ar5_rho": 0.1,
            "p_outlier": 0.05,
            "outlier_mag_abs": 0.3,
            "outlier_threshold_abs": 0.8,
            "skewness": 0.0,
            "kurtosis": 1.0,
            "fitted_distribution_type": "gaussian",
            "signal_family": "voltage_mag",
            "psd_summary": {
                "sample_rate_hz": 30.0,
                "lowfreq_power": 0.1,
                "total_power": 1.0,
                "lowfreq_power_ratio": 0.1,
                "spectral_centroid_hz": 1.0,
                "dominant_frequency_hz": 0.5,
                "dominant_amplitude": 0.2,
                "top_frequencies_hz": [0.5],
                "top_amplitudes": [0.2],
            },
            "fft_low_freq_summary": {
                "sample_rate_hz": 30.0,
                "lowfreq_power": 0.1,
                "total_power": 1.0,
                "lowfreq_power_ratio": 0.1,
                "spectral_centroid_hz": 1.0,
                "dominant_frequency_hz": 0.5,
                "dominant_amplitude": 0.2,
                "top_frequencies_hz": [0.5],
                "top_amplitudes": [0.2],
            },
            "quantile_summary": {"p01": -0.2, "p05": -0.1, "p50": 0.0, "p95": 0.1, "p99": 0.2},
            "residual_samples": [0.1, -0.1],
            "profile_source": "raw_event_chunk",
            "trend_method": "savgol_w11_p3",
        },
        "diagnostics": {
            "residual_mean": 0.0,
            "residual_std": 0.1,
            "residual_skew": 0.0,
            "residual_kurtosis": 1.0,
            "residual_lag1": 0.2,
            "residual_lag5": 0.1,
            "signal_psd_summary": {
                "sample_rate_hz": 30.0,
                "lowfreq_power": 0.2,
                "total_power": 2.0,
                "lowfreq_power_ratio": 0.1,
                "spectral_centroid_hz": 1.5,
                "dominant_frequency_hz": 0.7,
                "dominant_amplitude": 0.3,
                "top_frequencies_hz": [0.7],
                "top_amplitudes": [0.3],
            },
        },
        "sample_count": sample_count,
    }


def test_weighted_average_empty_returns_zero() -> None:
    assert weighted_average([], "x", np.array([])) == 0.0


def test_merge_spectral_returns_expected_keys() -> None:
    out = merge_spectral([_entry(10)["noise_model"]], "psd_summary", np.array([10.0]))
    assert "dominant_frequency_hz" in out
    assert "top_frequencies_hz" in out


def test_aggregate_entries_preserves_structure() -> None:
    out = aggregate_entries([_entry(10), _entry(20)])
    assert "eda_stats" in out
    assert "noise_model" in out
    assert "diagnostics" in out
    assert "sample_count" in out


def test_aggregate_entries_merges_residual_samples() -> None:
    out = aggregate_entries([_entry(10), _entry(20)])
    assert len(out["noise_model"]["residual_samples"]) >= 2


def test_aggregate_entries_sets_profile_source_to_raw_event_chunk_aggregate() -> None:
    out = aggregate_entries([_entry(10), _entry(20)])
    assert out["noise_model"]["profile_source"] == "raw_event_chunk_aggregate"
