"""Aggregation helpers for multi-chunk raw noise profiles."""

from __future__ import annotations

import numpy as np

from src.calibration.profile_schema import RESIDUAL_SAMPLE_LIMIT


def weighted_average(items: list[dict], key: str, weights: np.ndarray) -> float:
    """Compute weighted mean for a numeric dict key."""
    vals = np.asarray([float(item.get(key, 0.0)) for item in items], dtype=float)
    if len(vals) == 0:
        return 0.0
    return float(np.average(vals, weights=weights)) if np.sum(weights) > 0 else float(np.mean(vals))


def merge_spectral(items: list[dict], key: str, weights: np.ndarray) -> dict:
    """Merge spectral payloads with weighted numeric keys and first available top-k lists."""
    specs = [item.get(key, {}) or {} for item in items]
    numeric = [
        "sample_rate_hz",
        "lowfreq_power",
        "total_power",
        "lowfreq_power_ratio",
        "spectral_centroid_hz",
        "dominant_frequency_hz",
        "dominant_amplitude",
    ]
    out = {name: weighted_average(specs, name, weights) for name in numeric}
    first_top_f = next((s.get("top_frequencies_hz", []) for s in specs if s.get("top_frequencies_hz")), [])
    first_top_a = next((s.get("top_amplitudes", []) for s in specs if s.get("top_amplitudes")), [])
    out["top_frequencies_hz"] = [float(x) for x in first_top_f]
    out["top_amplitudes"] = [float(x) for x in first_top_a]
    return out


def aggregate_entries(entries: list[dict]) -> dict:
    """Aggregate per-chunk profiles into one per event/bus/signal profile."""
    weights = np.asarray([max(1, e.get("sample_count", 1)) for e in entries], dtype=float)
    eda_keys = entries[0]["eda_stats"].keys()
    diag_keys = [k for k in entries[0]["diagnostics"].keys() if k != "signal_psd_summary"]
    noise_keys = [
        "std_dev_abs",
        "std_dev_raw",
        "ar1_rho",
        "ar5_rho",
        "p_outlier",
        "outlier_mag_abs",
        "outlier_threshold_abs",
        "skewness",
        "kurtosis",
    ]

    eda = {k: weighted_average([e["eda_stats"] for e in entries], k, weights) for k in eda_keys}
    diag = {k: weighted_average([e["diagnostics"] for e in entries], k, weights) for k in diag_keys}
    diag["signal_psd_summary"] = merge_spectral([e["diagnostics"] for e in entries], "signal_psd_summary", weights)

    residual_samples: list[float] = []
    for entry in entries:
        residual_samples.extend(entry.get("noise_model", {}).get("residual_samples", []))
    if len(residual_samples) > RESIDUAL_SAMPLE_LIMIT:
        idx = np.linspace(0, len(residual_samples) - 1, RESIDUAL_SAMPLE_LIMIT).astype(int)
        residual_samples = [float(residual_samples[i]) for i in idx]

    noise = {k: weighted_average([e["noise_model"] for e in entries], k, weights) for k in noise_keys}
    families = [e["noise_model"]["signal_family"] for e in entries]
    distributions = [e["noise_model"]["fitted_distribution_type"] for e in entries]
    trends = [e["noise_model"]["trend_method"] for e in entries]
    noise["signal_family"] = max(set(families), key=families.count)
    noise["fitted_distribution_type"] = max(set(distributions), key=distributions.count)
    noise["trend_method"] = max(set(trends), key=trends.count)
    noise["profile_source"] = "raw_event_chunk_aggregate"
    noise["psd_summary"] = merge_spectral([e["noise_model"] for e in entries], "psd_summary", weights)
    noise["fft_low_freq_summary"] = merge_spectral([e["noise_model"] for e in entries], "fft_low_freq_summary", weights)
    noise["quantile_summary"] = {
        k: weighted_average([e["noise_model"]["quantile_summary"] for e in entries], k, weights)
        for k in entries[0]["noise_model"]["quantile_summary"].keys()
    }
    noise["residual_samples"] = residual_samples

    profile = {"eda_stats": eda, "noise_model": noise, "diagnostics": diag, "sample_count": int(np.sum(weights))}
    profile.update(noise)
    return profile
