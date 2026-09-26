"""Metrics and scoring helpers for m3 raw event-0 calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import ks_2samp


def wrap_deg(x):
    return ((np.asarray(x, dtype=float) + 180.0) % 360.0) - 180.0


def quality_bucket(ks):
    if not np.isfinite(ks):
        return "invalid_mapping"
    if ks < 0.03:
        return "very_good"
    if ks < 0.08:
        return "usable"
    if ks < 0.15:
        return "needs_tuning"
    return "poor"


def signal_family_from_key(key):
    if key.endswith("_mag") and key[0] == "V":
        return "voltage_mag"
    if key.endswith("_mag") and key[0] == "I":
        return "current_mag"
    if key == "Frequency":
        return "frequency"
    if key == "ROCOF":
        return "frequency"
    if key.endswith("_ang_delta"):
        return "angle_voltage"
    if key.endswith("_ang"):
        return "angle_current"
    return "other"


def autocorr(x, lag):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) <= lag or np.std(x) < 1e-15:
        return 0.0
    return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])


def sample_rate_from_t(t):
    t = np.asarray(t, dtype=float)
    if len(t) < 3:
        return 30.0
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return 30.0
    return float(1.0 / np.median(dt))


def spectral_features(x, fs=30.0):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 8 or np.std(x) < 1e-15:
        return {
            "lowfreq_power": 0.0,
            "total_power": 0.0,
            "lowfreq_power_ratio": 0.0,
            "spectral_centroid_hz": 0.0,
            "dominant_frequency_hz": 0.0,
        }
    centered = x - np.mean(x)
    nperseg = min(512, len(centered))
    freqs, psd = welch(centered, fs=fs, nperseg=nperseg, detrend="constant")
    total = float(np.trapezoid(psd, freqs)) if len(freqs) > 1 else float(np.sum(psd))
    low_mask = (freqs > 0.0) & (freqs <= 0.25)
    low = float(np.trapezoid(psd[low_mask], freqs[low_mask])) if np.any(low_mask) else 0.0
    centroid = float(np.sum(freqs * psd) / max(np.sum(psd), 1e-30))
    fft_freq = np.fft.rfftfreq(len(centered), d=1.0 / fs)
    fft_mag = np.abs(np.fft.rfft(centered)) / max(len(centered), 1)
    dominant_idx = int(np.argmax(fft_mag[1:]) + 1) if len(fft_mag) > 1 else 0
    return {
        "lowfreq_power": low,
        "total_power": total,
        "lowfreq_power_ratio": float(low / max(total, 1e-30)),
        "spectral_centroid_hz": centroid,
        "dominant_frequency_hz": float(fft_freq[dominant_idx]) if dominant_idx else 0.0,
    }


def spectral_mismatch(real, sim, fs=30.0):
    r = spectral_features(real, fs)
    s = spectral_features(sim, fs)
    low_mismatch = abs(s["lowfreq_power"] - r["lowfreq_power"]) / max(r["lowfreq_power"], np.std(real) ** 2, 1e-12)
    centroid_error = abs(s["spectral_centroid_hz"] - r["spectral_centroid_hz"]) / max(r["spectral_centroid_hz"], 1e-6)
    dominant_error = abs(s["dominant_frequency_hz"] - r["dominant_frequency_hz"]) / max(r["dominant_frequency_hz"], 1e-6)
    return {
        "psd_lowfreq_mismatch": float(min(low_mismatch, 10.0)),
        "spectral_centroid_error": float(min(centroid_error, 10.0)),
        "dominant_freq_error": float(min(dominant_error, 10.0)),
        "real_lowfreq_power_ratio": r["lowfreq_power_ratio"],
        "sim_lowfreq_power_ratio": s["lowfreq_power_ratio"],
        "real_spectral_centroid_hz": r["spectral_centroid_hz"],
        "sim_spectral_centroid_hz": s["spectral_centroid_hz"],
        "real_dominant_frequency_hz": r["dominant_frequency_hz"],
        "sim_dominant_frequency_hz": s["dominant_frequency_hz"],
    }


def percentile_mismatch(real, sim):
    real_p = np.percentile(real, [1, 5, 50, 95, 99])
    sim_p = np.percentile(sim, [1, 5, 50, 95, 99])
    denom = max(float(np.std(real)), float(np.median(np.abs(real_p))), 1e-9)
    return float(np.mean(np.abs(real_p - sim_p) / denom))


def composite_score(
    ks,
    rel_mean,
    rel_std,
    p_err,
    family,
    psd_lowfreq_mismatch=0.0,
    spectral_centroid_error=0.0,
    dominant_freq_error=0.0,
    chunk_stability_penalty=0.0,
):
    psd = min(float(psd_lowfreq_mismatch), 10.0)
    centroid = min(float(spectral_centroid_error), 10.0)
    dominant = min(float(dominant_freq_error), 10.0)
    stability = min(float(chunk_stability_penalty), 1.0)
    if family == "current_mag":
        return float(
            0.30 * ks + 0.06 * rel_mean + 0.24 * rel_std + 0.18 * p_err
            + 0.12 * psd + 0.05 * centroid + 0.02 * dominant + 0.03 * stability
        )
    if family == "frequency":
        return float(
            0.34 * ks + 0.10 * rel_mean + 0.18 * rel_std + 0.12 * p_err
            + 0.14 * psd + 0.08 * centroid + 0.02 * dominant + 0.02 * stability
        )
    return float(0.55 * ks + 0.15 * rel_mean + 0.15 * rel_std + 0.15 * p_err + 0.02 * stability)


def compute_basic_metrics(real, sim, family="current_mag", fs=30.0, chunk_penalty=0.0):
    real = np.asarray(real, dtype=float)
    sim = np.asarray(sim, dtype=float)
    real = real[np.isfinite(real)]
    sim = sim[np.isfinite(sim)]
    if len(real) == 0 or len(sim) == 0:
        return {
            "ks_stat": np.nan,
            "relative_mean_error": np.inf,
            "relative_std_error": np.inf,
            "percentile_error": np.inf,
            "psd_lowfreq_mismatch": np.inf,
            "spectral_centroid_error": np.inf,
            "dominant_freq_error": np.inf,
            "chunk_stability_penalty": np.inf,
            "composite_score": np.inf,
        }
    ks = float(ks_2samp(real, sim).statistic)
    rel_std = float(abs(np.std(sim) - np.std(real)) / (abs(np.std(real)) + 1e-12))
    rel_mean = float(abs(np.mean(sim) - np.mean(real)) / (abs(np.mean(real)) + 1e-12))
    p_err = percentile_mismatch(real, sim)
    spec = spectral_mismatch(real, sim, fs) if family in {"current_mag", "frequency"} else {
        "psd_lowfreq_mismatch": 0.0,
        "spectral_centroid_error": 0.0,
        "dominant_freq_error": 0.0,
    }
    return {
        "ks_stat": ks,
        "relative_mean_error": rel_mean,
        "relative_std_error": rel_std,
        "percentile_error": p_err,
        "psd_lowfreq_mismatch": spec["psd_lowfreq_mismatch"],
        "spectral_centroid_error": spec["spectral_centroid_error"],
        "dominant_freq_error": spec["dominant_freq_error"],
        "chunk_stability_penalty": float(chunk_penalty),
        "composite_score": composite_score(
            ks,
            rel_mean,
            rel_std,
            p_err,
            family,
            spec["psd_lowfreq_mismatch"],
            spec["spectral_centroid_error"],
            spec["dominant_freq_error"],
            chunk_penalty,
        ),
    }


def chunk_stability_penalty(chunks, bus_id, suffix, sim, spec):
    from src.calibration.raw_chunk_loader import raw_col, transform_real_for_spec

    vals = []
    for chunk in chunks:
        col = raw_col(bus_id, suffix)
        if col not in chunk["df"].columns:
            continue
        real = pd.to_numeric(chunk["df"][col], errors="coerce").dropna().to_numpy(float)
        real = transform_real_for_spec(real, spec)
        if len(real) > 0 and len(sim) > 0:
            vals.append(float(ks_2samp(real, sim).statistic))
    return float(np.std(vals)) if len(vals) > 1 else 0.0


def full_metrics(real, sim_clean, sim_drifted, sim_noisy, family, fs=30.0, chunk_stability=0.0):
    real = np.asarray(real, dtype=float)
    real = real[np.isfinite(real)]
    sim_clean = np.asarray(sim_clean, dtype=float)
    sim_drifted = np.asarray(sim_drifted, dtype=float)
    sim_noisy = np.asarray(sim_noisy, dtype=float)
    if len(real) == 0 or len(sim_noisy) == 0:
        return None
    ks, p = ks_2samp(real, sim_noisy)
    mean_real = float(np.mean(real))
    mean_clean = float(np.mean(sim_clean))
    mean_drifted = float(np.mean(sim_drifted))
    mean_noisy = float(np.mean(sim_noisy))
    std_real = float(np.std(real))
    std_clean = float(np.std(sim_clean))
    std_drifted = float(np.std(sim_drifted))
    std_noisy = float(np.std(sim_noisy))
    rel_mean = float(abs(mean_noisy - mean_real) / (abs(mean_real) + 1e-12))
    rel_std = float(abs(std_noisy - std_real) / (abs(std_real) + 1e-12))
    p_err = percentile_mismatch(real, sim_noisy)
    spec = spectral_mismatch(real, sim_noisy, fs) if family in {"current_mag", "frequency"} else {
        "psd_lowfreq_mismatch": 0.0,
        "spectral_centroid_error": 0.0,
        "dominant_freq_error": 0.0,
        "real_lowfreq_power_ratio": 0.0,
        "sim_lowfreq_power_ratio": 0.0,
        "real_spectral_centroid_hz": 0.0,
        "sim_spectral_centroid_hz": 0.0,
        "real_dominant_frequency_hz": 0.0,
        "sim_dominant_frequency_hz": 0.0,
    }
    score = composite_score(
        float(ks),
        rel_mean,
        rel_std,
        p_err,
        family,
        spec["psd_lowfreq_mismatch"],
        spec["spectral_centroid_error"],
        spec["dominant_freq_error"],
        chunk_stability,
    )
    out = {
        "ks_stat": float(ks),
        "ks_pvalue": float(p),
        "mean_real": mean_real,
        "mean_sim_clean": mean_clean,
        "mean_sim_drifted": mean_drifted,
        "mean_sim_noisy": mean_noisy,
        "std_real": std_real,
        "std_sim_clean": std_clean,
        "std_sim_drifted": std_drifted,
        "std_sim_noisy": std_noisy,
        "relative_mean_error": rel_mean,
        "relative_std_error": rel_std,
        "percentile_error": p_err,
        "autocorr_lag1_real": autocorr(real, 1),
        "autocorr_lag1_sim_noisy": autocorr(sim_noisy, 1),
        "autocorr_lag5_real": autocorr(real, 5),
        "autocorr_lag5_sim_noisy": autocorr(sim_noisy, 5),
        "psd_lowfreq_mismatch": spec["psd_lowfreq_mismatch"],
        "spectral_centroid_error": spec["spectral_centroid_error"],
        "dominant_freq_error": spec["dominant_freq_error"],
        "real_lowfreq_power_ratio": spec["real_lowfreq_power_ratio"],
        "sim_lowfreq_power_ratio": spec["sim_lowfreq_power_ratio"],
        "real_spectral_centroid_hz": spec["real_spectral_centroid_hz"],
        "sim_spectral_centroid_hz": spec["sim_spectral_centroid_hz"],
        "real_dominant_frequency_hz": spec["real_dominant_frequency_hz"],
        "sim_dominant_frequency_hz": spec["sim_dominant_frequency_hz"],
        "chunk_stability_penalty": float(chunk_stability),
        "composite_score": score,
        "num_samples_real": int(len(real)),
        "num_samples_sim": int(len(sim_noisy)),
        "quality_bucket": quality_bucket(float(ks)),
    }
    for label, arr in [("real", real), ("sim_clean", sim_clean), ("sim_drifted", sim_drifted), ("sim_noisy", sim_noisy)]:
        pcts = np.percentile(arr, [1, 5, 50, 95, 99])
        for name, val in zip(["p01", "p05", "p50", "p95", "p99"], pcts):
            out[f"{name}_{label}"] = float(val)
    return out
