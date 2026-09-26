"""Operating drift synthesis helpers for m3 raw calibration."""

from __future__ import annotations

import math

import numpy as np

from src.calibration.calibration_metrics import autocorr, spectral_features


def estimate_drift_targets_from_real(real, noise_stats, family):
    real = np.asarray(real, dtype=float)
    real = real[np.isfinite(real)]
    total_std = float(np.std(real)) if len(real) else 0.0
    noise_std = float(noise_stats.get("std_dev_abs", noise_stats.get("std_dev_raw", 0.0)))
    if family in {"voltage_mag", "current_mag", "frequency"}:
        drift_std = math.sqrt(max(total_std**2 - min(noise_std, 0.55 * total_std) ** 2, 0.0))
        drift_std = max(drift_std, 0.25 * total_std)
    elif family == "angle_voltage":
        drift_std = max(0.35 * total_std, math.sqrt(max(total_std**2 - min(noise_std, 0.45 * total_std) ** 2, 0.0)))
    else:
        drift_std = 0.0
    empirical_deviation = real - float(np.median(real)) if len(real) else np.array([], dtype=float)
    spectral = estimate_real_spectral_targets(real, noise_stats, fs=30.0)
    return {
        "total_std": total_std,
        "noise_std": noise_std,
        "drift_std": float(drift_std),
        "empirical_deviation": empirical_deviation,
        "lag1": autocorr(real, 1),
        "lag5": autocorr(real, 5),
        "spectral": spectral,
    }


def estimate_real_spectral_targets(real, noise_stats=None, fs=30.0):
    real_spec = spectral_features(real, fs)
    profile_spec = {}
    if isinstance(noise_stats, dict):
        profile_spec = noise_stats.get("psd_summary", {}) or noise_stats.get("fft_low_freq_summary", {}) or {}
    top_freqs = profile_spec.get("top_frequencies_hz", []) if isinstance(profile_spec, dict) else []
    top_amps = profile_spec.get("top_amplitudes", []) if isinstance(profile_spec, dict) else []
    if not top_freqs:
        centered = np.asarray(real, dtype=float) - float(np.nanmean(real))
        if len(centered) > 8:
            freqs = np.fft.rfftfreq(len(centered), d=1.0 / fs)
            amps = np.abs(np.fft.rfft(centered)) / max(len(centered), 1)
            order = np.argsort(amps[1:])[-3:][::-1] + 1
            top_freqs = [float(freqs[i]) for i in order if freqs[i] <= 0.5]
            top_amps = [float(amps[i]) for i in order if freqs[i] <= 0.5]
    return {
        **real_spec,
        "top_frequencies_hz": [float(f) for f in top_freqs[:3] if float(f) > 0.0],
        "top_amplitudes": [float(a) for a in top_amps[:3]],
    }


def synthesize_low_frequency_drift_from_fft(t, target_stats, rng, drift_std):
    spectral = target_stats.get("spectral", {}) or {}
    freqs = spectral.get("top_frequencies_hz", []) or []
    amps = spectral.get("top_amplitudes", []) or []
    t = np.asarray(t, dtype=float)
    if len(t) == 0 or drift_std <= 0:
        return np.zeros(len(t))
    y = np.zeros(len(t))
    for freq, amp in zip(freqs[:3], amps[:3]):
        freq = float(freq)
        if freq <= 0 or freq > 0.5:
            continue
        phase = rng.uniform(0.0, 2.0 * np.pi)
        y += float(amp) * np.sin(2.0 * np.pi * freq * (t - t[0]) + phase)
    if np.std(y) < 1e-15:
        duration = max(float(t[-1] - t[0]), 1.0) if len(t) > 1 else 1.0
        y = np.sin(2.0 * np.pi * (t - t[0]) / duration + rng.uniform(0.0, 2.0 * np.pi))
    y -= np.mean(y)
    if np.std(y) > 1e-15:
        lowfreq_ratio = float(spectral.get("lowfreq_power_ratio", 0.5))
        y *= drift_std * min(max(lowfreq_ratio, 0.25), 0.95) / np.std(y)
    return y


def rank_shape_to_empirical(source, empirical_samples):
    source = np.asarray(source, dtype=float)
    empirical_samples = np.asarray(empirical_samples, dtype=float)
    empirical_samples = empirical_samples[np.isfinite(empirical_samples)]
    if len(source) == 0 or len(empirical_samples) < 8 or np.std(source) < 1e-12:
        return source
    order = np.argsort(source)
    ranks = np.empty(len(source), dtype=float)
    ranks[order] = (np.arange(len(source), dtype=float) + 0.5) / len(source)
    shaped = np.quantile(empirical_samples, ranks)
    return shaped - float(np.mean(shaped))


def apply_operating_drift(signal, t, family, bus_id, target_stats, rng, drift_scale=1.0, drift_mode="fft_residual_synthesis"):
    signal = np.asarray(signal, dtype=float)
    n = len(signal)
    if n == 0 or drift_mode == "none":
        return signal.copy()
    drift_std = float(target_stats.get("drift_std", 0.0)) * float(drift_scale)
    if drift_std <= 0:
        return signal.copy()
    rho_by_family = {"voltage_mag": 0.996, "current_mag": 0.994, "frequency": 0.998, "angle_voltage": 0.997}
    rho = rho_by_family.get(family, 0.995)
    white_std = drift_std * math.sqrt(max(0.0, 1.0 - rho**2))
    ar = np.zeros(n)
    white = rng.normal(0.0, white_std, n)
    ar[0] = white[0]
    for i in range(1, n):
        ar[i] = rho * ar[i - 1] + white[i]

    if drift_mode == "fft_residual_synthesis":
        low = synthesize_low_frequency_drift_from_fft(t, target_stats, rng, drift_std)
    else:
        duration = max(float(t[-1] - t[0]), 1.0) if len(t) > 1 else 1.0
        phase = rng.uniform(0, 2 * np.pi)
        low = 0.35 * drift_std * np.sin(2 * np.pi * t / duration + phase)
    drift = 0.65 * ar + low
    empirical = target_stats.get("empirical_deviation")
    if drift_mode == "fft_residual_synthesis" and empirical is not None and family in {"current_mag", "frequency"}:
        drift = rank_shape_to_empirical(drift, np.asarray(empirical, dtype=float) * float(drift_scale))
    if np.std(drift) > 1e-12:
        drift *= drift_std / np.std(drift)
    _ = bus_id
    return signal + drift
