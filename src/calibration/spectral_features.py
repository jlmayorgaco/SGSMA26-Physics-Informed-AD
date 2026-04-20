"""Sampling-rate and spectral summaries for raw PMU signals."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import welch


def infer_sample_rate_hz(timestamps: np.ndarray | None) -> float:
    """Infer sample rate from timestamps with 30 Hz fallback."""
    if timestamps is None:
        return 30.0
    t = pd.to_numeric(pd.Series(timestamps), errors="coerce").dropna().to_numpy(float)
    if len(t) < 3:
        return 30.0
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return 30.0
    return float(1.0 / np.median(dt))


def spectral_summary(x: np.ndarray, fs: float) -> dict:
    """Compute legacy-compatible spectral diagnostics."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 8 or np.std(x) < 1e-15:
        return {
            "sample_rate_hz": float(fs),
            "lowfreq_power": 0.0,
            "total_power": 0.0,
            "lowfreq_power_ratio": 0.0,
            "spectral_centroid_hz": 0.0,
            "dominant_frequency_hz": 0.0,
            "dominant_amplitude": 0.0,
            "top_frequencies_hz": [],
            "top_amplitudes": [],
        }

    centered = x - np.mean(x)
    nperseg = min(512, len(centered))
    freqs, psd = welch(centered, fs=fs, nperseg=nperseg, detrend="constant")
    total = float(np.trapezoid(psd, freqs)) if len(freqs) > 1 else float(np.sum(psd))
    low_mask = (freqs > 0.0) & (freqs <= 0.25)
    low = float(np.trapezoid(psd[low_mask], freqs[low_mask])) if np.any(low_mask) else 0.0
    denom = max(float(np.sum(psd)), 1e-30)
    centroid = float(np.sum(freqs * psd) / denom)

    fft_freq = np.fft.rfftfreq(len(centered), d=1.0 / fs)
    fft_mag = np.abs(np.fft.rfft(centered)) / max(len(centered), 1)
    if len(fft_mag) > 1:
        order = np.argsort(fft_mag[1:])[-5:][::-1] + 1
        dominant = int(order[0])
    else:
        order = np.array([], dtype=int)
        dominant = 0

    return {
        "sample_rate_hz": float(fs),
        "lowfreq_power": low,
        "total_power": total,
        "lowfreq_power_ratio": float(low / max(total, 1e-30)),
        "spectral_centroid_hz": centroid,
        "dominant_frequency_hz": float(fft_freq[dominant]) if dominant else 0.0,
        "dominant_amplitude": float(fft_mag[dominant]) if dominant else 0.0,
        "top_frequencies_hz": [float(fft_freq[i]) for i in order],
        "top_amplitudes": [float(fft_mag[i]) for i in order],
    }
