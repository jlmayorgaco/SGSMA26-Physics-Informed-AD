from __future__ import annotations

import numpy as np


def spectral_summary(values: np.ndarray, fs: float) -> dict:
    """
    Low-frequency spectral summary for PMU signals.

    Notes:
    - This is intended for PMU-rate data (~30 fps), not raw 60 Hz waveform harmonics.
    - The useful interpretation is oscillatory / inter-area / low-frequency content.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    if x.size < 16 or not np.isfinite(fs) or fs <= 0:
        return {}

    x0 = x - np.mean(x)
    n = len(x0)

    # One-sided PSD-like estimate using FFT
    pxx = np.abs(np.fft.rfft(x0)) ** 2 / max(n, 1)
    f = np.fft.rfftfreq(n, d=1.0 / fs)

    if len(f) == 0:
        return {}

    if len(f) > 1:
        order = np.argsort(pxx[1:])[::-1][:5] + 1
    else:
        order = np.array([0])

    bands = {
        "0_to_0p1_hz": (0.0, 0.1),
        "0p1_to_1_hz": (0.1, 1.0),
        "1_to_5_hz": (1.0, 5.0),
        "5_to_nyquist_hz": (5.0, fs / 2.0),
    }

    band_energy: dict[str, float] = {}
    for name, (lo, hi) in bands.items():
        mask = (f >= lo) & (f < hi)
        band_energy[name] = float(np.trapezoid(pxx[mask], f[mask])) if mask.any() else 0.0

    dominant_freqs = [float(f[i]) for i in order]
    dominant_psd = [float(pxx[i]) for i in order]

    dominant_frequency_hz = dominant_freqs[0] if dominant_freqs else None
    dominant_power = dominant_psd[0] if dominant_psd else None

    total_energy = float(np.trapezoid(pxx, f)) if len(f) > 1 else float(np.sum(pxx))

    return {
        "sampling_rate_hz": float(fs),
        "n_samples": int(n),
        "dominant_frequency_hz": dominant_frequency_hz,
        "dominant_power": dominant_power,
        "dominant_frequencies_hz": dominant_freqs,
        "dominant_psd": dominant_psd,
        "band_energy": band_energy,
        "total_spectral_energy": total_energy,
    }


def band_energy_features(values: np.ndarray, fs: float) -> dict:
    """
    Convenience wrapper returning only the band-energy block.
    Useful if you want a lighter feature set.
    """
    spec = spectral_summary(values, fs)
    return spec.get("band_energy", {})


def dominant_frequency(values: np.ndarray, fs: float) -> float | None:
    """
    Convenience helper returning the strongest non-DC spectral component.
    """
    spec = spectral_summary(values, fs)
    return spec.get("dominant_frequency_hz")