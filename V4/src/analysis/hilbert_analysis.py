from __future__ import annotations

import numpy as np


def analytic_signal_fft(x: np.ndarray) -> np.ndarray:
    """
    Hilbert-style analytic signal using FFT only, avoiding scipy dependency.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return np.array([], dtype=np.complex128)

    Xf = np.fft.fft(x)
    h = np.zeros(n)

    if n % 2 == 0:
        h[0] = 1
        h[n // 2] = 1
        h[1:n // 2] = 2
    else:
        h[0] = 1
        h[1:(n + 1) // 2] = 2

    return np.fft.ifft(Xf * h)


def hilbert_features(values: np.ndarray) -> dict:
    x = np.asarray(values, dtype=float)
    mask = np.isfinite(x)

    if mask.sum() < 4:
        return {}

    x_clean = x.copy()
    if not mask.all():
        finite_idx = np.where(mask)[0]
        x_clean = np.interp(np.arange(len(x)), finite_idx, x[finite_idx])

    z = analytic_signal_fft(x_clean)
    envelope = np.abs(z)
    phase = np.angle(z)
    phase_unwrapped = np.unwrap(phase)

    return {
        "envelope_mean": float(np.mean(envelope)),
        "envelope_std": float(np.std(envelope, ddof=1)) if len(envelope) > 1 else 0.0,
        "phase_mean_rad": float(np.mean(phase)),
        "phase_std_rad": float(np.std(phase, ddof=1)) if len(phase) > 1 else 0.0,
        "phase_unwrapped_span_rad": float(np.max(phase_unwrapped) - np.min(phase_unwrapped)),
        "envelope_series": envelope,
        "phase_series_rad": phase,
        "phase_unwrapped_series_rad": phase_unwrapped,
    }