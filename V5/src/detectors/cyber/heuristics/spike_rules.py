from __future__ import annotations

import numpy as np


def spike_evidence(windows: np.ndarray, *, z_threshold: float = 4.5) -> dict[str, np.ndarray]:
    n = windows.shape[0]
    if n == 0:
        zeros = np.zeros((0,), dtype=float)
        return {"spike_ratio": zeros, "max_abs_z": zeros, "score": zeros}
    mean = windows.mean(axis=1, keepdims=True)
    std = windows.std(axis=1, keepdims=True)
    std = np.where(std < 1e-6, 1e-6, std)
    z = (windows - mean) / std
    spike_ratio = (np.abs(z) > z_threshold).mean(axis=(1, 2))
    max_abs_z = np.max(np.abs(z), axis=(1, 2))
    score = np.clip(0.7 * np.minimum(spike_ratio / 0.10, 1.0) + 0.3 * np.minimum(max_abs_z / 10.0, 1.0), 0.0, 1.0)
    return {
        "spike_ratio": spike_ratio,
        "max_abs_z": max_abs_z,
        "score": score,
    }

