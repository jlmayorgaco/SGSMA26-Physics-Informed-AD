from __future__ import annotations

import numpy as np


def freq_rocof_evidence(windows: np.ndarray, feature_names: list[str]) -> dict[str, np.ndarray]:
    n = windows.shape[0]
    if n == 0:
        zeros = np.zeros((0,), dtype=float)
        return {"freq_dev": zeros, "rocof_dev": zeros, "score": zeros}
    freq_idx = [i for i, name in enumerate(feature_names) if name.endswith("Freq")]
    rocof_idx = [i for i, name in enumerate(feature_names) if name.endswith("ROCOF")]
    freq_dev = np.zeros((n,), dtype=float)
    rocof_dev = np.zeros((n,), dtype=float)
    if freq_idx:
        freq = windows[:, :, freq_idx]
        freq_dev = np.abs(freq - freq.mean(axis=1, keepdims=True)).mean(axis=(1, 2))
    if rocof_idx:
        rocof = windows[:, :, rocof_idx]
        rocof_dev = np.abs(rocof).mean(axis=(1, 2))
    score = np.clip(0.5 * np.minimum(freq_dev / 0.2, 1.0) + 0.5 * np.minimum(rocof_dev / 0.2, 1.0), 0.0, 1.0)
    return {"freq_dev": freq_dev, "rocof_dev": rocof_dev, "score": score}

