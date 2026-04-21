from __future__ import annotations

import numpy as np


def voltage_sag_evidence(windows: np.ndarray, feature_names: list[str]) -> dict[str, np.ndarray]:
    n = windows.shape[0]
    if n == 0:
        zeros = np.zeros((0,), dtype=float)
        return {"sag_ratio": zeros, "sag_depth": zeros, "score": zeros}
    mag_idx = [i for i, name in enumerate(feature_names) if name.endswith("_MAG") and "_V" in name]
    if not mag_idx:
        zeros = np.zeros((n,), dtype=float)
        return {"sag_ratio": zeros, "sag_depth": zeros, "score": zeros}
    voltage = windows[:, :, mag_idx]
    mins = voltage.min(axis=1)
    means = voltage.mean(axis=1)
    sag_depth = np.maximum(0.0, means - mins).mean(axis=1)
    sag_ratio = (voltage < (voltage.mean(axis=1, keepdims=True) - voltage.std(axis=1, keepdims=True))).mean(axis=(1, 2))
    score = np.clip(0.5 * np.minimum(sag_ratio / 0.20, 1.0) + 0.5 * np.minimum(sag_depth / 2.0, 1.0), 0.0, 1.0)
    return {"sag_ratio": sag_ratio, "sag_depth": sag_depth, "score": score}

