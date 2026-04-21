from __future__ import annotations

import numpy as np


def current_surge_evidence(windows: np.ndarray, feature_names: list[str]) -> dict[str, np.ndarray]:
    n = windows.shape[0]
    if n == 0:
        zeros = np.zeros((0,), dtype=float)
        return {"surge_ratio": zeros, "surge_strength": zeros, "score": zeros}
    idx = [i for i, name in enumerate(feature_names) if name.endswith("_MAG") and "_I" in name]
    if not idx:
        zeros = np.zeros((n,), dtype=float)
        return {"surge_ratio": zeros, "surge_strength": zeros, "score": zeros}
    current = windows[:, :, idx]
    mean = current.mean(axis=1, keepdims=True)
    std = np.maximum(current.std(axis=1, keepdims=True), 1e-6)
    z = (current - mean) / std
    surge_ratio = (z > 3.0).mean(axis=(1, 2))
    surge_strength = np.maximum(0.0, z.max(axis=(1, 2)) - 2.0)
    score = np.clip(0.6 * np.minimum(surge_ratio / 0.10, 1.0) + 0.4 * np.minimum(surge_strength / 4.0, 1.0), 0.0, 1.0)
    return {"surge_ratio": surge_ratio, "surge_strength": surge_strength, "score": score}

