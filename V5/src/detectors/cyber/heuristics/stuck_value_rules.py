from __future__ import annotations

import numpy as np


def stuck_value_evidence(
    windows: np.ndarray,
    *,
    std_threshold: float = 1e-4,
    low_delta_threshold: float = 1e-4,
) -> dict[str, np.ndarray]:
    n = windows.shape[0]
    if n == 0:
        zeros = np.zeros((0,), dtype=float)
        return {"stuck_feature_fraction": zeros, "low_delta_fraction": zeros, "score": zeros}
    std_per_feature = windows.std(axis=1)
    stuck_feature_fraction = (std_per_feature < std_threshold).mean(axis=1)
    deltas = np.diff(windows, axis=1, prepend=windows[:, :1, :])
    low_delta_fraction = (np.abs(deltas) < low_delta_threshold).mean(axis=(1, 2))
    score = np.clip(0.6 * stuck_feature_fraction + 0.4 * low_delta_fraction, 0.0, 1.0)
    return {
        "stuck_feature_fraction": stuck_feature_fraction,
        "low_delta_fraction": low_delta_fraction,
        "score": score,
    }

