from __future__ import annotations

import numpy as np


def temporal_feature_summary(window: np.ndarray) -> dict[str, np.ndarray]:
    arr = np.asarray(window, dtype=float)
    if arr.ndim != 2:
        raise ValueError("window must be 2D [window_size, n_features]")
    return {
        "mean": np.mean(arr, axis=0),
        "std": np.std(arr, axis=0),
        "min": np.min(arr, axis=0),
        "max": np.max(arr, axis=0),
        "rms": np.sqrt(np.mean(arr**2, axis=0)),
    }

