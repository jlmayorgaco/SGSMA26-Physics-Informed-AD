from __future__ import annotations

import numpy as np


def derivative_feature_summary(window: np.ndarray) -> dict[str, np.ndarray]:
    arr = np.asarray(window, dtype=float)
    if arr.ndim != 2:
        raise ValueError("window must be 2D [window_size, n_features]")
    deriv = np.diff(arr, axis=0, prepend=arr[:1, :])
    return {
        "deriv_mean": np.mean(deriv, axis=0),
        "deriv_std": np.std(deriv, axis=0),
        "deriv_abs_mean": np.mean(np.abs(deriv), axis=0),
        "deriv_max_abs": np.max(np.abs(deriv), axis=0),
    }

