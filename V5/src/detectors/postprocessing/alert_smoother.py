from __future__ import annotations

import numpy as np


def exponential_smooth(probabilities: np.ndarray, alpha: float | np.ndarray = 0.35) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=float)
    if len(probs) == 0:
        return probs.copy()
    if np.isscalar(alpha):
        alpha_values = np.full_like(probs, float(alpha), dtype=float)
    else:
        alpha_values = np.asarray(alpha, dtype=float)
        if alpha_values.shape != probs.shape:
            raise ValueError("alpha array must match probabilities shape")
    alpha_values = np.clip(alpha_values, 0.0, 1.0)
    out = np.zeros_like(probs, dtype=float)
    out[0] = probs[0]
    for i in range(1, len(probs)):
        a = float(alpha_values[i])
        out[i] = a * probs[i] + (1.0 - a) * out[i - 1]
    return np.clip(out, 0.0, 1.0)
