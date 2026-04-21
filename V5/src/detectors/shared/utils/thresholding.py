from __future__ import annotations

import numpy as np


def apply_binary_threshold(probabilities: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=float)
    return (probs >= float(threshold)).astype(int)

