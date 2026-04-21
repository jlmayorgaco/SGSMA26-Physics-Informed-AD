from __future__ import annotations

import numpy as np


def apply_hysteresis_thresholds(probabilities: np.ndarray, start_threshold: float, stop_threshold: float) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=float)
    state = 0
    out = np.zeros((len(probs),), dtype=int)
    for i, value in enumerate(probs):
        if state == 0 and value >= start_threshold:
            state = 1
        elif state == 1 and value <= stop_threshold:
            state = 0
        out[i] = state
    return out

