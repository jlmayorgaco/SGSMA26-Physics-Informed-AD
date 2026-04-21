from __future__ import annotations

import numpy as np

from src.detectors.configs import PostprocessingConfig


def apply_hysteresis(scores: np.ndarray, config: PostprocessingConfig) -> np.ndarray:
    state = 0
    out = np.zeros((len(scores),), dtype=int)
    for i, s in enumerate(scores):
        if state == 0 and s >= config.start_threshold:
            state = 1
        elif state == 1 and s <= config.stop_threshold:
            state = 0
        out[i] = state
    return out

