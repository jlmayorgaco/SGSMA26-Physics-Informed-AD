from __future__ import annotations

import numpy as np

from src.detectors.configs import PostprocessingConfig


def enforce_event_state_machine(binary: np.ndarray, config: PostprocessingConfig) -> np.ndarray:
    if len(binary) == 0:
        return binary.astype(int)
    out = binary.astype(int).copy()
    # Remove very short ON islands.
    i = 0
    while i < len(out):
        if out[i] == 1:
            j = i
            while j < len(out) and out[j] == 1:
                j += 1
            if (j - i) < config.min_on_frames:
                out[i:j] = 0
            i = j
        else:
            i += 1
    # Fill very short OFF gaps.
    i = 0
    while i < len(out):
        if out[i] == 0:
            j = i
            while j < len(out) and out[j] == 0:
                j += 1
            left_on = i > 0 and out[i - 1] == 1
            right_on = j < len(out) and out[j] == 1
            if left_on and right_on and (j - i) < config.min_off_frames:
                out[i:j] = 1
            i = j
        else:
            i += 1
    return out

