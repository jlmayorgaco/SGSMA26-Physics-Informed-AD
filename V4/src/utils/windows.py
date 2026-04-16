from __future__ import annotations

from typing import List, Tuple

import numpy as np


def contiguous_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    if mask.size == 0:
        return []

    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []

    diff = np.diff(mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, len(mask)]

    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def seconds_to_samples(seconds: float, dt: float, minimum: int = 1) -> int:
    if not np.isfinite(seconds) or not np.isfinite(dt) or dt <= 0:
        return minimum
    return max(minimum, int(round(seconds / dt)))


def expand_window(
    start_idx: int,
    end_idx: int,
    total_len: int,
    left: int,
    right: int,
) -> tuple[int, int]:
    s = max(0, start_idx - left)
    e = min(total_len, end_idx + right)
    return s, e