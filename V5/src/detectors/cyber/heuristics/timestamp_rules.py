from __future__ import annotations

import numpy as np


def timestamp_evidence(timestamps: np.ndarray) -> dict[str, np.ndarray]:
    ts = np.asarray(timestamps, dtype=float)
    n = len(ts)
    if n == 0:
        zeros = np.zeros((0,), dtype=float)
        return {"jitter": zeros, "irregular_flag": zeros, "score": zeros}
    if n == 1:
        return {"jitter": np.zeros((1,), dtype=float), "irregular_flag": np.zeros((1,), dtype=float), "score": np.zeros((1,), dtype=float)}
    diffs = np.diff(ts, prepend=ts[0])
    median_dt = float(np.median(diffs[1:])) if n > 1 else 0.0
    jitter = np.abs(diffs - median_dt)
    scale = max(abs(median_dt), 1e-6)
    irregular_flag = (jitter > 0.25 * scale).astype(float)
    score = np.clip(jitter / max(0.5 * scale, 1e-6), 0.0, 1.0)
    return {"jitter": jitter, "irregular_flag": irregular_flag, "score": score}

