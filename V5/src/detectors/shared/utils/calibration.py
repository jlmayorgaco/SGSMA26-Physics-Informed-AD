from __future__ import annotations

import numpy as np


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_prob, dtype=float)
    if len(yt) == 0:
        return 0.0
    return float(np.mean((yp - yt) ** 2))


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_prob, dtype=float)
    if len(yt) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    ece = 0.0
    for i in range(len(edges) - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i == len(edges) - 2:
            mask = (yp >= lo) & (yp <= hi)
        else:
            mask = (yp >= lo) & (yp < hi)
        if not np.any(mask):
            continue
        conf = float(np.mean(yp[mask]))
        acc = float(np.mean(yt[mask]))
        ece += abs(conf - acc) * float(np.mean(mask))
    return float(ece)

