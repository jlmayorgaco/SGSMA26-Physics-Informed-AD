from __future__ import annotations

import numpy as np

from src.detectors.metrics import compute_binary_metrics


def test_binary_metrics_values() -> None:
    y_true = np.array([0, 0, 1, 1, 1], dtype=int)
    y_prob = np.array([0.1, 0.4, 0.8, 0.7, 0.2], dtype=float)
    m = compute_binary_metrics(y_true, y_prob, threshold=0.5)
    assert 0.0 <= m.f1 <= 1.0
    assert m.support_positive == 3
    assert m.support_negative == 2

