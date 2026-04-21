from __future__ import annotations

import numpy as np

from src.metrics.dynamic_state_estimation_metrics import compute_dynamic_tracking_metrics


def test_dynamic_tracking_metrics_finite() -> None:
    t = np.asarray([0.0, 0.1, 0.2, 0.3, 0.4], dtype=float)
    truth = np.ones((5, 2), dtype=complex)
    truth[:, 0] += np.asarray([0.0, 0.01, 0.02, 0.01, 0.0])
    est = truth * (1.0 + 0.001j)
    out = compute_dynamic_tracking_metrics(timestamps=t, est=est, truth=truth, selected_bus_indices=[0, 1])
    assert "dvdt_corr_mean" in out
    assert "dangdt_corr_mean" in out

