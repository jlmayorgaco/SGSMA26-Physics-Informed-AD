from __future__ import annotations

import numpy as np

from src.metrics.state_estimation_metrics import compute_state_estimation_metrics


def test_benchmark_metrics_known_input() -> None:
    t = np.asarray([0.0, 0.1, 0.2])
    truth = np.asarray([[1 + 0j, 1 + 0j], [1 + 0j, 1 + 0j], [1 + 0j, 1 + 0j]], dtype=complex)
    est = np.asarray([[1 + 0j, 0.99 + 0.0j], [1 + 0j, 0.98 + 0.0j], [1 + 0j, 1 + 0.0j]], dtype=complex)
    per_bus, global_m = compute_state_estimation_metrics(t, ["BUS1", "BUS2"], est, truth, {"BUS1"})
    assert len(per_bus) == 2
    assert global_m["rmse_v_mag_all"] >= 0.0

