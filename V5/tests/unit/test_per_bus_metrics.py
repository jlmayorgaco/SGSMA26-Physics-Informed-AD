from __future__ import annotations

import numpy as np

from src.metrics.state_estimation_metrics import compute_state_estimation_metrics


def test_per_bus_metrics_columns_exist() -> None:
    t = np.asarray([0.0, 0.1])
    est = np.asarray([[1 + 0j, 0.99 + 0j], [1 + 0j, 1.01 + 0j]], dtype=complex)
    true = np.asarray([[1 + 0j, 1.0 + 0j], [1 + 0j, 1.0 + 0j]], dtype=complex)
    per_bus, _ = compute_state_estimation_metrics(t, ["BUS1", "BUS2"], est, true, {"BUS1"})
    assert {"BUS", "RMSE_V_MAG", "RMSE_ANG_DEG", "IS_PMU_BUS"}.issubset(per_bus.columns)

