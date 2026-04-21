from __future__ import annotations

import numpy as np

from src.metrics.state_estimation_metrics import compute_state_estimation_metrics


def test_metrics_shapes() -> None:
    t = np.asarray([0.0, 0.1, 0.2])
    est = np.asarray([[1 + 0j, 0.99 + 0.01j], [1 + 0j, 0.98 + 0.02j], [1 + 0j, 1.0 + 0.0j]], dtype=complex)
    true = np.asarray([[1 + 0j, 1.0 + 0j], [1 + 0j, 1.0 + 0j], [1 + 0j, 1.0 + 0j]], dtype=complex)
    per_bus, global_metrics = compute_state_estimation_metrics(t, ["BUS1", "BUS2"], est, true, {"BUS1"})
    assert len(per_bus) == 2
    assert "rmse_mag_pu_global" in global_metrics

