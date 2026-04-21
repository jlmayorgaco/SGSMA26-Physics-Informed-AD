from __future__ import annotations

import numpy as np

from src.estimation.state_estimation.measurement_model import FrameMeasurements
from src.estimation.state_estimation.models import EstimationConfig, NetworkModel
from src.estimation.state_estimation.pmu_state_estimator import PmuStateEstimator


def test_state_estimator_recovers_simple_state() -> None:
    y = np.asarray([[12 - 30j, -12 + 30j], [-12 + 30j, 12 - 30j]], dtype=complex)
    net = NetworkModel(ybus=y, bus_order=["BUS1", "BUS2"], bus_kv_map={"BUS1": 345.0, "BUS2": 345.0})
    x_true = np.asarray([1.0 + 0.0j, 0.98 - 0.05j], dtype=complex)
    i_true = y @ x_true
    frame = FrameMeasurements(
        timestamp=0.0,
        voltage_by_bus={"BUS1": x_true[0]},
        current_by_bus={"BUS1": i_true[0]},
        data_present_count=1,
    )
    est = PmuStateEstimator(net, EstimationConfig(lambda_reg=1e-2, mu_reg=1e-4, current_weight=1.0))
    result = est.estimate_timeseries([frame], x_prior=np.asarray([1 + 0j, 1 + 0j], dtype=complex))
    assert result.voltage_estimates_pu.shape == (1, 2)
    assert np.isfinite(np.real(result.voltage_estimates_pu)).all()

