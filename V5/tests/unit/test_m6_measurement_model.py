from __future__ import annotations

import numpy as np

from src.estimation.state_estimation.measurement_model import FrameMeasurements, build_linear_frame
from src.estimation.state_estimation.models import EstimationConfig, NetworkModel


def test_measurement_model_dimensions() -> None:
    y = np.asarray([[10 - 20j, -10 + 20j], [-10 + 20j, 10 - 20j]], dtype=complex)
    net = NetworkModel(ybus=y, bus_order=["BUS1", "BUS2"], bus_kv_map={"BUS1": 345.0, "BUS2": 345.0})
    frame = FrameMeasurements(
        timestamp=0.0,
        voltage_by_bus={"BUS1": 1.0 + 0j},
        current_by_bus={"BUS1": 0.1 - 0.2j},
        data_present_count=1,
    )
    lf = build_linear_frame(frame, net, EstimationConfig())
    assert lf.A.shape == (2, 2)
    assert lf.b.shape == (2,)
    assert lf.n_voltage == 1
    assert lf.n_current == 1

