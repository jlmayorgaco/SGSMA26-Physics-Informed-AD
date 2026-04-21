from __future__ import annotations

import numpy as np

from src.estimation.dynamic_state_estimation.dynamic_measurement_model import build_measurement_matrices
from src.estimation.dynamic_state_estimation.hybrid_state_definition import build_hybrid_state_layout
from src.estimation.state_estimation.measurement_model import FrameMeasurements
from src.estimation.state_estimation.models import NetworkModel


def test_dynamic_measurement_model_dimensions() -> None:
    net = NetworkModel(
        ybus=np.asarray([[5 - 15j, -5 + 15j], [-5 + 15j, 5 - 15j]], dtype=complex),
        bus_order=["BUS1", "BUS2"],
        bus_kv_map={"BUS1": 345.0, "BUS2": 345.0},
    )
    layout = build_hybrid_state_layout(net.bus_order, ["BUS1"])
    x = np.zeros(layout.total_dim, dtype=float)
    x[layout.vr_offset + 0] = 1.0
    x[layout.vr_offset + 1] = 0.98
    frame = FrameMeasurements(
        timestamp=0.0,
        voltage_by_bus={"BUS1": 1.0 + 0j},
        current_by_bus={"BUS1": 0.05 + 0.01j},
        data_present_count=1,
        expected_pmu_buses=["BUS1"],
        valid_pmu_buses=["BUS1"],
        dropped_reasons={},
    )
    meas = build_measurement_matrices(frame=frame, network=net, layout=layout, x_pred=x)
    assert meas.H.shape[1] == layout.total_dim
    assert meas.z.shape[0] == 4
    assert meas.n_voltage == 2
    assert meas.n_current == 2

