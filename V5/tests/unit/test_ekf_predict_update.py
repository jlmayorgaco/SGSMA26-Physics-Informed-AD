from __future__ import annotations

import numpy as np

from src.estimation.dynamic_state_estimation.ekf_hybrid_estimator import run_hybrid_ekf
from src.estimation.dynamic_state_estimation.hybrid_state_definition import build_hybrid_state_layout
from src.estimation.state_estimation.measurement_model import FrameMeasurements
from src.estimation.state_estimation.models import NetworkModel


def test_ekf_predict_update_runs_small_case() -> None:
    net = NetworkModel(
        ybus=np.asarray([[8 - 18j, -8 + 18j], [-8 + 18j, 8 - 18j]], dtype=complex),
        bus_order=["BUS1", "BUS2"],
        bus_kv_map={"BUS1": 345.0, "BUS2": 345.0},
    )
    layout = build_hybrid_state_layout(net.bus_order, ["BUS1"])
    frames = [
        FrameMeasurements(
            timestamp=0.0,
            voltage_by_bus={"BUS1": 1.0 + 0j},
            current_by_bus={"BUS1": 0.03 + 0.01j},
            data_present_count=1,
            expected_pmu_buses=["BUS1"],
            valid_pmu_buses=["BUS1"],
            dropped_reasons={},
        )
    ]
    out = run_hybrid_ekf(
        network=net,
        prior=np.asarray([1.0 + 0j, 1.0 + 0j], dtype=complex),
        frames=frames,
        window_types=["quiet"],
        layout=layout,
        graph_laplacian=np.asarray([[1.0, -1.0], [-1.0, 1.0]]),
    )
    assert out["states"].shape == (1, 2)
    assert len(out["diagnostics"]) == 1

