from __future__ import annotations

import pandas as pd

from src.application.use_cases.m7_state_estimator_benchmark import _group_metrics


def test_group_metrics_pmu_nonpmu() -> None:
    df = pd.DataFrame(
        [
            {"BUS": "BUS1", "IS_PMU_BUS": True, "RMSE_V_MAG": 0.01, "RMSE_ANG_DEG": 1.0},
            {"BUS": "BUS2", "IS_PMU_BUS": False, "RMSE_V_MAG": 0.03, "RMSE_ANG_DEG": 3.0},
        ]
    )
    g = _group_metrics(df)
    assert g["rmse_v_mag_pmu"] < g["rmse_v_mag_nonpmu"]

