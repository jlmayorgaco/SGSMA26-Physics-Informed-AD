from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.infrastructure.legacy.m4_adapter import compute_estimation_metrics


@pytest.mark.integration
@pytest.mark.andes
def test_legacy_m4_metrics_columns_and_angle_wrap() -> None:
    bus = "1"
    sim_df = pd.DataFrame(
        {
            "BUS1_VA_ANG": [179.0, 179.0],
            "BUS1_VA_MAG": [1.0, 1.0],
            "BUS1_VB_ANG": [0.0, 0.0],
            "BUS1_VB_MAG": [1.0, 1.0],
            "BUS1_VC_ANG": [0.0, 0.0],
            "BUS1_VC_MAG": [1.0, 1.0],
            "BUS1_IA_ANG": [0.0, 0.0],
            "BUS1_IA_MAG": [1.0, 1.0],
            "BUS1_IB_ANG": [0.0, 0.0],
            "BUS1_IB_MAG": [1.0, 1.0],
            "BUS1_IC_ANG": [0.0, 0.0],
            "BUS1_IC_MAG": [1.0, 1.0],
            "BUS1_Freq": [60.0, 60.0],
            "BUS1_ROCOF": [0.0, 0.0],
        }
    )
    est_df = sim_df.copy()
    est_df["BUS1_VA_ANG"] = -179.0

    metrics_long, _, _ = compute_estimation_metrics({bus: sim_df}, {bus: est_df})
    expected_columns = {
        "bus_id",
        "signal",
        "source",
        "rmse",
        "mae",
        "relative_rmse",
        "corr",
        "max_abs_error",
    }
    assert expected_columns.issubset(metrics_long.columns)

    va_ang = metrics_long[metrics_long["signal"] == "VA_ANG"].iloc[0]
    assert np.isclose(va_ang["rmse"], 2.0)
    assert np.isclose(va_ang["mae"], 2.0)
