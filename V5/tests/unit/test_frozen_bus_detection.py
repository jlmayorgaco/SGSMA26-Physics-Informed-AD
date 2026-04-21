from __future__ import annotations

import pandas as pd

from src.application.use_cases.m6_topology_aware_state_estimation import _build_bus_variability_summary


def test_frozen_bus_detection_logic() -> None:
    df = pd.DataFrame(
        [
            {"TIMESTAMP": 0.0, "BUS": "BUS1", "IS_PMU_BUS": False, "V_EST_MAG_PU": 1.0, "V_EST_ANG_DEG": 0.0},
            {"TIMESTAMP": 0.1, "BUS": "BUS1", "IS_PMU_BUS": False, "V_EST_MAG_PU": 1.0, "V_EST_ANG_DEG": 0.0},
            {"TIMESTAMP": 0.0, "BUS": "BUS2", "IS_PMU_BUS": False, "V_EST_MAG_PU": 1.0, "V_EST_ANG_DEG": 0.0},
            {"TIMESTAMP": 0.1, "BUS": "BUS2", "IS_PMU_BUS": False, "V_EST_MAG_PU": 1.1, "V_EST_ANG_DEG": 2.0},
        ]
    )
    v = _build_bus_variability_summary(df)
    b1 = bool(v.loc[v["BUS"] == "BUS1", "IS_NEARLY_FROZEN"].iloc[0])
    b2 = bool(v.loc[v["BUS"] == "BUS2", "IS_NEARLY_FROZEN"].iloc[0])
    assert b1 is True and b2 is False

