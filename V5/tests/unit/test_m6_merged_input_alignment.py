from __future__ import annotations

import pandas as pd

from src.application.use_cases.m6_topology_aware_state_estimation import _merged_input_from_pmu_tables


def test_merged_input_alignment_uses_timestamp_key() -> None:
    df1 = pd.DataFrame({"TIMESTAMP": [0.0, 0.033], "BUS10_VA_MAG": [1, 2], "DATA_PRESENT": [1, 1], "Event": [0, 0]})
    df2 = pd.DataFrame({"TIMESTAMP": [0.0001, 0.0332], "BUS22_VA_MAG": [3, 4], "DATA_PRESENT": [1, 1], "Event": [0, 0]})
    merged = _merged_input_from_pmu_tables({"10": df1, "22": df2})
    assert "BUS10_DATA_PRESENT" in merged.columns
    assert "BUS22_DATA_PRESENT" in merged.columns
    assert len(merged) == 2

