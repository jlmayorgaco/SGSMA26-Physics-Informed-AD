from __future__ import annotations

import pandas as pd

from tests.helpers.m9_test_utils import generate_single


def test_all_bus_truth_export(tmp_path) -> None:
    _, scenario_dir = generate_single(tmp_path, "TEMPLATE_EVENT2_LINE_OUTAGE")
    truth_path = scenario_dir / "all_buses" / "all_bus_truth.csv"
    assert truth_path.exists()
    truth = pd.read_csv(truth_path)
    required = {
        "TIMESTAMP",
        "BUS",
        "IS_PMU_BUS",
        "IS_NON_PMU_BUS",
        "V_TRUE_REAL_PU",
        "V_TRUE_IMAG_PU",
        "V_TRUE_MAG_PU",
        "V_TRUE_ANG_DEG",
        "EVENT",
        "EVENT_ORIGIN_BUS",
        "EVENT_ORIGIN_LINE",
        "PHYSICAL_EVENT_TYPE",
        "CYBER_EVENT_TYPE",
        "WINDOW_TYPE",
    }
    assert required.issubset(truth.columns)
    assert truth["BUS"].nunique() == 39
    assert truth["IS_NON_PMU_BUS"].astype(bool).sum() > 0
