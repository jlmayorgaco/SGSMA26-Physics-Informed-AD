from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.application.use_cases.m6_topology_aware_state_estimation import run_m6_topology_aware_state_estimation_use_case
from tests.helpers.m6_test_utils import create_synthetic_pmu_csvs


def test_bus_variability_summary_exists() -> None:
    root = Path("output") / "_test_runs" / f"m6_var_{uuid4().hex[:8]}"
    data_dir = create_synthetic_pmu_csvs(root / "pmu", n=4)
    run_m6_topology_aware_state_estimation_use_case(
        raw_path=Path("data/metadata/IEEE_39_Bus_Power_System.raw"),
        pmu_location_path=Path("data/metadata/PMUbus_ Location.txt"),
        pmu_data_dir=data_dir,
        output_dir=root / "out",
        use_andes_truth=False,
        diagnostic=True,
    )
    df = pd.read_csv(root / "out" / "metrics" / "bus_variability_summary.csv")
    assert {"BUS", "STD_V_MAG", "IS_NEARLY_FROZEN"}.issubset(df.columns)
    assert len(df) == 39

