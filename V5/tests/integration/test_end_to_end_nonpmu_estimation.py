from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.application.use_cases.m6_topology_aware_state_estimation import (
    run_m6_topology_aware_state_estimation_use_case,
)


def test_end_to_end_nonpmu_estimation() -> None:
    out = Path("output") / "_test_runs" / f"m6_nonpmu_{uuid4().hex[:8]}"
    run_m6_topology_aware_state_estimation_use_case(
        raw_path=Path("data/metadata/IEEE_39_Bus_Power_System.raw"),
        pmu_location_path=Path("data/metadata/PMUbus_ Location.txt"),
        pmu_data_dir=Path("tests/fixtures/raw_small"),
        output_dir=out,
        use_andes_truth=False,
        stride=3,
    )
    df = pd.read_csv(out / "estimated" / "estimated_bus_states.csv")
    assert (df["IS_PMU_BUS"] == False).any()  # noqa: E712
    assert df["V_EST_MAG_PU"].notna().all()

