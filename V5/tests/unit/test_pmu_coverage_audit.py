from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.application.use_cases.m6_topology_aware_state_estimation import run_m6_topology_aware_state_estimation_use_case
from tests.helpers.m6_test_utils import create_synthetic_pmu_csvs


def test_pmu_coverage_audit_has_expected_8() -> None:
    root = Path("output") / "_test_runs" / f"m6_cov_{uuid4().hex[:8]}"
    data_dir = root / "pmu"
    create_synthetic_pmu_csvs(data_dir, n=5)
    run_m6_topology_aware_state_estimation_use_case(
        raw_path=Path("data/metadata/IEEE_39_Bus_Power_System.raw"),
        pmu_location_path=Path("data/metadata/PMUbus_ Location.txt"),
        pmu_data_dir=data_dir,
        output_dir=root / "out",
        diagnostic=True,
        use_andes_truth=False,
    )
    df = pd.read_csv(root / "out" / "metadata" / "pmu_coverage_audit.csv")
    assert len(df) >= 8
    assert int(df["FOUND_IN_MERGED_INPUT"].astype(bool).sum()) >= 8

