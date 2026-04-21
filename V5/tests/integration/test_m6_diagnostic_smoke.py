from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from src.application.use_cases.m6_topology_aware_state_estimation import run_m6_topology_aware_state_estimation_use_case
from tests.helpers.m6_test_utils import create_synthetic_pmu_csvs


def test_m6_diagnostic_smoke_artifacts_exist() -> None:
    root = Path("output") / "_test_runs" / f"m6_diag_{uuid4().hex[:8]}"
    data_dir = create_synthetic_pmu_csvs(root / "pmu", n=4)
    run_m6_topology_aware_state_estimation_use_case(
        raw_path=Path("data/metadata/IEEE_39_Bus_Power_System.raw"),
        pmu_location_path=Path("data/metadata/PMUbus_ Location.txt"),
        pmu_data_dir=data_dir,
        output_dir=root / "out",
        diagnostic=True,
        use_andes_truth=False,
    )
    assert (root / "out" / "metadata" / "pmu_coverage_audit.csv").exists()
    assert (root / "out" / "intermediate" / "measurement_assembly_audit.csv").exists()
    assert (root / "out" / "metrics" / "frame_diagnostics.csv").exists()

