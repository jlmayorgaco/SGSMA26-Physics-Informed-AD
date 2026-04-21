from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from src.application.use_cases.m6_topology_aware_state_estimation import (
    run_m6_topology_aware_state_estimation_use_case,
)


def test_m6_pipeline_smoke() -> None:
    out = Path("output") / "_test_runs" / f"m6_smoke_{uuid4().hex[:8]}"
    summary = run_m6_topology_aware_state_estimation_use_case(
        raw_path=Path("data/metadata/IEEE_39_Bus_Power_System.raw"),
        pmu_location_path=Path("data/metadata/PMUbus_ Location.txt"),
        pmu_data_dir=Path("tests/fixtures/raw_small"),
        output_dir=out,
        use_andes_truth=False,
        stride=2,
    )
    assert Path(summary["output_dir"]).exists()
    assert (out / "estimated" / "estimated_bus_states.csv").exists()
    assert (out / "metrics" / "residual_diagnostics.csv").exists()

