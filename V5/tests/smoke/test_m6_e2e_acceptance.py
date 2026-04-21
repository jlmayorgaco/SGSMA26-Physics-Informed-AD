from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from src.application.use_cases.m6_topology_aware_state_estimation import (
    run_m6_topology_aware_state_estimation_use_case,
)


@pytest.mark.andes
def test_m6_e2e_acceptance_ieee39() -> None:
    out = Path("output") / "_test_runs" / f"m6_e2e_{uuid4().hex[:8]}"
    summary = run_m6_topology_aware_state_estimation_use_case(
        raw_path=Path("data/metadata/IEEE_39_Bus_Power_System.raw"),
        pmu_location_path=Path("data/metadata/PMUbus_ Location.txt"),
        output_dir=out,
        use_andes_truth=True,
        end_time=3.0,
        stride=4,
    )
    assert summary["timestamp_count"] > 0
    assert (out / "report" / "report.md").exists()
    assert (out / "estimated" / "estimated_bus_states.csv").exists()
    assert (out / "metrics" / "global_metrics.json").exists()
    assert summary["min_pmus_used"] >= 8
