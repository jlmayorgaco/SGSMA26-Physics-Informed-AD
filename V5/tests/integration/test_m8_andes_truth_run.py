from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from src.application.use_cases.m8_hybrid_dynamic_benchmark import run_m8_hybrid_dynamic_benchmark_use_case


@pytest.mark.andes
def test_m8_andes_truth_run() -> None:
    out = Path("output") / "_test_runs" / f"m8_andes_{uuid4().hex[:8]}"
    summary = run_m8_hybrid_dynamic_benchmark_use_case(
        raw_path=Path("data/metadata/IEEE_39_Bus_Power_System.raw"),
        pmu_location_path=Path("data/metadata/PMUbus_ Location.txt"),
        output_dir=out,
        use_andes_truth=True,
        end_time=3.0,
        stride=6,
    )
    assert summary["best_estimator"] != ""
    assert (out / "truth" / "andes_truth_bus_states.csv").exists()
    assert (out / "report" / "m8_benchmark_results.json").exists()

