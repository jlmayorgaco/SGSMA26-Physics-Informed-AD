from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from src.application.use_cases.build_metadata_model import run_build_metadata_model_use_case


def test_m5_build_metadata_model_use_case() -> None:
    out = Path("output") / "_test_runs" / f"m5_integration_{uuid4().hex[:8]}"
    out.mkdir(parents=True, exist_ok=True)
    summary = run_build_metadata_model_use_case(
        pmu_location_path=Path("data/metadata/PMUbus_ Location.txt"),
        raw_path=Path("data/metadata/IEEE_39_Bus_Power_System.raw"),
        output_dir=out,
    )
    assert Path(summary["output_dir"]).exists()
    assert (out / "validation_report.json").exists()
    assert (out / "ybus_real.json").exists()
    assert (out / "zbus_real.json").exists()
