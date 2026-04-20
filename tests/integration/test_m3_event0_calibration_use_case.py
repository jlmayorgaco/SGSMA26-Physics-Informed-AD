from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from src.application.use_cases.calibrate_event0 import run_calibrate_event0_use_case


def _workspace_dir(prefix: str) -> Path:
    root = Path("output") / "_test_runs" / f"{prefix}_{uuid4().hex[:8]}"
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.mark.integration
@pytest.mark.andes
def test_m3_event0_calibration_use_case() -> None:
    out_dir = _workspace_dir("m3_integration")
    summary = run_calibrate_event0_use_case(
        raw_chunks_dir=Path("output/SCENARIO_RAW0001/chunks"),
        raw_profile_file=Path("output/SCENARIO_RAW0001/dataset_profiles_raw.json"),
        output_dir=out_dir,
        event_label=0,
    )
    assert Path(summary["output_dir"]).exists()
    assert (out_dir / "metrics" / "calibration_results.json").exists()
    assert (out_dir / "current_mapping_selection.csv").exists()
    assert (out_dir / "signal_support_matrix.csv").exists()
    assert summary["record_count"] > 0
    assert summary["evaluated_row_count"] > 0
    statuses = summary["status_counts"]
    assert statuses.get("unsupported", 0) < summary["record_count"]
