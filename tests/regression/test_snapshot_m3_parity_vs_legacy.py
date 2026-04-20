from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from src.application.use_cases.calibrate_event0 import run_calibrate_event0_use_case
from src.infrastructure.legacy.m3_adapter import run_raw_event0_calibration


def _workspace_dir(prefix: str) -> Path:
    root = Path("output") / "_test_runs" / f"{prefix}_{uuid4().hex[:8]}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.regression
@pytest.mark.andes
def test_m3_migrated_matches_legacy_key_artifacts() -> None:
    root = _workspace_dir("m3_regression")
    migrated = root / "migrated"
    legacy = root / "legacy"

    summary = run_calibrate_event0_use_case(
        raw_chunks_dir=Path("output/SCENARIO_RAW0001/chunks"),
        raw_profile_file=Path("output/SCENARIO_RAW0001/dataset_profiles_raw.json"),
        output_dir=migrated,
        event_label=0,
    )
    assert summary["record_count"] > 0
    assert summary["evaluated_row_count"] > 0
    run_raw_event0_calibration(
        raw_chunks_dir=str(Path("output/SCENARIO_RAW0001/chunks")),
        raw_profile_file=str(Path("output/SCENARIO_RAW0001/dataset_profiles_raw.json")),
        output_dir=str(legacy),
        event_label=0,
    )

    for p in [
        migrated / "current_mapping_selection.csv",
        migrated / "signal_support_matrix.csv",
        migrated / "metrics" / "calibration_metrics_long.csv",
        migrated / "metrics" / "calibration_results.json",
    ]:
        assert p.exists()

    assert_frame_equal(
        pd.read_csv(migrated / "current_mapping_selection.csv"),
        pd.read_csv(legacy / "current_mapping_selection.csv"),
        check_dtype=False,
        check_like=False,
        rtol=1e-7,
        atol=1e-9,
    )
    assert_frame_equal(
        pd.read_csv(migrated / "signal_support_matrix.csv"),
        pd.read_csv(legacy / "signal_support_matrix.csv"),
        check_dtype=False,
        check_like=False,
        rtol=1e-7,
        atol=1e-9,
    )
    assert_frame_equal(
        pd.read_csv(migrated / "metrics" / "calibration_metrics_long.csv"),
        pd.read_csv(legacy / "metrics" / "calibration_metrics_long.csv"),
        check_dtype=False,
        check_like=False,
        rtol=1e-6,
        atol=1e-8,
    )
    assert _load_json(migrated / "metrics" / "calibration_results.json")["records"] == _load_json(
        legacy / "metrics" / "calibration_results.json"
    )["records"]

    migrated_plots = list((migrated / "plots").rglob("*.png")) if (migrated / "plots").exists() else []
    legacy_plots = list((legacy / "plots").rglob("*.png")) if (legacy / "plots").exists() else []
    assert len(migrated_plots) == len(legacy_plots)
