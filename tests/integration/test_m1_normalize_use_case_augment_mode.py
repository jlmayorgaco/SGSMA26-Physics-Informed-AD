from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

from src.application.use_cases.normalize_data import run_normalize_data_use_case


def _workspace_test_dir(prefix: str) -> Path:
    out_dir = Path("output") / "_test_runs" / f"{prefix}_{uuid4().hex[:8]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


@pytest.mark.integration
def test_m1_normalize_use_case_augment_mode_adds_expected_features(raw_small_dir: Path) -> None:
    out_dir = _workspace_test_dir("m1_integration_augment")
    run_normalize_data_use_case(
        input_dir=raw_small_dir,
        output_dir=out_dir,
        generate_plots=False,
        feature_mode="augment_angles",
    )

    csvs = sorted((out_dir / "chunks").glob("chunk*/Bus*_normalized.csv"))
    assert csvs, "No normalized chunk CSVs produced in augment mode"
    sample = pd.read_csv(csvs[0])

    assert "DATA_PRESENT" in sample.columns
    assert "Event" in sample.columns
    assert "BUS10_VA_ANG" in sample.columns
    assert "BUS10_VA_MAG" in sample.columns
    assert "BUS10_Freq" in sample.columns
    assert "BUS10_ROCOF" in sample.columns

    assert "BUS10_VA_ANG_SPEED_RAD_S" in sample.columns
    assert "BUS10_VA_ANG_SIN" in sample.columns
    assert "BUS10_VA_ANG_COS" in sample.columns
    assert "BUS10_VA_ANG_DEV_DEG" in sample.columns
    assert "BUS10_VA_MAG_PU" in sample.columns
    assert "BUS10_VA_MAG_DEV_PU" in sample.columns
    assert "BUS10_Freq_PU" in sample.columns
    assert "BUS10_Freq_DEV_HZ" in sample.columns
    assert "BUS10_ROCOF_CENTERED" in sample.columns
