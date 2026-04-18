from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import pytest

from src.infrastructure.legacy.m1_adapter import normalize_bus_data


@pytest.mark.regression
def test_snapshot_normalization_baselines(snapshot_dir: Path) -> None:
    idx = pd.Index([0.0, 0.1, 0.2, 0.3], name="TIMESTAMP")
    bus_df = pd.DataFrame(
        {
            "BUS10_VA_MAG": [100.0, 100.0, 120.0, 140.0],
            "BUS10_ROCOF": [0.1, 0.1, 0.4, 0.5],
            "DATA_PRESENT": [1, 1, 1, 1],
            "Event": [0, 0, 1, 1],
        },
        index=idx,
    )

    _, baselines = normalize_bus_data({"Bus10": bus_df}, pd.DataFrame({"Bus10": [0, 0, 1, 1]}, index=idx))
    actual = baselines[["bus_id", "raw_signal", "transform", "baseline_method", "baseline_value"]].copy()
    actual = actual.sort_values(["bus_id", "raw_signal"]).reset_index(drop=True)

    expected_csv = (snapshot_dir / "snapshot_normalization_baselines.csv").read_text(encoding="utf-8")
    expected = pd.read_csv(StringIO(expected_csv))
    assert actual.equals(expected)
