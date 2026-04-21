from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED = {"TIMESTAMP", "DATA_PRESENT", "Event"}


def test_fixture_schema_contract(raw_small_dir: Path) -> None:
    csv_files = sorted(raw_small_dir.glob("Bus*_Competition_Data_nanmask.csv"))
    assert csv_files
    for path in csv_files:
        df = pd.read_csv(path)
        assert REQUIRED.issubset(set(df.columns))
        assert any(col.endswith("_VA_MAG") for col in df.columns)
        assert any(col.endswith("_ROCOF") for col in df.columns)
