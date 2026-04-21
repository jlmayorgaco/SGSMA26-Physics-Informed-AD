from __future__ import annotations

from pathlib import Path

from src.infrastructure.io.pmu_csv_loader import PmuCsvLoader


def test_pmu_csv_loader_loads_single_bus() -> None:
    loader = PmuCsvLoader(Path("tests/fixtures/raw_small"))
    df = loader.load_bus("10")
    assert "TIMESTAMP" in df.columns
    assert "BUS10_VA_MAG" in df.columns
    assert len(df) > 0

