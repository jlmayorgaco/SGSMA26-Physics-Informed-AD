from __future__ import annotations

from pathlib import Path

from src.infrastructure.io.metadata_loader import load_pmu_metadata


def test_pmu_metadata_loader_expected_buses() -> None:
    meta = load_pmu_metadata(Path("data/metadata/PMUbus_ Location.txt"))
    pmu_buses = {entry["bus_label_canonical"] for entry in meta["pmu_map"]}
    expected = {"BUS39", "BUS29", "BUS10", "BUS22", "BUS19", "BUS2", "BUS5", "BUS6"}
    assert expected.issubset(pmu_buses)
    assert float(meta["base_mva"]) == 100.0

