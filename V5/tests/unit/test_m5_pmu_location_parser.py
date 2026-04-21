from __future__ import annotations

from pathlib import Path

from src.metadata.pmu_location_parser import parse_pmu_location_file


PMU_PATH = Path("data/metadata/PMUbus_ Location.txt")


def test_parses_base_mva_correctly() -> None:
    meta = parse_pmu_location_file(PMU_PATH)
    assert meta["base_mva"] == 100.0


def test_parses_rated_frequency_correctly() -> None:
    meta = parse_pmu_location_file(PMU_PATH)
    assert meta["rated_frequency_hz"] == 60.0


def test_parses_bus_count_39() -> None:
    meta = parse_pmu_location_file(PMU_PATH)
    assert meta["bus_count"] == 39


def test_parses_pmu_bus_ids_correctly() -> None:
    meta = parse_pmu_location_file(PMU_PATH)
    assert set(meta["pmu_bus_ids"]) == {"BUS39", "BUS29", "BUS10", "BUS22", "BUS19", "BUS2", "BUS5", "BUS6"}


def test_parses_bus30x1_correctly() -> None:
    meta = parse_pmu_location_file(PMU_PATH)
    labels = {b["bus_label_canonical"] for b in meta["buses"]}
    assert "BUS30X1" in labels


def test_bus_types_1_2_3_map_to_names() -> None:
    meta = parse_pmu_location_file(PMU_PATH)
    type_names = {b["bus_type_name"] for b in meta["buses"]}
    assert {"PQ", "PV", "SWING"}.issubset(type_names)
