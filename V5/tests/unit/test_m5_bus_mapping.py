from __future__ import annotations

from pathlib import Path

from src.metadata.bus_mapping import build_bus_alignment, canonicalize_bus_label, numeric_bus_sort_key
from src.metadata.pmu_location_parser import parse_pmu_location_file
from src.metadata.raw_parser import parse_raw_file


def test_canonicalize_bus_label_works() -> None:
    assert canonicalize_bus_label("'bus10'") == "BUS10"


def test_numeric_bus_sort_key_orders_bus2_before_bus10() -> None:
    labels = ["BUS10", "BUS2"]
    assert sorted(labels, key=numeric_bus_sort_key) == ["BUS2", "BUS10"]


def test_bus30x1_handled_correctly() -> None:
    assert canonicalize_bus_label("BUS30x1") == "BUS30X1"


def test_bus_alignment_between_files_works() -> None:
    pmu = parse_pmu_location_file(Path("data/metadata/PMUbus_ Location.txt"))
    raw = parse_raw_file(Path("data/metadata/IEEE_39_Bus_Power_System.raw"))
    align = build_bus_alignment(pmu, raw)
    assert align["aligned_ok"]
