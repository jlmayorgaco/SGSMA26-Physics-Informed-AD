from __future__ import annotations

from pathlib import Path

from src.metadata.pmu_location_parser import parse_pmu_location_file
from src.metadata.raw_parser import parse_raw_file
from src.metadata.validation import validate_metadata_consistency
from src.metadata.ybus_builder import build_ybus


def _parsed():
    pmu = parse_pmu_location_file(Path("data/metadata/PMUbus_ Location.txt"))
    raw = parse_raw_file(Path("data/metadata/IEEE_39_Bus_Power_System.raw"))
    y, _, yinfo = build_ybus(raw)
    yinfo["zbus_status"] = {"built": True}
    _ = y
    return pmu, raw, yinfo


def test_pmu_bus_set_matches_expected_set() -> None:
    pmu, raw, yinfo = _parsed()
    rep = validate_metadata_consistency(pmu, raw, yinfo)
    assert rep["pmu_mapping_ok"]


def test_validation_catches_missing_bus() -> None:
    pmu, raw, yinfo = _parsed()
    pmu["buses"] = pmu["buses"][:-1]
    rep = validate_metadata_consistency(pmu, raw, yinfo)
    assert not rep["metadata_vs_raw_bus_set_ok"]


def test_validation_catches_mismatched_kv() -> None:
    pmu, raw, yinfo = _parsed()
    pmu["buses"][0]["kv_ll"] = 999.0
    rep = validate_metadata_consistency(pmu, raw, yinfo)
    assert not rep["bus_kv_ok"]


def test_validation_report_has_overall_flags() -> None:
    pmu, raw, yinfo = _parsed()
    rep = validate_metadata_consistency(pmu, raw, yinfo)
    for k in ["metadata_vs_raw_bus_set_ok", "pmu_mapping_ok", "ready_for_estimation"]:
        assert k in rep
