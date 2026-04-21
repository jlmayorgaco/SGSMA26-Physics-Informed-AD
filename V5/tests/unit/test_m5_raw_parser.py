from __future__ import annotations

from pathlib import Path

from src.metadata.raw_parser import parse_raw_file


RAW_PATH = Path("data/metadata/IEEE_39_Bus_Power_System.raw")


def test_parses_raw_base_mva() -> None:
    raw = parse_raw_file(RAW_PATH)
    assert raw["base_mva"] == 100.0


def test_parses_nonempty_bus_section() -> None:
    raw = parse_raw_file(RAW_PATH)
    assert len(raw["buses"]) > 0


def test_parses_nonempty_branch_section() -> None:
    raw = parse_raw_file(RAW_PATH)
    assert len(raw["branches"]) > 0


def test_parses_nonempty_generator_section() -> None:
    raw = parse_raw_file(RAW_PATH)
    assert len(raw["generators"]) > 0


def test_parser_handles_quoted_bus_labels() -> None:
    raw = parse_raw_file(RAW_PATH)
    labels = {b["bus_label_original"] for b in raw["buses"]}
    assert "BUS1" in labels


def test_parser_is_deterministic() -> None:
    a = parse_raw_file(RAW_PATH)
    b = parse_raw_file(RAW_PATH)
    assert a["raw_sections_summary"] == b["raw_sections_summary"]
    assert [x["bus_label_canonical"] for x in a["buses"]] == [x["bus_label_canonical"] for x in b["buses"]]
