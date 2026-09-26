from __future__ import annotations

from pathlib import Path

import numpy as np

from src.calibration.raw_chunk_loader import (
    discover_event0_chunks,
    load_event0_chunks_for_bus,
    raw_col,
    raw_values_from_chunks,
)


def test_raw_col_formats_correctly() -> None:
    assert raw_col("10", "VA_MAG") == "BUS10_VA_MAG"


def test_discover_event0_chunks_finds_only_event0_chunks() -> None:
    chunks = discover_event0_chunks(Path("tests/fixtures/m2_chunks_small"))
    assert chunks
    assert all("_event_0_" in c.name for c in chunks)


def test_load_event0_chunks_for_bus_loads_existing_bus_csv_only() -> None:
    chunks = load_event0_chunks_for_bus("10", Path("tests/fixtures/m2_chunks_small"))
    assert chunks
    assert all("Bus10.csv" in c["path"] for c in chunks)


def test_raw_values_from_chunks_concatenates_values() -> None:
    chunks = load_event0_chunks_for_bus("10", Path("tests/fixtures/m2_chunks_small"))
    vals = raw_values_from_chunks(chunks, "10", "VA_MAG")
    assert len(vals) > 0


def test_wrapped_delta_representation_works_correctly() -> None:
    chunks = load_event0_chunks_for_bus("10", Path("tests/fixtures/m2_chunks_small"))
    vals = raw_values_from_chunks(chunks, "10", "VA_ANG", representation="wrapped_delta")
    assert np.all(vals >= -180.0)
    assert np.all(vals < 180.0)
