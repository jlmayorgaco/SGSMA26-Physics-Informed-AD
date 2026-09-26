from __future__ import annotations

from pathlib import Path

import pytest

from src.calibration.raw_chunk_loader import discover_event0_chunks, load_event0_chunks_for_bus


@pytest.mark.integration
def test_m3_event0_chunk_loading_from_fixture() -> None:
    base = Path("tests/fixtures/m2_chunks_small")
    chunk_dirs = discover_event0_chunks(base)
    assert len(chunk_dirs) > 0
    assert all("_event_0_" in p.name for p in chunk_dirs)

    bus10_chunks = load_event0_chunks_for_bus("10", base)
    assert len(bus10_chunks) > 0
    assert all(Path(c["path"]).name == "Bus10.csv" for c in bus10_chunks)
