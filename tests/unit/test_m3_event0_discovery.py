from __future__ import annotations

from pathlib import Path

from src.calibration.raw_chunk_loader import discover_event0_chunks


def test_discover_event0_chunks_only_event0() -> None:
    base = Path("tests/fixtures/m2_chunks_small")
    chunks = discover_event0_chunks(base)
    assert chunks
    assert all("_event_0_" in c.name for c in chunks)
    assert not any("_event_5_" in c.name or "_event_7_" in c.name for c in chunks)


def test_discover_event0_chunks_sorted_deterministic() -> None:
    base = Path("tests/fixtures/m2_chunks_small")
    chunks = discover_event0_chunks(base)
    names = [c.name for c in chunks]
    assert names == sorted(names)
