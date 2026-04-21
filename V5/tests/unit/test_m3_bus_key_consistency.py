from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.calibration.raw_chunk_loader import load_event0_chunks_for_bus, normalize_bus_id, raw_col


def _write_bus_csv(path: Path, bus_id: str) -> None:
    df = pd.DataFrame(
        {
            "TIMESTAMP": [0.0, 1.0],
            "DATA_PRESENT": [1, 1],
            "Event": [0, 0],
            f"BUS{bus_id}_VA_MAG": [1.0, 1.01],
        }
    )
    df.to_csv(path, index=False)


def test_bus_id_normalization_to_string_consistent() -> None:
    assert normalize_bus_id("2") == "2"
    assert normalize_bus_id(2) == "2"
    assert normalize_bus_id("02") == "2"
    assert raw_col(2, "VA_MAG") == "BUS2_VA_MAG"


def test_trajectories_lookup_works_with_string_bus_ids() -> None:
    trajectories = {"2": {"ok": True}, "5": {"ok": True}, "39": {"ok": True}}
    for bus_id in ["2", "5", "39"]:
        assert trajectories.get(normalize_bus_id(bus_id)) is not None


def test_load_event0_chunks_for_bus_string_and_int() -> None:
    root = Path("tests/fixtures/_tmp_m3_bus_key_consistency")
    chunk = root / "chunk01_event_0_normal_operation"
    chunk.mkdir(parents=True, exist_ok=True)
    _write_bus_csv(chunk / "Bus2.csv", "2")

    try:
        from_str = load_event0_chunks_for_bus("2", root)
        from_int = load_event0_chunks_for_bus(2, root)
        assert len(from_str) == 1
        assert len(from_int) == 1
        assert from_str[0]["path"].endswith("Bus2.csv")
        assert from_int[0]["path"].endswith("Bus2.csv")
    finally:
        if root.exists():
            for p in sorted(root.rglob("*"), reverse=True):
                if p.is_file():
                    p.unlink()
                else:
                    p.rmdir()


def test_no_zero_padded_bus_filename_lookup() -> None:
    root = Path("tests/fixtures/_tmp_m3_bus_key_consistency_padded")
    chunk = root / "chunk01_event_0_normal_operation"
    chunk.mkdir(parents=True, exist_ok=True)
    _write_bus_csv(chunk / "Bus02.csv", "02")

    try:
        loaded = load_event0_chunks_for_bus("2", root)
        assert loaded == []
    finally:
        if root.exists():
            for p in sorted(root.rglob("*"), reverse=True):
                if p.is_file():
                    p.unlink()
                else:
                    p.rmdir()
