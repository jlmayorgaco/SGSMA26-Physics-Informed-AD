from __future__ import annotations

from src.simulation.m9.calibration import extract_reference_statistics


def test_noise_calibration() -> None:
    stats = extract_reference_statistics("data/RAW0001", max_rows=5000)
    assert stats["bus_count"] >= 8
    assert stats["source"] in {"RAW0001", "fallback"}
    bus39 = stats["buses"].get("BUS39", {})
    ch = bus39.get("channels", {}).get("BUS39_Freq", {})
    assert "mean" in ch
    assert "std" in ch
