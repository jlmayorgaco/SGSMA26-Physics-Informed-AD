from __future__ import annotations

from pathlib import Path

from src.detectors.pipelines.raw_holdout_loader import discover_raw_scenario_dirs, load_raw_scenario_frame


def test_raw_holdout_loader_reads_direct_raw_folder() -> None:
    raw_root = Path("tests/fixtures/raw_small")
    refs = discover_raw_scenario_dirs(raw_root)
    assert len(refs) == 1
    assert refs[0].scenario_dir.resolve() == raw_root.resolve()

    frame = load_raw_scenario_frame(raw_root)
    assert {"TIMESTAMP", "DATA_PRESENT", "EVENT"}.issubset(frame.columns)
    assert frame["TIMESTAMP"].is_monotonic_increasing
    assert frame["EVENT"].isin(range(9)).all()
    assert frame["DATA_PRESENT"].between(0.0, 1.0).all()
