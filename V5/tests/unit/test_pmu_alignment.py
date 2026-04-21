from __future__ import annotations

import pandas as pd
import pytest

from src.detectors.shared.preprocessing.pmu_alignment import align_pmu_frames


def test_pmu_alignment_preserves_timestamps() -> None:
    a = pd.DataFrame({"TIMESTAMP": [0.0, 0.1], "BUS10_VA_MAG": [1.0, 2.0], "DATA_PRESENT": [1, 1], "Event": [0, 5]})
    b = pd.DataFrame({"TIMESTAMP": [0.0, 0.1], "BUS39_VA_MAG": [3.0, 4.0], "DATA_PRESENT": [1, 0], "Event": [0, 0]})
    merged = align_pmu_frames([a, b], ["BUS10", "BUS39"])
    assert list(merged["TIMESTAMP"]) == [0.0, 0.1]
    assert "DATA_PRESENT_BUS10" in merged.columns
    assert "Event_BUS39" in merged.columns


def test_pmu_alignment_missing_timestamp_raises() -> None:
    bad = pd.DataFrame({"BUS10_VA_MAG": [1.0]})
    with pytest.raises(ValueError):
        align_pmu_frames([bad], ["BUS10"])

