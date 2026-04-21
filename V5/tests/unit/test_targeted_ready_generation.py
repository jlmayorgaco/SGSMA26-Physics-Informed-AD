from __future__ import annotations

from collections import Counter

from src.simulation.m9.targeted_ready import _build_targeted_specs


def test_targeted_ready_specs_cover_requested_families() -> None:
    specs = _build_targeted_specs()
    counts = Counter(item.targeted_family for item in specs)
    assert counts["NORMAL_HARD_NEGATIVES"] == 60
    assert counts["CYBER_HEAVY_MISSING"] == 30
    assert counts["CYBER_HEAVY_BAD_DATA"] == 30
    assert counts["CONCURRENT_EXTRA"] == 40
    split_counts = Counter(item.split for item in specs)
    assert split_counts["train"] > split_counts["val"]
    assert split_counts["train"] > split_counts["test"]
