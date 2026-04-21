from __future__ import annotations

from src.detectors.pipelines.raw_holdout_loader import event_family_from_event, event_to_binary


def test_raw_binary_label_mapping() -> None:
    assert event_to_binary(0) == 0
    assert event_to_binary("0") == 0
    for value in range(1, 9):
        assert event_to_binary(value) == 1
    assert event_to_binary(None) == 0
    assert event_to_binary("bad") == 0

    assert event_family_from_event(0) == "normal"
    assert event_family_from_event(1) == "physical_heavy"
    assert event_family_from_event(5) == "cyber_heavy"
    assert event_family_from_event(6) == "concurrent_heavy"
    assert event_family_from_event(8) == "concurrent_heavy"
