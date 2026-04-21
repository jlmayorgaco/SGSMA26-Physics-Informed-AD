from __future__ import annotations

from src.detectors.domain.enums.event_family import EventFamily
from src.detectors.shared.utils.label_mapping import event_to_binary_label, event_to_family, map_event_metadata


def test_binary_label_mapping() -> None:
    assert event_to_binary_label(0) == 0
    for event in range(1, 9):
        assert event_to_binary_label(event) == 1


def test_family_mapping_and_flags() -> None:
    assert event_to_family(0) == EventFamily.NORMAL
    assert event_to_family(2) == EventFamily.PHYSICAL_HEAVY
    assert event_to_family(5) == EventFamily.CYBER_HEAVY
    assert event_to_family(6) == EventFamily.CONCURRENT_OR_AMBIGUOUS
    meta = map_event_metadata(6, subtype="x", origin="BUS29")
    assert meta["is_physical_event"] is True
    assert meta["is_cyber_event"] is True
    assert meta["is_concurrent_event"] is True
    assert meta["subtype"] == "x"
    assert meta["origin"] == "BUS29"

