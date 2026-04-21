from __future__ import annotations

from typing import Any

from src.detectors.domain.enums.event_family import EventFamily

PHYSICAL_EVENTS = {1, 2, 3, 4}
CYBER_EVENTS = {5, 7}
CONCURRENT_OR_AMBIGUOUS_EVENTS = {6, 8}


def event_to_binary_label(event_coarse: int | float | None) -> int:
    event = int(event_coarse or 0)
    return 0 if event == 0 else 1


def event_to_family(event_coarse: int | float | None) -> EventFamily:
    event = int(event_coarse or 0)
    if event == 0:
        return EventFamily.NORMAL
    if event in PHYSICAL_EVENTS:
        return EventFamily.PHYSICAL_HEAVY
    if event in CYBER_EVENTS:
        return EventFamily.CYBER_HEAVY
    if event in CONCURRENT_OR_AMBIGUOUS_EVENTS:
        return EventFamily.CONCURRENT_OR_AMBIGUOUS
    return EventFamily.UNKNOWN


def infer_event_flags(
    event_coarse: int | float | None,
    *,
    is_physical_event: bool | None = None,
    is_cyber_event: bool | None = None,
    is_concurrent_event: bool | None = None,
) -> dict[str, bool]:
    event = int(event_coarse or 0)
    default_physical = event in PHYSICAL_EVENTS or event in CONCURRENT_OR_AMBIGUOUS_EVENTS
    default_cyber = event in CYBER_EVENTS or event in CONCURRENT_OR_AMBIGUOUS_EVENTS
    default_concurrent = event in CONCURRENT_OR_AMBIGUOUS_EVENTS
    return {
        "is_physical_event": bool(default_physical if is_physical_event is None else is_physical_event),
        "is_cyber_event": bool(default_cyber if is_cyber_event is None else is_cyber_event),
        "is_concurrent_event": bool(default_concurrent if is_concurrent_event is None else is_concurrent_event),
    }


def map_event_metadata(
    event_coarse: int | float | None,
    *,
    subtype: str = "",
    origin: str = "",
    is_physical_event: bool | None = None,
    is_cyber_event: bool | None = None,
    is_concurrent_event: bool | None = None,
) -> dict[str, Any]:
    flags = infer_event_flags(
        event_coarse,
        is_physical_event=is_physical_event,
        is_cyber_event=is_cyber_event,
        is_concurrent_event=is_concurrent_event,
    )
    event = int(event_coarse or 0)
    return {
        "event_coarse": event,
        "y_binary": event_to_binary_label(event),
        "event_family": event_to_family(event),
        "subtype": str(subtype),
        "origin": str(origin),
        **flags,
    }

