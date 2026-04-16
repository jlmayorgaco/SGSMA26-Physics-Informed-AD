from __future__ import annotations

from enum import IntEnum


class EventId(IntEnum):
    NORMAL_OPERATION = 0
    FAULT = 1
    LINE_OUTAGE = 2
    GENERATION_CHANGE_OUTAGE = 3
    LOAD_CHANGE_DROP = 4
    MISSING_DATA = 5
    MISSING_DATA_PLUS_PHYSICAL_EVENT = 6
    BAD_DATA = 7
    UNKNOWN_EVENT = 8


class ReportScope(str):
    DATASET = "dataset"
    GENERAL = "general"
    BUS = "bus"
    EVENT = "event"
    EVENT_BUS = "event_bus"