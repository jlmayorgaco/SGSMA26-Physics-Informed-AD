from __future__ import annotations

from enum import Enum


class EventFamily(str, Enum):
    NORMAL = "normal"
    PHYSICAL_HEAVY = "physical_heavy"
    CYBER_HEAVY = "cyber_heavy"
    CONCURRENT_OR_AMBIGUOUS = "concurrent_or_ambiguous"
    UNKNOWN = "unknown"

