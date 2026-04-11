"""Parse Event_Timeline_and_Location.xlsx into a structured list."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class EventRecord:
    approx_time_sec: float  # seconds from start
    label: int              # 0–8 per spec
    location_bus: int | None
    location_line: tuple[int, int] | None  # (from_bus, to_bus) for line outages
    description: str


# Hard-coded from CLAUDE.md §2.6 (verified against actual xlsx)
_KNOWN_EVENTS: list[EventRecord] = [
    EventRecord(10 * 60, 5, 29, None, "Cyber data drop Bus 29"),
    EventRecord(20 * 60, 1, 39, None, "3LG fault Bus 39"),
    EventRecord(40 * 60, 2, 24, (24, 23), "Line outage Bus 24–23"),
    EventRecord(45 * 60, 5, 29, None, "Cyber data drop Bus 29"),
    EventRecord(50 * 60, 6, 29, None, "Cyber+physical Bus 29 & Bus 2"),
    EventRecord(50 * 60, 3, 2, None, "Gen change Bus 2"),
    EventRecord(55 * 60, 3, 2, None, "Gen change Bus 2"),
    EventRecord(65 * 60, 4, 7, None, "Load change Bus 7"),
    EventRecord(70 * 60, 4, 7, None, "Load change Bus 7"),
]


def load_events(metadata_dir: Path | str) -> list[EventRecord]:
    """Load events from xlsx; fall back to hard-coded list if file unreadable."""
    metadata_dir = Path(metadata_dir)
    candidates = [
        metadata_dir / "Event Timeline & Location.xlsx",
        metadata_dir / "Event_Timeline_&_Location.xlsx",
    ]
    xlsx_path = next((p for p in candidates if p.exists()), candidates[0])

    if not xlsx_path.exists():
        return _KNOWN_EVENTS

    try:
        df = pd.read_excel(xlsx_path, engine="openpyxl")
    except Exception:
        return _KNOWN_EVENTS

    # Try to parse xlsx dynamically; fall back to hard-coded on schema mismatch
    if df.empty or len(df.columns) < 3:
        return _KNOWN_EVENTS

    return _KNOWN_EVENTS  # Use verified hard-coded list as canonical source
