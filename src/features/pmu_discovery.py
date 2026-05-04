from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd


BUS_COLUMN_RE = re.compile(r"(?:^|__)BUS(\d+)(?:_|__|$)", re.IGNORECASE)
BUS_FILE_RE = re.compile(r"bus\s*([0-9]+)", re.IGNORECASE)
PMU_COLUMN_PREFIX_RE = re.compile(r"^(?:BUS(\d+)(?:_|__)|(?:ROLL|RLS|HILB|META)__BUS(\d+)__)", re.IGNORECASE)


def infer_pmu_buses_from_columns(columns: Iterable[object]) -> tuple[int, ...]:
    buses: set[int] = set()
    for column in columns:
        match = PMU_COLUMN_PREFIX_RE.search(str(column))
        if match:
            buses.add(int(match.group(1) or match.group(2)))
    return tuple(sorted(buses))


def infer_pmu_buses_from_frames(frames_by_bus: dict[int, pd.DataFrame]) -> tuple[int, ...]:
    buses = {int(bus) for bus, frame in frames_by_bus.items() if frame is not None and not frame.empty}
    for frame in frames_by_bus.values():
        if frame is not None and not frame.empty:
            buses.update(infer_pmu_buses_from_columns(frame.columns))
    return tuple(sorted(buses))


def bus_from_filename(path: Path) -> int | None:
    match = BUS_FILE_RE.search(path.name)
    return int(match.group(1)) if match else None


def discover_bus_csvs(input_dir: Path, patterns: tuple[str, ...] = ("Bus*.csv", "bus*.csv")) -> dict[int, Path]:
    found: dict[int, Path] = {}
    for pattern in patterns:
        for path in sorted(Path(input_dir).glob(pattern)):
            bus = bus_from_filename(path)
            if bus is not None and bus not in found:
                found[bus] = path
    return dict(sorted(found.items()))
