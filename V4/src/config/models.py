from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd


@dataclass(slots=True)
class BusData:
    """
    Raw loaded PMU CSV after header normalization and numeric casting.
    """
    bus_id: str
    path: Path
    df: pd.DataFrame
    sampling_rate_hz: float


@dataclass(slots=True)
class EventSpan:
    """
    One contiguous run of a non-zero event label.
    """
    event_id: int
    label: str
    start_idx: int
    end_idx: int
    start_time: float
    end_time: float
    duration_s: float


@dataclass(slots=True)
class TimeWindow:
    """
    Generic time window helper for event slicing and report sections.
    """
    start_time: float
    end_time: float
    label: Optional[str] = None

    @property
    def duration_s(self) -> float:
        return float(self.end_time - self.start_time)


@dataclass(slots=True)
class OutputPaths:
    """
    Organized output folders for one analyzer run.
    """
    root: Path
    dataset_dir: Path
    general_dir: Path
    buses_dir: Path
    events_dir: Path


@dataclass(slots=True)
class DatasetSummary:
    """
    High-level metadata about a multi-bus PMU dataset.
    """
    input_dir: str
    file_count: int
    bus_ids: list[str]
    sampling_rates_hz: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BusSummary:
    """
    Generic per-bus analysis container.
    Keep it loose for now so refactors do not become painful too early.
    """
    bus_id: str
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EventSummary:
    """
    Generic event-level analysis container.
    """
    event_key: str
    event_id: int
    label: str
    bus_ids: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)