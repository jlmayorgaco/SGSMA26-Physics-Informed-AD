from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.detectors.domain.enums.event_family import EventFamily


@dataclass(slots=True)
class WindowSample:
    scenario_id: str
    split: str
    window_index: int
    start_idx: int
    end_idx: int
    center_timestamp: float
    x: np.ndarray
    y_binary: int
    event_coarse: int
    event_family: EventFamily
    is_cyber_event: bool
    is_physical_event: bool
    is_concurrent_event: bool
    data_present_ratio: float
    subtype: str = ""
    origin: str = ""
    difficulty_level: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=float)
        if self.x.ndim != 2:
            raise ValueError("WindowSample.x must be 2D [window_size, n_features]")
        self.y_binary = int(self.y_binary)
        self.event_coarse = int(self.event_coarse)

