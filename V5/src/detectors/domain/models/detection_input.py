from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(slots=True)
class DetectionInput:
    scenario_id: str
    split: str
    x_windows: np.ndarray
    timestamps: np.ndarray
    feature_names: list[str]
    metadata: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self) -> None:
        self.x_windows = np.asarray(self.x_windows, dtype=float)
        self.timestamps = np.asarray(self.timestamps, dtype=float)
        if self.x_windows.ndim != 3:
            raise ValueError("x_windows must be 3D [n_windows, window_size, n_features]")
        if len(self.timestamps) != self.x_windows.shape[0]:
            raise ValueError("timestamps length must match n_windows")

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "split": self.split,
            "n_windows": int(self.x_windows.shape[0]),
            "window_size": int(self.x_windows.shape[1]) if self.x_windows.size else 0,
            "n_features": int(self.x_windows.shape[2]) if self.x_windows.size else 0,
            "feature_names": list(self.feature_names),
        }

