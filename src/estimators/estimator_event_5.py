from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.helpers.raw_loader import measurement_columns


@dataclass(slots=True)
class Event5EstimatorConfig:
    bus_missing_threshold: float = 0.80
    use_data_present: bool = True


@dataclass(slots=True)
class Event5EstimatorResult:
    frame_score: pd.Series
    frame_prediction: pd.Series
    per_bus_score: pd.DataFrame
    per_bus_missing: pd.DataFrame


@dataclass(slots=True)
class Event5Estimator:
    config: Event5EstimatorConfig = field(default_factory=Event5EstimatorConfig)

    def estimate(self, aligned_frames: dict[str, pd.DataFrame]) -> Event5EstimatorResult:
        if not aligned_frames:
            raise ValueError("aligned_frames is empty.")
        timeline = next(iter(aligned_frames.values())).index
        per_bus_score: dict[str, pd.Series] = {}

        for bus, frame in sorted(aligned_frames.items()):
            cols = measurement_columns(frame)
            if not cols:
                raise ValueError(f"{bus} has no PMU measurement columns.")
            nan_fraction = frame[cols].isna().mean(axis=1).astype(float)
            if self.config.use_data_present and "DATA_PRESENT" in frame.columns:
                data_present = pd.to_numeric(frame["DATA_PRESENT"], errors="coerce")
                data_present_missing = (data_present.fillna(0.0) <= 0.0).astype(float)
            else:
                data_present_missing = pd.Series(np.zeros(len(frame), dtype=float), index=frame.index)
            score = np.maximum(nan_fraction.to_numpy(dtype=float), data_present_missing.to_numpy(dtype=float))
            per_bus_score[bus] = pd.Series(score, index=timeline, name=bus)

        score_df = pd.DataFrame(per_bus_score, index=timeline).sort_index()
        frame_score = score_df.max(axis=1).astype(float).rename("event5_score")
        frame_prediction = (frame_score >= float(self.config.bus_missing_threshold)).astype(int).rename("pred_event5")
        per_bus_missing = (score_df >= float(self.config.bus_missing_threshold)).astype(int)
        return Event5EstimatorResult(
            frame_score=frame_score,
            frame_prediction=frame_prediction,
            per_bus_score=score_df,
            per_bus_missing=per_bus_missing,
        )
