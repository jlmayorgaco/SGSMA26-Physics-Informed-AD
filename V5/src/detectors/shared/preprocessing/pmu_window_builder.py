from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.detectors.domain.models.detection_input import DetectionInput
from src.detectors.domain.models.window_sample import WindowSample
from src.detectors.shared.utils.label_mapping import map_event_metadata


@dataclass(slots=True)
class PMUWindowConfig:
    window_size: int = 64
    stride: int = 16
    positive_ratio_threshold: float = 0.10


class PMUWindowBuilder:
    def __init__(self, config: PMUWindowConfig) -> None:
        self.config = config

    def build(
        self,
        frame: pd.DataFrame,
        *,
        feature_columns: Sequence[str],
        scenario_id: str,
        split: str,
        default_metadata: dict[str, Any] | None = None,
    ) -> list[WindowSample]:
        if len(frame) < self.config.window_size:
            return []
        default_metadata = default_metadata or {}
        ts = (
            pd.to_numeric(frame["TIMESTAMP"], errors="coerce").ffill().bfill().to_numpy(dtype=float)
            if "TIMESTAMP" in frame.columns
            else np.arange(len(frame), dtype=float)
        )
        x_full = frame[list(feature_columns)].to_numpy(dtype=float, copy=True)
        events = pd.to_numeric(frame.get("EVENT", 0), errors="coerce").fillna(0).astype(int).to_numpy()
        data_present = (
            pd.to_numeric(frame.get("DATA_PRESENT", 1.0), errors="coerce").fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
        )
        subtypes = frame.get("CYBER_SUBTYPE", pd.Series([""] * len(frame))).astype(str).to_numpy()
        origins = frame.get("TARGET_PMU", pd.Series([""] * len(frame))).astype(str).to_numpy()
        difficulties = frame.get("DIFFICULTY_LEVEL", pd.Series([""] * len(frame))).astype(str).to_numpy()
        phy_flags = (
            frame.get("IS_PHYSICAL_EVENT", pd.Series([False] * len(frame))).astype(bool).to_numpy()
            if "IS_PHYSICAL_EVENT" in frame.columns
            else np.array([False] * len(frame), dtype=bool)
        )
        cyber_flags = (
            frame.get("IS_CYBER_EVENT", pd.Series([False] * len(frame))).astype(bool).to_numpy()
            if "IS_CYBER_EVENT" in frame.columns
            else np.array([False] * len(frame), dtype=bool)
        )
        conc_flags = (
            frame.get("IS_CONCURRENT_EVENT", pd.Series([False] * len(frame))).astype(bool).to_numpy()
            if "IS_CONCURRENT_EVENT" in frame.columns
            else np.array([False] * len(frame), dtype=bool)
        )
        samples: list[WindowSample] = []
        idx = 0
        for start in range(0, len(frame) - self.config.window_size + 1, self.config.stride):
            end = start + self.config.window_size
            window_x = x_full[start:end]
            event_slice = events[start:end]
            data_slice = data_present[start:end]
            ratio = float((event_slice > 0).mean())
            y_binary = int(ratio >= self.config.positive_ratio_threshold)
            center = start + self.config.window_size // 2
            event_coarse = int(np.max(event_slice))
            flags = map_event_metadata(
                event_coarse,
                subtype=str(subtypes[center]),
                origin=str(origins[center]),
                is_physical_event=bool(np.any(phy_flags[start:end])),
                is_cyber_event=bool(np.any(cyber_flags[start:end])),
                is_concurrent_event=bool(np.any(conc_flags[start:end])),
            )
            metadata = {
                "positive_ratio": ratio,
                "window_size": self.config.window_size,
                **default_metadata,
            }
            samples.append(
                WindowSample(
                    scenario_id=scenario_id,
                    split=split,
                    window_index=idx,
                    start_idx=start,
                    end_idx=end - 1,
                    center_timestamp=float(ts[center]),
                    x=window_x,
                    y_binary=y_binary,
                    event_coarse=event_coarse,
                    event_family=flags["event_family"],
                    is_cyber_event=flags["is_cyber_event"],
                    is_physical_event=flags["is_physical_event"],
                    is_concurrent_event=flags["is_concurrent_event"],
                    data_present_ratio=float(np.mean(data_slice)),
                    subtype=flags["subtype"],
                    origin=flags["origin"],
                    difficulty_level=str(difficulties[center]),
                    metadata=metadata,
                )
            )
            idx += 1
        return samples


def collate_samples_to_detection_input(
    samples: Sequence[WindowSample],
    feature_names: list[str],
    *,
    scenario_id: str,
    split: str,
) -> DetectionInput:
    if not samples:
        return DetectionInput(
            scenario_id=scenario_id,
            split=split,
            x_windows=np.zeros((0, 1, max(len(feature_names), 1)), dtype=float),
            timestamps=np.zeros((0,), dtype=float),
            feature_names=list(feature_names),
            metadata=pd.DataFrame(),
        )
    x = np.stack([s.x for s in samples], axis=0).astype(float)
    ts = np.asarray([s.center_timestamp for s in samples], dtype=float)
    metadata = pd.DataFrame(
        [
            {
                "scenario_id": s.scenario_id,
                "split": s.split,
                "window_index": s.window_index,
                "start_idx": s.start_idx,
                "end_idx": s.end_idx,
                "event_coarse": s.event_coarse,
                "y_binary": s.y_binary,
                "event_family": s.event_family.value,
                "is_cyber_event": s.is_cyber_event,
                "is_physical_event": s.is_physical_event,
                "is_concurrent_event": s.is_concurrent_event,
                "data_present_ratio": s.data_present_ratio,
                "subtype": s.subtype,
                "origin": s.origin,
                "difficulty_level": s.difficulty_level,
                **s.metadata,
            }
            for s in samples
        ]
    )
    return DetectionInput(
        scenario_id=scenario_id,
        split=split,
        x_windows=x,
        timestamps=ts,
        feature_names=list(feature_names),
        metadata=metadata,
    )
