from __future__ import annotations

import numpy as np
import pandas as pd

from src.detectors.configs import WindowConfig
from src.detectors.models import WindowedBatch


class SlidingWindowBuilder:
    def __init__(self, config: WindowConfig) -> None:
        self.config = config

    def build(self, frame: pd.DataFrame, feature_columns: list[str], scenario_id: str) -> WindowedBatch:
        size = self.config.size
        stride = self.config.stride
        if len(frame) < size:
            x = np.zeros((0, size, len(feature_columns)), dtype=float)
            return WindowedBatch(
                x=x,
                y=np.zeros((0,), dtype=int),
                timestamps=np.zeros((0,), dtype=float),
                metadata=pd.DataFrame(columns=["scenario_id", "start_idx", "end_idx", "center_timestamp", "positive_ratio"]),
                feature_names=feature_columns,
            )
        arr = frame[feature_columns].to_numpy(dtype=float, copy=True)
        event = frame["EVENT"].to_numpy(dtype=int) if "EVENT" in frame.columns else np.zeros((len(frame),), dtype=int)
        ts = frame["TIMESTAMP"].to_numpy(dtype=float) if "TIMESTAMP" in frame.columns else np.arange(len(frame), dtype=float)
        windows: list[np.ndarray] = []
        y: list[int] = []
        t_center: list[float] = []
        meta_rows: list[dict[str, float | int | str]] = []
        for start in range(0, len(frame) - size + 1, stride):
            end = start + size
            w = arr[start:end]
            event_slice = event[start:end]
            pos_ratio = float((event_slice > 0).mean())
            label = int(pos_ratio >= self.config.positive_ratio_threshold)
            windows.append(w)
            y.append(label)
            center = start + size // 2
            t_center.append(float(ts[center]))
            if "ESTIMATOR_DIFFICULTY_SCORE" in frame.columns:
                est_score = float(pd.to_numeric(frame["ESTIMATOR_DIFFICULTY_SCORE"], errors="coerce").fillna(0.0).iloc[center])
            else:
                est_score = 0.0
            meta_rows.append(
                {
                    "scenario_id": scenario_id,
                    "start_idx": start,
                    "end_idx": end - 1,
                    "center_timestamp": float(ts[center]),
                    "positive_ratio": pos_ratio,
                    "event_max": int(event_slice.max(initial=0)),
                    "event_mean": float(event_slice.mean()),
                    "estimator_difficulty_score": est_score,
                    "event_coarse": float(pd.to_numeric(frame["EVENT_COARSE"], errors="coerce").fillna(np.nan).iloc[center]) if "EVENT_COARSE" in frame.columns else np.nan,
                    "difficulty_level": str(frame["DIFFICULTY_LEVEL"].iloc[center]) if "DIFFICULTY_LEVEL" in frame.columns else "",
                    "template_name": str(frame["TEMPLATE_NAME"].iloc[center]) if "TEMPLATE_NAME" in frame.columns else "",
                    "split": str(frame["SPLIT"].iloc[center]) if "SPLIT" in frame.columns else "",
                    "data_present_center": float(pd.to_numeric(frame["DATA_PRESENT"], errors="coerce").fillna(0.0).iloc[center]) if "DATA_PRESENT" in frame.columns else 0.0,
                }
            )
        x = np.stack(windows, axis=0) if windows else np.zeros((0, size, len(feature_columns)))
        return WindowedBatch(
            x=x.astype(float),
            y=np.asarray(y, dtype=int),
            timestamps=np.asarray(t_center, dtype=float),
            metadata=pd.DataFrame(meta_rows),
            feature_names=feature_columns,
        )


def concat_batches(batches: list[WindowedBatch]) -> WindowedBatch:
    if not batches:
        return WindowedBatch(
            x=np.zeros((0, 1, 1), dtype=float),
            y=np.zeros((0,), dtype=int),
            timestamps=np.zeros((0,), dtype=float),
            metadata=pd.DataFrame(),
            feature_names=[],
        )
    feature_names = batches[0].feature_names
    x = np.concatenate([b.x for b in batches], axis=0) if batches else np.zeros((0, 1, 1))
    y = np.concatenate([b.y for b in batches], axis=0) if batches else np.zeros((0,), dtype=int)
    ts = np.concatenate([b.timestamps for b in batches], axis=0) if batches else np.zeros((0,), dtype=float)
    meta = pd.concat([b.metadata for b in batches], ignore_index=True) if batches else pd.DataFrame()
    return WindowedBatch(x=x, y=y, timestamps=ts, metadata=meta, feature_names=feature_names)
