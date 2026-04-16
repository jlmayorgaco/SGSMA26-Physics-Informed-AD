from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.constants import MEASUREMENT_COLUMNS, TIMESTAMP_COLUMN
from src.utils.windows import contiguous_runs


def missing_profile(df: pd.DataFrame) -> dict:
    measurement_nan_mask = df[MEASUREMENT_COLUMNS].isna().any(axis=1).to_numpy(dtype=bool)

    data_present_mask = None
    if "DATA_PRESENT" in df.columns:
        data_present_mask = (df["DATA_PRESENT"].fillna(0).to_numpy(dtype=float) <= 0.5)

    combined_missing = measurement_nan_mask.copy()
    if data_present_mask is not None:
        combined_missing = combined_missing | data_present_mask

    runs = contiguous_runs(combined_missing)
    t = df[TIMESTAMP_COLUMN].to_numpy(dtype=float)

    gaps = []
    for s, e in runs:
        gaps.append({
            "start_idx": s,
            "end_idx": e - 1,
            "start_time": float(t[s]),
            "end_time": float(t[e - 1]),
            "duration_s": float(t[e - 1] - t[s]) if e - s > 1 else 0.0,
        })

    durations = [g["duration_s"] for g in gaps]

    return {
        "missing_frame_count": int(combined_missing.sum()),
        "missing_ratio": float(combined_missing.mean()),
        "gap_count": int(len(runs)),
        "gaps": gaps,
        "nan_frame_count": int(measurement_nan_mask.sum()),
        "data_present_zero_count": int(data_present_mask.sum()) if data_present_mask is not None else None,
        "mask_consistency_disagreement_count": int(np.sum(measurement_nan_mask != data_present_mask)) if data_present_mask is not None else None,
        "max_gap_seconds": float(max(durations)) if durations else 0.0,
        "median_gap_seconds": float(np.median(durations)) if durations else 0.0,
    }