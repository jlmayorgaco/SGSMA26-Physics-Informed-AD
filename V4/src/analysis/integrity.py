from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.constants import TIMESTAMP_COLUMN
from src.config.models import BusData


def timestamp_integrity(df: pd.DataFrame) -> dict:
    t = df[TIMESTAMP_COLUMN].to_numpy(dtype=float)
    dt = np.diff(t)
    valid_dt = dt[np.isfinite(dt)]

    if valid_dt.size == 0:
        return {}

    median_dt = float(np.median(valid_dt))

    return {
        "row_count": int(len(t)),
        "timestamp_start": float(t[0]),
        "timestamp_end": float(t[-1]),
        "dt_mean_s": float(np.mean(valid_dt)),
        "dt_median_s": median_dt,
        "dt_std_s": float(np.std(valid_dt, ddof=1)) if valid_dt.size > 1 else 0.0,
        "sampling_rate_hz": float(1.0 / median_dt) if median_dt > 0 else None,
        "duplicate_timestamp_count": int(pd.Series(t).duplicated().sum()),
        "non_monotonic_count": int(np.sum(valid_dt <= 0)),
        "timestamp_jitter_std_s": float(np.std(valid_dt - median_dt, ddof=1)) if valid_dt.size > 1 else 0.0,
    }


def align_integrity(buses: list[BusData]) -> dict:
    if not buses:
        return {}

    ref = buses[0].df[TIMESTAMP_COLUMN].to_numpy(dtype=float)

    out = {
        "reference_bus": buses[0].bus_id,
        "reference_rows": int(len(ref)),
        "per_bus_alignment": {},
    }

    for bus in buses:
        t = bus.df[TIMESTAMP_COLUMN].to_numpy(dtype=float)
        min_len = min(len(ref), len(t))

        exact_match_len = int(np.sum(np.isclose(ref[:min_len], t[:min_len], atol=1e-9, rtol=0.0)))
        common = len(np.intersect1d(ref, t))

        out["per_bus_alignment"][bus.bus_id] = {
            "row_count": int(len(t)),
            "common_timestamp_count": int(common),
            "exact_prefix_match_count": exact_match_len,
            "prefix_match_ratio": float(exact_match_len / max(min_len, 1)),
            "full_length_equal": bool(len(ref) == len(t) and exact_match_len == len(ref)),
        }

    return out