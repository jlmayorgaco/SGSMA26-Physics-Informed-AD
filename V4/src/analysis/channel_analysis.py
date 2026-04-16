
# -----------------------------------------------------------------------------
# Core analysis helpers
# -----------------------------------------------------------------------------

from typing import Any

import numpy as np
import pandas as pd

from src.analysis.events import event_channel_profile, event_spans
from src.analysis.hilbert_analysis import hilbert_features
from src.analysis.noise import noise_model
from src.analysis.spectral import spectral_summary
from src.analysis.stats import basic_stats
from src.config.constants import MEASUREMENT_COLUMNS


def compute_dt(df: pd.DataFrame) -> float:
    t = df["TIMESTAMP"].to_numpy(dtype=float)
    if len(t) < 2:
        return 1.0
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return 1.0
    return float(np.median(dt))



def compute_channel_analysis(
    df: pd.DataFrame,
    label_map: dict[int, str],
    event_baseline_seconds: float,
    event_post_seconds: float,
    trend_window_seconds: float,
) -> dict[str, Any]:
    dt = compute_dt(df)
    spans = event_spans(df, label_map)
    channel_payload: dict[str, Any] = {}

    for col in MEASUREMENT_COLUMNS:
        values = df[col].to_numpy(dtype=float)
        signal_hilbert = hilbert_features(values)
        channel_payload[col] = {
            "basic": basic_stats(values, dt, col),
            "noise": noise_model(values, dt, col, trend_window_seconds),
            "spectral": spectral_summary(values, 1.0 / dt if dt > 0 else np.nan),
            "hilbert": {
                k: v
                for k, v in signal_hilbert.items()
                if not str(k).endswith("_series") and not str(k).endswith("_series_rad")
            },
            "event_response": event_channel_profile(
                df=df,
                column=col,
                spans=spans,
                dt=dt,
                baseline_seconds=event_baseline_seconds,
                post_seconds=event_post_seconds,
            ),
        }
    return channel_payload