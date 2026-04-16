from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.constants import CORE_CROSS_BUS_COLUMNS, TIMESTAMP_COLUMN
from src.config.models import BusData
from src.utils.numeric import mad
from src.utils.signals import unwrap_if_angle


def make_merged_frame(buses: list[BusData]) -> pd.DataFrame:
    merged = None

    for bus in buses:
        cols = [TIMESTAMP_COLUMN] + [c for c in CORE_CROSS_BUS_COLUMNS if c in bus.df.columns]
        bus_df = bus.df[cols].copy()
        bus_df = bus_df.rename(columns={c: f"{bus.bus_id}__{c}" for c in cols if c != TIMESTAMP_COLUMN})
        merged = bus_df if merged is None else merged.merge(bus_df, on=TIMESTAMP_COLUMN, how="inner")

    if merged is None:
        return pd.DataFrame(columns=[TIMESTAMP_COLUMN])

    return merged


def cross_bus_correlations(buses: list[BusData]) -> dict:
    merged = make_merged_frame(buses)
    out = {}

    for col in CORE_CROSS_BUS_COLUMNS:
        cols = [f"{bus.bus_id}__{col}" for bus in buses if f"{bus.bus_id}__{col}" in merged.columns]
        if not cols:
            out[col] = {}
            continue
        frame = merged[cols]
        out[col] = frame.corr().replace({np.nan: None}).to_dict()

    return out


def cross_bus_rankings(buses: list[BusData], spans: list[dict]) -> dict:
    outputs = {c: [] for c in CORE_CROSS_BUS_COLUMNS}

    for sp in spans:
        s, e = int(sp["start_idx"]), int(sp["end_idx"]) + 1

        for col in CORE_CROSS_BUS_COLUMNS:
            ranked = []

            for bus in buses:
                if col not in bus.df.columns:
                    continue

                series = unwrap_if_angle(bus.df[col].to_numpy(dtype=float), col)
                baseline = series[max(0, s - 60):s]
                during = series[s:e]

                if np.isfinite(baseline).sum() < 3 or np.isfinite(during).sum() < 1:
                    continue

                mu = float(np.nanmedian(baseline))
                sigma = float(mad(baseline))
                if not np.isfinite(sigma) or sigma <= 1e-12:
                    sigma = float(np.nanstd(baseline))
                if not np.isfinite(sigma) or sigma <= 1e-12:
                    sigma = 1.0

                score = float(np.nanmax(np.abs((during - mu) / sigma)))
                ranked.append({"bus": bus.bus_id, "score": score})

            ranked.sort(key=lambda d: d["score"], reverse=True)

            outputs[col].append({
                "event_id": int(sp["event_id"]),
                "label": sp["label"],
                "start_time": float(sp["start_time"]),
                "end_time": float(sp["end_time"]),
                "value_suffix": col,
                "ranked_buses_by_peak_abs_zscore": ranked,
            })

    return outputs