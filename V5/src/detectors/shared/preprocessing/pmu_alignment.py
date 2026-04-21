from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


def _bus_token_from_name(path: Path) -> str:
    match = re.search(r"Bus(\d+)", path.name, flags=re.IGNORECASE)
    if not match:
        return "BUSUNK"
    return f"BUS{int(match.group(1))}"


def align_pmu_frames(frames: list[pd.DataFrame], bus_tokens: list[str] | None = None) -> pd.DataFrame:
    if not frames:
        raise ValueError("frames cannot be empty")
    if bus_tokens is None:
        bus_tokens = [f"BUS{i}" for i in range(len(frames))]
    if len(bus_tokens) != len(frames):
        raise ValueError("bus_tokens length must match frames length")
    merged: pd.DataFrame | None = None
    for frame, bus in zip(frames, bus_tokens):
        if "TIMESTAMP" not in frame.columns:
            raise ValueError(f"TIMESTAMP column missing for {bus}")
        local = frame.copy()
        local["TIMESTAMP"] = pd.to_numeric(local["TIMESTAMP"], errors="coerce")
        keep = [c for c in local.columns if c != "TIMESTAMP"]
        rename: dict[str, str] = {}
        for col in keep:
            if col in {"DATA_PRESENT", "Event"}:
                rename[col] = f"{col}_{bus}"
        bus_frame = local[["TIMESTAMP"] + keep].rename(columns=rename)
        merged = bus_frame if merged is None else merged.merge(bus_frame, on="TIMESTAMP", how="outer")
    assert merged is not None
    merged = merged.dropna(subset=["TIMESTAMP"]).sort_values("TIMESTAMP")
    merged = merged.drop_duplicates(subset=["TIMESTAMP"]).reset_index(drop=True)
    return merged


def align_scenario_pmu_dir(pmu_dir: Path, pattern: str = "Bus*_Competition_Data_*.csv") -> pd.DataFrame:
    files = sorted(pmu_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No PMU csv files found in {pmu_dir}")
    frames = [pd.read_csv(path) for path in files]
    buses = [_bus_token_from_name(path) for path in files]
    merged = align_pmu_frames(frames, buses)
    data_present_cols = [c for c in merged.columns if c.startswith("DATA_PRESENT_")]
    event_cols = [c for c in merged.columns if c.startswith("Event_")]
    if data_present_cols:
        merged["DATA_PRESENT"] = (
            merged[data_present_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .min(axis=1)
            .clip(0.0, 1.0)
        )
    else:
        merged["DATA_PRESENT"] = 1.0
    if event_cols:
        merged["EVENT"] = (
            merged[event_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .astype(int)
            .max(axis=1)
        )
    else:
        merged["EVENT"] = 0
    return merged

