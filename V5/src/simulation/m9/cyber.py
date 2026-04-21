"""Cyber/data-quality operators for observed M9 PMU streams."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.simulation.m9.constants import BAD_DATA_EVENT_TYPES, MISSING_EVENT_TYPES, PHYSICAL_EVENT_LABELS, PMU_MEASUREMENT_SUFFIXES


def measurement_columns(bus: str) -> list[str]:
    return [f"{bus}_{suffix}" for suffix in PMU_MEASUREMENT_SUFFIXES]


def selected_columns(df: pd.DataFrame, bus: str, target_channels: list[str] | None) -> list[str]:
    all_cols = measurement_columns(bus)
    if not target_channels or "ALL" in {str(c).upper() for c in target_channels}:
        return [c for c in all_cols if c in df.columns]
    wanted = {f"{bus}_{c}" if not str(c).upper().startswith(bus) else str(c) for c in target_channels}
    return [c for c in all_cols if c in df.columns and c in wanted]


def event_mask(df: pd.DataFrame, event: dict[str, Any]) -> pd.Series:
    t = pd.to_numeric(df["TIMESTAMP"], errors="coerce")
    start = float(event.get("start_time_s", 0.0))
    end = float(event.get("end_time_s", start + float(event.get("duration_s", 0.0))))
    return (t >= start) & (t <= end)


def is_missing_event(event: dict[str, Any]) -> bool:
    et = str(event.get("event_type", "")).lower()
    sub = str(event.get("subtype", "")).lower()
    return et in MISSING_EVENT_TYPES or sub in MISSING_EVENT_TYPES


def is_bad_data_event(event: dict[str, Any]) -> bool:
    et = str(event.get("event_type", "")).lower()
    sub = str(event.get("subtype", "")).lower()
    return et in BAD_DATA_EVENT_TYPES or sub in BAD_DATA_EVENT_TYPES


def apply_missing_data(df: pd.DataFrame, bus: str, event: dict[str, Any], physical_label: np.ndarray | None = None) -> pd.DataFrame:
    out = df.copy()
    mask = event_mask(out, event)
    subtype = str(event.get("subtype", "full_dropout")).lower()
    params = dict(event.get("params", {}))
    if subtype == "periodic_dropout":
        period = max(1, int(params.get("period_frames", 5)))
        width = max(1, int(params.get("width_frames", 1)))
        idx = np.arange(len(out))
        local = mask.to_numpy() & ((idx % period) < width)
        mask = pd.Series(local, index=out.index)
    elif subtype == "burst_dropout":
        # The template interval itself is the burst; this branch documents intent.
        mask = mask
    cols = selected_columns(out, bus, event.get("target_channels"))
    if subtype in {"full_dropout", "burst_dropout", "periodic_dropout"}:
        cols = [c for c in measurement_columns(bus) if c in out.columns]
    out.loc[mask, cols] = np.nan
    if subtype in {"full_dropout", "burst_dropout", "periodic_dropout"} or set(cols) == set(measurement_columns(bus)):
        out.loc[mask, "DATA_PRESENT"] = 0
    if physical_label is not None:
        phys = np.asarray(physical_label, dtype=int)
        concurrent = mask.to_numpy() & (phys != 0)
        missing_only = mask.to_numpy() & (phys == 0)
        out.loc[missing_only, "Event"] = 5
        out.loc[concurrent, "Event"] = 6
    else:
        out.loc[mask, "Event"] = int(event.get("event_label", 5))
    return out


def apply_bad_data(df: pd.DataFrame, bus: str, event: dict[str, Any], seed: int = 0) -> pd.DataFrame:
    out = df.copy()
    rng = np.random.default_rng(seed)
    mask = event_mask(out, event)
    cols = selected_columns(out, bus, event.get("target_channels"))
    subtype = str(event.get("subtype", "spike")).lower()
    params = dict(event.get("params", {}))
    idx = np.where(mask.to_numpy())[0]
    if not cols or idx.size == 0:
        return out
    for col in cols:
        x = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)
        clean_std = float(np.nanstd(x)) if np.isfinite(np.nanstd(x)) and np.nanstd(x) > 0 else 1.0
        if subtype == "spike":
            amplitude = float(params.get("amplitude", 6.0)) * clean_std
            signs = rng.choice([-1.0, 1.0], size=idx.size)
            x[idx] = x[idx] + signs * amplitude
        elif subtype == "bias":
            bias = float(params.get("bias", 3.0 if col.endswith("ANG") else 0.03 * np.nanmean(np.abs(x))))
            x[idx] = x[idx] + bias
        elif subtype == "drift":
            drift = np.linspace(0.0, float(params.get("drift", 4.0)), idx.size)
            x[idx] = x[idx] + drift
        elif subtype == "stuck_at_last_value":
            first = max(idx[0] - 1, 0)
            x[idx] = x[first]
        elif subtype == "gain_error":
            x[idx] = x[idx] * float(params.get("gain", 1.2))
        elif subtype == "clipping":
            lo = float(params.get("min", np.nanpercentile(x, 5)))
            hi = float(params.get("max", np.nanpercentile(x, 95)))
            x[idx] = np.clip(x[idx], lo, hi)
        elif subtype == "replay_window":
            lag = max(1, int(params.get("lag_frames", min(10, idx[0]))))
            src = np.maximum(idx - lag, 0)
            x[idx] = x[src]
        elif subtype == "angle_wrap_corruption":
            if col.endswith("ANG"):
                x[idx] = ((x[idx] + 180.0) % 360.0) - 180.0 + 360.0
            else:
                x[idx] = x[idx] + 5.0 * clean_std
        else:
            x[idx] = x[idx] + rng.normal(0.0, 4.0 * clean_std, size=idx.size)
        out[col] = x
    if subtype == "channel_swap" and len(cols) >= 2:
        c1, c2 = cols[0], cols[1]
        tmp = out.loc[mask, c1].copy()
        out.loc[mask, c1] = out.loc[mask, c2]
        out.loc[mask, c2] = tmp
    out.loc[mask, "DATA_PRESENT"] = out.loc[mask, "DATA_PRESENT"].fillna(1).replace(0, 1)
    out.loc[mask, "Event"] = int(event.get("event_label", 7))
    return out


def apply_timing_attack(df: pd.DataFrame, bus: str, event: dict[str, Any], seed: int = 0, preserve_timestamp_alignment: bool = True) -> pd.DataFrame:
    out = df.copy()
    rng = np.random.default_rng(seed)
    mask = event_mask(out, event)
    subtype = str(event.get("subtype", "fixed_delay")).lower()
    params = dict(event.get("params", {}))
    cols = selected_columns(out, bus, event.get("target_channels"))
    if not cols:
        cols = [c for c in measurement_columns(bus) if c in out.columns]
    idx = np.where(mask.to_numpy())[0]
    if idx.size == 0:
        return out
    if subtype == "timestamp_jitter" and not preserve_timestamp_alignment:
        jitter = rng.normal(0.0, float(params.get("std_s", 0.004)), size=idx.size)
        out.loc[out.index[idx], "TIMESTAMP"] = pd.to_numeric(out.loc[out.index[idx], "TIMESTAMP"], errors="coerce") + jitter
    elif subtype in {"fixed_delay", "variable_delay"}:
        max_delay = int(params.get("delay_frames", 2))
        for col in cols:
            values = out[col].to_numpy(copy=True)
            delayed = values.copy()
            for k in idx:
                lag = max_delay if subtype == "fixed_delay" else int(rng.integers(1, max_delay + 1))
                delayed[k] = values[max(k - lag, 0)]
            out[col] = delayed
    elif subtype == "duplicated_frames":
        for col in cols:
            values = out[col].to_numpy(copy=True)
            values[idx[1::2]] = values[idx[::2][: len(idx[1::2])]]
            out[col] = values
    elif subtype == "frame_reordering":
        shuffled = idx.copy()
        rng.shuffle(shuffled)
        for col in cols:
            values = out[col].to_numpy(copy=True)
            values[idx] = values[shuffled]
            out[col] = values
    out.loc[mask, "DATA_PRESENT"] = out.loc[mask, "DATA_PRESENT"].fillna(1).replace(0, 1)
    out.loc[mask, "Event"] = int(event.get("event_label", 7))
    return out


def apply_cyber_events(
    pmu_frames: dict[str, pd.DataFrame],
    cyber_events: list[dict[str, Any]],
    physical_labels: np.ndarray,
    seed: int = 12345,
    preserve_timestamp_alignment: bool = True,
) -> dict[str, pd.DataFrame]:
    out = {bus: df.copy() for bus, df in pmu_frames.items()}
    for k, event in enumerate(cyber_events):
        for bus in event.get("target_pmus", []):
            if bus not in out:
                continue
            if is_missing_event(event):
                out[bus] = apply_missing_data(out[bus], bus, event, physical_labels)
            elif str(event.get("subtype", "")).lower() in {"fixed_delay", "variable_delay", "timestamp_jitter", "duplicated_frames", "frame_reordering"}:
                out[bus] = apply_timing_attack(out[bus], bus, event, seed=seed + k, preserve_timestamp_alignment=preserve_timestamp_alignment)
            else:
                out[bus] = apply_bad_data(out[bus], bus, event, seed=seed + k)
    return out


def build_cyber_frame_metadata(timestamps: np.ndarray, cyber_events: list[dict[str, Any]]) -> tuple[np.ndarray, list[str], list[str], list[str]]:
    t = np.asarray(timestamps, dtype=float)
    labels = np.zeros(len(t), dtype=int)
    types = [""] * len(t)
    subtypes = [""] * len(t)
    target_pmus = [""] * len(t)
    for event in cyber_events:
        start = float(event.get("start_time_s", 0.0))
        end = float(event.get("end_time_s", start + float(event.get("duration_s", 0.0))))
        mask = (t >= start) & (t <= end)
        label = int(event.get("event_label", 5 if is_missing_event(event) else 7))
        for i in np.where(mask)[0]:
            labels[i] = label
            types[i] = str(event.get("event_type", ""))
            subtypes[i] = str(event.get("subtype", ""))
            target_pmus[i] = ";".join(str(x) for x in event.get("target_pmus", []))
    return labels, types, subtypes, target_pmus


def combine_event_label(physical_label: int, cyber_label: int, cyber_type: str = "") -> int:
    if cyber_label == 8 or physical_label == 8:
        return 8
    if physical_label and cyber_label:
        if cyber_type in MISSING_EVENT_TYPES or cyber_label == 5 or cyber_label == 6:
            return 6
        return 8
    if cyber_label:
        return int(cyber_label)
    if physical_label:
        return int(physical_label)
    return 0
