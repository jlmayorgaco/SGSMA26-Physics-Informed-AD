"""Extract event transition timestamps and produce labeled interval tables."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class EventInterval:
    start_ts: float
    end_ts: float
    label: int
    location_bus: int | None
    location_line: tuple[int, int] | None


def extract_transitions(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows where Event value changes.

    Args:
        df: merged DataFrame with TIMESTAMP and Event columns

    Returns:
        DataFrame with columns [TIMESTAMP, Event] at transition points.
    """
    mask = df["Event"].diff().fillna(0) != 0
    return df.loc[mask, ["TIMESTAMP", "Event"]].copy()


def build_intervals(df: pd.DataFrame) -> list[EventInterval]:
    """Convert transition rows into a list of (start, end, label) intervals.

    The last interval extends to the final timestamp.
    """
    transitions = extract_transitions(df)
    intervals: list[EventInterval] = []

    timestamps = transitions["TIMESTAMP"].tolist()
    labels = transitions["Event"].tolist()

    all_ts = df["TIMESTAMP"].tolist()
    final_ts = all_ts[-1]

    for i, (ts, lbl) in enumerate(zip(timestamps, labels)):
        end = timestamps[i + 1] if i + 1 < len(timestamps) else final_ts
        intervals.append(
            EventInterval(
                start_ts=float(ts),
                end_ts=float(end),
                label=int(lbl),
                location_bus=None,
                location_line=None,
            )
        )

    return intervals


def label_summary(df: pd.DataFrame) -> dict:
    """Return a dict with label → count and label → time ranges."""
    summary: dict = {}
    for label, group in df.groupby("Event"):
        summary[int(label)] = {
            "count": len(group),
            "fraction": len(group) / len(df),
            "time_range": (group["TIMESTAMP"].min(), group["TIMESTAMP"].max()),
        }
    return summary


def compute_dt(df: pd.DataFrame) -> np.ndarray:
    """Compute per-step Δt from TIMESTAMP column.

    Per CLAUDE.md §2.5: do not assume 1/30 exactly.
    Returns array of length len(df)-1.
    """
    ts = df["TIMESTAMP"].to_numpy(dtype=float)
    return np.diff(ts)


def event_transition_frames(
    df: pd.DataFrame,
    *,
    ignore_labels: set[int] | None = None,
    include_label_0: bool = False,
) -> np.ndarray:
    """Return frame indices where the Event label changes.

    This is the canonical way to collect real event onsets for classifier
    training. It keeps direct transitions such as ``5 -> 6 -> 3`` that would be
    merged by the debounced detector into a single long alarm.

    Args:
        df: merged DataFrame with ``Event`` column.
        ignore_labels: labels to exclude from the returned transitions.
        include_label_0: when True, include transitions into label 0 as well.

    Returns:
        Sorted ``(K,)`` integer array of frame indices.
    """
    if "Event" not in df.columns:
        return np.empty(0, dtype=int)

    ignore = set(ignore_labels or set())
    ev = df["Event"].to_numpy(dtype=int)
    frames: list[int] = []
    for i in range(1, len(ev)):
        prev, curr = int(ev[i - 1]), int(ev[i])
        if curr == prev:
            continue
        if not include_label_0 and curr == 0:
            continue
        if curr in ignore:
            continue
        frames.append(i)
    return np.array(frames, dtype=int)
