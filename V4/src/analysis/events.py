from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np
import pandas as pd

from src.config.constants import TIMESTAMP_COLUMN
from src.config.models import BusData
from src.utils.signals import unwrap_if_angle
from src.utils.windows import contiguous_runs


def event_spans(df: pd.DataFrame, label_map: dict[int, str]) -> list[dict[str, Any]]:
    """
    Build contiguous event spans from a single DataFrame.

    This is useful for per-bus analysis. It detects contiguous runs of each
    nonzero Event label independently.
    """
    if "Event" not in df.columns or TIMESTAMP_COLUMN not in df.columns or df.empty:
        return []

    ev = pd.to_numeric(df["Event"], errors="coerce").fillna(0).to_numpy(dtype=int)
    t = pd.to_numeric(df[TIMESTAMP_COLUMN], errors="coerce").to_numpy(dtype=float)

    if len(ev) != len(t) or len(ev) == 0:
        return []

    spans: list[dict[str, Any]] = []
    for event_id in sorted(set(ev.tolist())):
        if event_id == 0:
            continue

        runs = contiguous_runs(ev == event_id)
        for s, e in runs:
            if s < 0 or e <= s or e > len(t):
                continue

            spans.append(
                {
                    "event_id": int(event_id),
                    "label": label_map.get(int(event_id), f"Event {event_id}"),
                    "start_idx": int(s),
                    "end_idx": int(e - 1),
                    "start_time": float(t[s]),
                    "end_time": float(t[e - 1]),
                    "duration_s": float(t[e - 1] - t[s]) if e - s > 1 else 0.0,
                }
            )

    spans.sort(key=lambda d: (d["start_time"], d["event_id"]))
    return spans


def build_global_event_spans(
    buses: list[BusData],
    label_map: dict[int, str],
) -> list[dict[str, Any]]:
    """
    Build dataset-level event spans using all buses together instead of only one
    reference bus.

    Strategy:
    - Align on TIMESTAMP by outer merge of per-bus Event columns
    - For each timestamp, derive one global Event label
    - Detect contiguous runs on that global label series

    Label selection rule per timestamp:
    1. Ignore zeros
    2. If only one nonzero label exists, use it
    3. If multiple nonzero labels exist:
       - choose the most frequent nonzero label
       - break ties by smaller label id for determinism
    """
    if not buses:
        return []

    event_frames: list[pd.DataFrame] = []
    for bus in buses:
        df = bus.df
        if TIMESTAMP_COLUMN not in df.columns or "Event" not in df.columns:
            continue

        event_frames.append(
            pd.DataFrame(
                {
                    TIMESTAMP_COLUMN: pd.to_numeric(df[TIMESTAMP_COLUMN], errors="coerce"),
                    f"Event_{bus.bus_id}": pd.to_numeric(df["Event"], errors="coerce").fillna(0).astype(int),
                }
            )
        )

    if not event_frames:
        return []

    merged = event_frames[0].copy()
    for frame in event_frames[1:]:
        merged = merged.merge(frame, on=TIMESTAMP_COLUMN, how="outer")

    merged = merged.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)

    event_cols = [c for c in merged.columns if c.startswith("Event_")]
    if not event_cols:
        return []

    merged[event_cols] = merged[event_cols].fillna(0).astype(int)

    def choose_global_label(row: pd.Series) -> int:
        labels = [int(v) for v in row[event_cols].tolist() if int(v) != 0]
        if not labels:
            return 0
        # Priority: label 6 (missing+physical) overrides others
        if 6 in labels:
            return 6
        # Label 7 (bad data) is bus-specific, keep as separate
        if 7 in labels:
            return 7
        # For other labels, pick most frequent
        counts = Counter(labels)
        best = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        return int(best)

    merged["Global_Event"] = merged.apply(choose_global_label, axis=1)

    global_df = pd.DataFrame(
        {
            TIMESTAMP_COLUMN: merged[TIMESTAMP_COLUMN].to_numpy(dtype=float),
            "Event": merged["Global_Event"].to_numpy(dtype=int),
        }
    )

    spans = event_spans(global_df, label_map)

    for sp in spans:
        s = int(sp["start_idx"])
        e = int(sp["end_idx"])

        involved_buses: set[str] = set()
        per_bus_labels: dict[str, list[int]] = {}

        window = merged.iloc[s : e + 1]
        for col in event_cols:
            bus_id = col.replace("Event_", "", 1)
            labels = sorted({int(v) for v in window[col].tolist() if int(v) != 0})
            if labels:
                involved_buses.add(bus_id)
                per_bus_labels[bus_id] = labels

        sp["involved_buses"] = sorted(involved_buses)
        sp["per_bus_labels"] = per_bus_labels
        sp["bus_count"] = int(len(involved_buses))

    return spans


def summarize_event_spans(spans: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Summarize a list of spans by event_id.
    """
    grouped: dict[str, Any] = {}
    by_event: dict[int, list[float]] = defaultdict(list)
    labels: dict[int, str] = {}

    for item in spans:
        event_id = int(item["event_id"])
        duration = float(item.get("duration_s", 0.0))
        by_event[event_id].append(duration)
        labels[event_id] = str(item.get("label", f"Event {event_id}"))

    for event_id, durations in by_event.items():
        arr = np.asarray(durations, dtype=float)
        grouped[str(event_id)] = {
            "label": labels[event_id],
            "count": int(len(arr)),
            "durations_s": [float(x) for x in arr.tolist()],
            "mean_duration_s": float(np.mean(arr)),
            "median_duration_s": float(np.median(arr)),
            "p95_duration_s": float(np.quantile(arr, 0.95)) if len(arr) > 1 else float(arr[0]),
        }

    return grouped


def event_channel_profile(
    df: pd.DataFrame,
    column: str,
    spans: list[dict[str, Any]],
    dt: float,
    baseline_seconds: float,
    post_seconds: float,
) -> dict[str, Any]:
    """
    Summarize the response of one signal channel around each event span.
    """
    if (
        not spans
        or column not in df.columns
        or TIMESTAMP_COLUMN not in df.columns
        or df.empty
    ):
        return {}

    y = unwrap_if_angle(pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float), column)
    t = pd.to_numeric(df[TIMESTAMP_COLUMN], errors="coerce").to_numpy(dtype=float)

    if len(y) == 0 or len(t) == 0 or len(y) != len(t):
        return {}

    baseline_n = max(1, int(round(baseline_seconds / max(dt, 1e-9))))
    post_n = max(1, int(round(post_seconds / max(dt, 1e-9))))

    by_event: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for sp in spans:
        s = int(sp["start_idx"])
        e = int(sp["end_idx"]) + 1

        if s < 0 or e <= s or e > len(y):
            continue

        bs = max(0, s - baseline_n)
        pe = min(len(y), e + post_n)

        baseline = y[bs:s]
        during = y[s:e]
        post = y[e:pe]

        if baseline.size == 0 or during.size == 0:
            continue
        if np.isfinite(baseline).sum() == 0 or np.isfinite(during).sum() == 0:
            continue

        base_med = float(np.nanmedian(baseline))
        deltas = during - base_med

        if np.isfinite(deltas).sum() == 0:
            continue

        peak_idx = int(np.nanargmax(np.abs(deltas)))
        peak_abs_delta = float(np.nanmax(np.abs(deltas)))
        peak_signed_delta = float(during[peak_idx] - base_med)
        settle_value = float(np.nanmedian(post)) if post.size > 0 and np.isfinite(post).sum() else None

        by_event[int(sp["event_id"])].append(
            {
                "start_time": float(t[s]),
                "end_time": float(t[e - 1]),
                "duration_s": float(sp.get("duration_s", 0.0)),
                "baseline_median": base_med,
                "peak_abs_delta": peak_abs_delta,
                "peak_signed_delta": peak_signed_delta,
                "post_median": settle_value,
            }
        )

    summarized: dict[str, Any] = {}
    for event_id, rows in by_event.items():
        peak_abs = np.asarray([r["peak_abs_delta"] for r in rows], dtype=float)
        peak_signed = np.asarray([r["peak_signed_delta"] for r in rows], dtype=float)
        durations = np.asarray([r["duration_s"] for r in rows], dtype=float)

        summarized[str(event_id)] = {
            "count": int(len(rows)),
            "peak_abs_delta_mean": float(np.mean(peak_abs)),
            "peak_abs_delta_median": float(np.median(peak_abs)),
            "peak_signed_delta_mean": float(np.mean(peak_signed)),
            "duration_mean_s": float(np.mean(durations)),
            "instances": rows,
        }

    return summarized