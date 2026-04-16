from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from src.config.constants import TIMESTAMP_COLUMN
from src.utils.signals import unwrap_if_angle
from src.utils.windows import contiguous_runs


def event_spans(df: pd.DataFrame, label_map: dict[int, str]) -> list[dict]:
    if "Event" not in df.columns:
        return []

    ev = df["Event"].fillna(0).to_numpy(dtype=int)
    t = df[TIMESTAMP_COLUMN].to_numpy(dtype=float)

    spans = []
    for event_id in sorted(set(ev.tolist())):
        if event_id == 0:
            continue

        runs = contiguous_runs(ev == event_id)
        for s, e in runs:
            spans.append({
                "event_id": int(event_id),
                "label": label_map.get(int(event_id), f"Event {event_id}"),
                "start_idx": s,
                "end_idx": e - 1,
                "start_time": float(t[s]),
                "end_time": float(t[e - 1]),
                "duration_s": float(t[e - 1] - t[s]) if e - s > 1 else 0.0,
            })

    spans.sort(key=lambda d: (d["start_time"], d["event_id"]))
    return spans


def summarize_event_spans(spans: list[dict]) -> dict:
    grouped = {}
    by_event = defaultdict(list)
    labels = {}

    for item in spans:
        by_event[int(item["event_id"])].append(float(item["duration_s"]))
        labels[int(item["event_id"])] = str(item["label"])

    for event_id, durations in by_event.items():
        grouped[str(event_id)] = {
            "label": labels[event_id],
            "count": int(len(durations)),
            "durations_s": [float(x) for x in durations],
            "mean_duration_s": float(np.mean(durations)),
            "median_duration_s": float(np.median(durations)),
            "p95_duration_s": float(np.quantile(durations, 0.95)) if len(durations) > 1 else float(durations[0]),
        }

    return grouped


def event_channel_profile(
    df: pd.DataFrame,
    column: str,
    spans: list[dict],
    dt: float,
    baseline_seconds: float,
    post_seconds: float,
) -> dict:
    if not spans or column not in df.columns:
        return {}

    y = unwrap_if_angle(df[column].to_numpy(dtype=float), column)
    t = df[TIMESTAMP_COLUMN].to_numpy(dtype=float)

    baseline_n = max(1, int(round(baseline_seconds / max(dt, 1e-9))))
    post_n = max(1, int(round(post_seconds / max(dt, 1e-9))))

    by_event = defaultdict(list)

    for sp in spans:
        s, e = int(sp["start_idx"]), int(sp["end_idx"]) + 1

        bs = max(0, s - baseline_n)
        pe = min(len(y), e + post_n)

        baseline = y[bs:s]
        during = y[s:e]
        post = y[e:pe]

        if np.isfinite(baseline).sum() == 0 or np.isfinite(during).sum() == 0:
            continue

        base_med = float(np.nanmedian(baseline))
        peak_abs_delta = float(np.nanmax(np.abs(during - base_med)))
        peak_signed_delta = float(during[np.nanargmax(np.abs(during - base_med))] - base_med)
        settle_value = float(np.nanmedian(post)) if np.isfinite(post).sum() else None

        by_event[int(sp["event_id"])].append({
            "start_time": float(t[s]),
            "end_time": float(t[e - 1]),
            "duration_s": float(sp["duration_s"]),
            "baseline_median": base_med,
            "peak_abs_delta": peak_abs_delta,
            "peak_signed_delta": peak_signed_delta,
            "post_median": settle_value,
        })

    summarized = {}
    for event_id, rows in by_event.items():
        peak_abs = [r["peak_abs_delta"] for r in rows]
        peak_signed = [r["peak_signed_delta"] for r in rows]
        durations = [r["duration_s"] for r in rows]

        summarized[str(event_id)] = {
            "count": int(len(rows)),
            "peak_abs_delta_mean": float(np.mean(peak_abs)),
            "peak_abs_delta_median": float(np.median(peak_abs)),
            "peak_signed_delta_mean": float(np.mean(peak_signed)),
            "duration_mean_s": float(np.mean(durations)),
            "instances": rows,
        }

    return summarized