from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models"))

from enevt0_d3tector import _bus_csv_paths, _bus_sort_key  # noqa: E402
from event2_line_outage_detector import (  # noqa: E402
    BUSES,
    PHASES,
    Event2LineOutageDetector,
    Event2LineOutageFeatureExtractor,
    _json_float,
    _metrics,
    _passes_rule,
)


OUT_ROOT = ROOT / "data" / "simulated" / "EVENT0_EVENT2_EVENT0_RANDOM_INSERT"
OUT_PMU = OUT_ROOT / "pmu"
OUT_EVAL = OUT_ROOT / "detector_eval"


def _load_bus_frames(chunk_dir: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for csv_path in _bus_csv_paths(chunk_dir):
        bus = csv_path.name.split("_", 1)[0].upper()
        frames[bus] = pd.read_csv(csv_path)
    if not frames:
        raise FileNotFoundError(f"No PMU CSV files found in {chunk_dir}")
    return frames


def _median_dt(frames: dict[str, pd.DataFrame]) -> float:
    values: list[float] = []
    for frame in frames.values():
        ts = pd.to_numeric(frame.get("TIMESTAMP"), errors="coerce")
        diff = ts.diff().dropna()
        diff = diff[np.isfinite(diff) & (diff > 0)]
        if len(diff):
            values.append(float(diff.median()))
    return float(np.median(values)) if values else 1.0 / 30.0


def _with_time(frame: pd.DataFrame, start: float, dt: float) -> pd.DataFrame:
    output = frame.copy()
    output["TIMESTAMP"] = start + np.arange(len(output), dtype=float) * dt
    return output


def build_sequence(seed: int = 20260501) -> dict[str, Any]:
    rng = random.Random(seed)
    event0_dirs = [
        path
        for path in (ROOT / "data" / "chunked").iterdir()
        if path.is_dir() and path.name.endswith("_event0")
    ]
    pre_dir = rng.choice(sorted(event0_dirs, key=lambda path: path.name))
    post_dir = rng.choice(sorted([path for path in event0_dirs if path != pre_dir], key=lambda path: path.name))
    event_dir = ROOT / "data" / "chunked" / "chunk14_event2"

    pre = _load_bus_frames(pre_dir)
    event = _load_bus_frames(event_dir)
    post = _load_bus_frames(post_dir)
    common_buses = sorted(set(pre) & set(event) & set(post), key=_bus_sort_key)
    dt = _median_dt(event)
    segment_lengths = {
        "event0_pre": min(len(pre[bus]) for bus in common_buses),
        "event2": min(len(event[bus]) for bus in common_buses),
        "event0_post": min(len(post[bus]) for bus in common_buses),
    }

    OUT_PMU.mkdir(parents=True, exist_ok=True)
    OUT_EVAL.mkdir(parents=True, exist_ok=True)
    bounds: dict[str, dict[str, float]] = {}
    sample_labels: list[dict[str, Any]] | None = None

    for bus in common_buses:
        cursor = 0.0
        parts: list[pd.DataFrame] = []
        labels: list[dict[str, Any]] = []
        for phase_name, frames in (("event0_pre", pre), ("event2", event), ("event0_post", post)):
            part = _with_time(frames[bus].iloc[: segment_lengths[phase_name]].reset_index(drop=True), cursor, dt)
            start = float(part["TIMESTAMP"].iloc[0])
            end = float(part["TIMESTAMP"].iloc[-1])
            bounds[phase_name] = {"start": start, "end": end}
            parts.append(part)
            if sample_labels is None:
                labels.extend(
                    {
                        "sample_index": int(len(labels) + idx),
                        "timestamp": float(ts),
                        "true_phase": phase_name,
                        "true_event2": phase_name == "event2",
                    }
                    for idx, ts in enumerate(part["TIMESTAMP"])
                )
            cursor = end + dt
        if sample_labels is None:
            sample_labels = labels
        pd.concat(parts, ignore_index=True).to_csv(OUT_PMU / f"{bus.title()}_Competition_Data_sim.csv", index=False)

    if sample_labels is None:
        raise RuntimeError("No samples written")
    pd.DataFrame(sample_labels).to_csv(OUT_EVAL / "sample_labels.csv", index=False)
    return {
        "seed": seed,
        "pre_event0_chunk": pre_dir.name,
        "event2_chunk": event_dir.name,
        "post_event0_chunk": post_dir.name,
        "dt_seconds": dt,
        "segment_lengths": segment_lengths,
        "bounds": bounds,
    }


def _summarize_features(features: pd.DataFrame, chunk_name: str) -> dict[str, Any]:
    bus_rows: list[dict[str, Any]] = []
    for bus in BUSES:
        v_cols = [f"V{phase}_MAG_PU_FILTERED_{bus}" for phase in PHASES]
        i_cols = [f"I{phase}_MAG_PU_FILTERED_{bus}" for phase in PHASES]
        dv_cols = [f"{col}_D1_PER_S" for col in v_cols]
        di_cols = [f"{col}_D1_PER_S" for col in i_cols]

        v_min = features[v_cols].min(axis=0)
        v_max = features[v_cols].max(axis=0)
        i_min = features[i_cols].min(axis=0)
        i_max = features[i_cols].max(axis=0)
        dv_abs_max = features[dv_cols].abs().max(axis=0)
        di_abs_max = features[di_cols].abs().max(axis=0)
        row = {
            "bus": bus,
            "min_vabc_pu": float(v_min.min()),
            "max_vabc_pu": float(v_max.max()),
            "span_vabc_pu": float(v_max.max() - v_min.min()),
            "max_iabc_pu": float(i_max.max()),
            "min_iabc_pu": float(i_min.min()),
            "span_iabc_pu": float(i_max.max() - i_min.min()),
            "max_abs_dvabc_dt": float(dv_abs_max.max()),
            "max_abs_diabc_dt": float(di_abs_max.max()),
            "mean_abs_diabc_dt": float(di_abs_max.mean()),
            "i_phase_spread_at_max": float(i_max.max() - i_max.min()),
            "v_phase_spread_at_min": float(v_min.max() - v_min.min()),
        }
        for phase, value in zip(PHASES, i_max):
            row[f"i{phase}_max_pu"] = float(value)
        for phase, value in zip(PHASES, di_abs_max):
            row[f"di{phase}_abs_max"] = float(value)
        bus_rows.append(row)

    best = max(bus_rows, key=lambda row: (row["max_iabc_pu"], row["max_abs_diabc_dt"], -abs(row["min_vabc_pu"] - 1.0)))
    return {
        "chunk_name": chunk_name,
        "event_label": None,
        "true_event2": None,
        "best_bus": best["bus"],
        "bus_summaries": bus_rows,
        **{f"best_{key}": value for key, value in best.items() if key != "bus"},
    }


def _merge_positive_windows(rows: list[dict[str, Any]], dt: float) -> list[dict[str, float]]:
    positives = [row for row in rows if row["pred_event2"]]
    if not positives:
        return []
    intervals: list[dict[str, float]] = []
    start = positives[0]["start_time"]
    end = positives[0]["end_time"]
    for row in positives[1:]:
        if row["start_time"] <= end + dt:
            end = max(end, row["end_time"])
        else:
            intervals.append({"start": start, "end": end})
            start = row["start_time"]
            end = row["end_time"]
    intervals.append({"start": start, "end": end})
    return intervals


def evaluate_windows(window_seconds: float = 5.0, step_seconds: float = 0.25, seed: int = 20260501) -> dict[str, Any]:
    sequence = build_sequence(seed=seed)
    frames = _load_bus_frames(OUT_PMU)
    detector = Event2LineOutageDetector()
    extractor = Event2LineOutageFeatureExtractor()
    timestamps = pd.to_numeric(next(iter(frames.values()))["TIMESTAMP"], errors="coerce").reset_index(drop=True)
    dt = sequence["dt_seconds"]
    window_n = max(5, int(round(window_seconds / dt)))
    step_n = max(1, int(round(step_seconds / dt)))
    true_start = sequence["bounds"]["event2"]["start"]
    true_end = sequence["bounds"]["event2"]["end"]
    n_samples = min(len(frame) for frame in frames.values())

    rows: list[dict[str, Any]] = []
    for start_idx in range(0, max(1, n_samples - window_n + 1), step_n):
        end_idx = min(n_samples, start_idx + window_n)
        if end_idx - start_idx < 5:
            continue
        window_frames = {bus: frame.iloc[start_idx:end_idx].reset_index(drop=True) for bus, frame in frames.items()}
        features = extractor.build_detector_features(window_frames)
        summary = _summarize_features(features, f"window_{start_idx}_{end_idx}")
        pred = _passes_rule(summary, detector.params)
        start_time = float(timestamps.iloc[start_idx])
        end_time = float(timestamps.iloc[end_idx - 1])
        rows.append(
            {
                "start_idx": start_idx,
                "end_idx": end_idx - 1,
                "start_time": start_time,
                "end_time": end_time,
                "center_time": 0.5 * (start_time + end_time),
                "true_event2_overlap": bool(start_time <= true_end and end_time >= true_start),
                "pred_event2": bool(pred),
                "pred_label": "event2" if pred else "non2",
                "best_bus": summary["best_bus"],
                "best_min_vabc_pu": _json_float(summary["best_min_vabc_pu"]),
                "best_span_vabc_pu": _json_float(summary["best_span_vabc_pu"]),
                "best_max_iabc_pu": _json_float(summary["best_max_iabc_pu"]),
                "best_span_iabc_pu": _json_float(summary["best_span_iabc_pu"]),
                "best_max_abs_diabc_dt": _json_float(summary["best_max_abs_diabc_dt"]),
            }
        )

    pred_df = pd.DataFrame(rows)
    y = pred_df["true_event2_overlap"].to_numpy(dtype=bool)
    pred = pred_df["pred_event2"].to_numpy(dtype=bool)
    metrics = _metrics(y, pred)
    intervals = _merge_positive_windows(rows, dt)
    pred_df.to_csv(OUT_EVAL / "event2_sliding_window_predictions.csv", index=False)
    first = pred_df[pred_df["pred_event2"]].head(1)
    first_window = None if first.empty else first.iloc[0].to_dict()
    report = {
        "sequence_dir": str(OUT_PMU.resolve()),
        "window_seconds": window_seconds,
        "step_seconds": step_seconds,
        **sequence,
        "true_event2_interval": sequence["bounds"]["event2"],
        "detected_event2_intervals": intervals,
        "first_positive_window": first_window,
        "online_alarm_time_seconds": None if first_window is None else float(first_window["end_time"]),
        "alarm_delay_seconds_from_true_start": None if first_window is None else float(first_window["end_time"] - true_start),
        "window_metrics_overlap_label": metrics,
        "n_windows": int(len(pred_df)),
    }
    (OUT_EVAL / "event2_temporal_detection_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    print(json.dumps(evaluate_windows(), indent=2))


if __name__ == "__main__":
    main()
