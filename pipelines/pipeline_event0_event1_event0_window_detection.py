from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models"))

from enevt0_d3tector import Event0RangeDetector, _bus_csv_paths, _bus_sort_key  # noqa: E402
from event1_fault_detector import BUSES, PHASES, Event1FaultDetector, Event1FaultFeatureExtractor, _json_float, _metrics, _passes_rule  # noqa: E402


OUT_ROOT = ROOT / "data" / "simulated" / "ANDES_PROFILED_EVENT0_EVENT1_EVENT0"
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
        if "TIMESTAMP" not in frame:
            continue
        ts = pd.to_numeric(frame["TIMESTAMP"], errors="coerce")
        diff = ts.diff().dropna()
        diff = diff[np.isfinite(diff) & (diff > 0)]
        if len(diff):
            values.append(float(diff.median()))
    return float(np.median(values)) if values else 1.0 / 30.0


def _with_continuous_time(frame: pd.DataFrame, start_time: float, dt: float) -> pd.DataFrame:
    output = frame.copy()
    output["TIMESTAMP"] = start_time + np.arange(len(output), dtype=float) * dt
    return output


def build_sequence() -> dict[str, Any]:
    pre_dir = ROOT / "data" / "chunked" / "chunk11_event0"
    event_dir = ROOT / "data" / "simulated" / "ANDES_PROFILED_EVENT1_BUS39_RAWMATCH" / "pmu"
    post_dir = ROOT / "data" / "chunked" / "chunk13_event0"

    pre = _load_bus_frames(pre_dir)
    event = _load_bus_frames(event_dir)
    post = _load_bus_frames(post_dir)
    dt = _median_dt(event)
    common_buses = sorted(set(pre) & set(event) & set(post), key=_bus_sort_key)
    segment_lengths = {
        "event0_pre": min(len(pre[bus]) for bus in common_buses),
        "event1": min(len(event[bus]) for bus in common_buses),
        "event0_post": min(len(post[bus]) for bus in common_buses),
    }

    OUT_PMU.mkdir(parents=True, exist_ok=True)
    OUT_EVAL.mkdir(parents=True, exist_ok=True)

    sample_labels: list[dict[str, Any]] | None = None
    sequence_bounds: dict[str, dict[str, float]] = {}

    for bus in common_buses:
        cursor = 0.0
        parts: list[pd.DataFrame] = []
        labels: list[dict[str, Any]] = []
        for phase_name, frames in (("event0_pre", pre), ("event1", event), ("event0_post", post)):
            part = _with_continuous_time(frames[bus].iloc[: segment_lengths[phase_name]].reset_index(drop=True), cursor, dt)
            start = float(part["TIMESTAMP"].iloc[0])
            end = float(part["TIMESTAMP"].iloc[-1])
            sequence_bounds[phase_name] = {"start": start, "end": end}
            parts.append(part)
            if sample_labels is None:
                labels.extend(
                    {
                        "sample_index": int(len(labels) + idx),
                        "timestamp": float(ts),
                        "true_phase": phase_name,
                        "true_event1": phase_name == "event1",
                    }
                    for idx, ts in enumerate(part["TIMESTAMP"])
                )
            cursor = end + dt
        if sample_labels is None:
            sample_labels = labels
        output = pd.concat(parts, ignore_index=True)
        output.to_csv(OUT_PMU / f"{bus.title()}_Competition_Data_sim.csv", index=False)

    if sample_labels is None:
        raise RuntimeError("No sequence labels were created")
    pd.DataFrame(sample_labels).to_csv(OUT_EVAL / "sample_labels.csv", index=False)
    return {"dt": dt, "bounds": sequence_bounds, "n_samples": len(sample_labels), "segment_lengths": segment_lengths}


def _summarize_features(features: pd.DataFrame, chunk_name: str) -> dict[str, Any]:
    bus_rows: list[dict[str, Any]] = []
    for bus in BUSES:
        v_cols = [f"V{phase}_MAG_PU_FILTERED_{bus}" for phase in PHASES]
        i_cols = [f"I{phase}_MAG_PU_FILTERED_{bus}" for phase in PHASES]
        dv_cols = [f"{col}_D1_PER_S" for col in v_cols]
        di_cols = [f"{col}_D1_PER_S" for col in i_cols]

        v_min_by_phase = features[v_cols].min(axis=0)
        i_max_by_phase = features[i_cols].max(axis=0)
        dv_abs_max_by_phase = features[dv_cols].abs().max(axis=0)
        di_abs_max_by_phase = features[di_cols].abs().max(axis=0)
        row = {
            "bus": bus,
            "min_vabc_pu": float(v_min_by_phase.min()),
            "max_iabc_pu": float(i_max_by_phase.max()),
            "max_abs_dvabc_dt": float(dv_abs_max_by_phase.max()),
            "max_abs_diabc_dt": float(di_abs_max_by_phase.max()),
            "mean_vabc_min_pu": float(v_min_by_phase.mean()),
            "mean_iabc_max_pu": float(i_max_by_phase.mean()),
            "mean_abs_dvabc_dt": float(dv_abs_max_by_phase.mean()),
            "mean_abs_diabc_dt": float(di_abs_max_by_phase.mean()),
            "v_phase_spread_at_min": float(v_min_by_phase.max() - v_min_by_phase.min()),
            "i_phase_spread_at_max": float(i_max_by_phase.max() - i_max_by_phase.min()),
        }
        for phase, value in zip(PHASES, v_min_by_phase):
            row[f"v{phase}_min_pu"] = float(value)
        for phase, value in zip(PHASES, i_max_by_phase):
            row[f"i{phase}_max_pu"] = float(value)
        for phase, value in zip(PHASES, dv_abs_max_by_phase):
            row[f"dv{phase}_abs_max"] = float(value)
        for phase, value in zip(PHASES, di_abs_max_by_phase):
            row[f"di{phase}_abs_max"] = float(value)
        bus_rows.append(row)

    best = max(bus_rows, key=lambda row: (1.0 - row["min_vabc_pu"], row["mean_abs_diabc_dt"], row["mean_iabc_max_pu"]))
    return {
        "chunk_name": chunk_name,
        "event_label": None,
        "true_event1": None,
        "best_bus": best["bus"],
        "bus_summaries": bus_rows,
        **{f"best_{key}": value for key, value in best.items() if key != "bus"},
    }


def _merge_positive_windows(rows: list[dict[str, Any]], dt: float) -> list[dict[str, float]]:
    positives = [row for row in rows if row["pred_event1"]]
    if not positives:
        return []
    intervals: list[dict[str, float]] = []
    current_start = positives[0]["start_time"]
    current_end = positives[0]["end_time"]
    for row in positives[1:]:
        if row["start_time"] <= current_end + dt:
            current_end = max(current_end, row["end_time"])
        else:
            intervals.append({"start": current_start, "end": current_end})
            current_start = row["start_time"]
            current_end = row["end_time"]
    intervals.append({"start": current_start, "end": current_end})
    return intervals


def evaluate_windows(window_seconds: float = 5.0, step_seconds: float = 0.25) -> dict[str, Any]:
    sequence = build_sequence()
    frames = _load_bus_frames(OUT_PMU)
    detector = Event1FaultDetector()
    extractor = Event1FaultFeatureExtractor()
    timestamps = pd.to_numeric(next(iter(frames.values()))["TIMESTAMP"], errors="coerce").reset_index(drop=True)
    dt = sequence["dt"]
    window_n = max(5, int(round(window_seconds / dt)))
    step_n = max(1, int(round(step_seconds / dt)))
    true_start = sequence["bounds"]["event1"]["start"]
    true_end = sequence["bounds"]["event1"]["end"]

    rows: list[dict[str, Any]] = []
    n_samples = min(len(frame) for frame in frames.values())
    for start_idx in range(0, max(1, n_samples - window_n + 1), step_n):
        end_idx = min(n_samples, start_idx + window_n)
        if end_idx - start_idx < 5:
            continue
        window_frames = {
            bus: frame.iloc[start_idx:end_idx].reset_index(drop=True)
            for bus, frame in frames.items()
        }
        window_features = extractor.build_detector_features(window_frames)
        summary = _summarize_features(window_features, f"window_{start_idx}_{end_idx}")
        pred = _passes_rule(summary, detector.params)
        start_time = float(timestamps.iloc[start_idx])
        end_time = float(timestamps.iloc[end_idx - 1])
        true_event1 = bool(start_time <= true_end and end_time >= true_start)
        rows.append(
            {
                "start_idx": start_idx,
                "end_idx": end_idx - 1,
                "start_time": start_time,
                "end_time": end_time,
                "center_time": 0.5 * (start_time + end_time),
                "true_event1_overlap": true_event1,
                "pred_event1": bool(pred),
                "pred_label": "event1" if pred else "non1",
                "best_bus": summary["best_bus"],
                "best_min_vabc_pu": _json_float(summary["best_min_vabc_pu"]),
                "best_max_iabc_pu": _json_float(summary["best_max_iabc_pu"]),
                "best_max_abs_diabc_dt": _json_float(summary["best_max_abs_diabc_dt"]),
                "best_max_abs_dvabc_dt": _json_float(summary["best_max_abs_dvabc_dt"]),
            }
        )

    pred_df = pd.DataFrame(rows)
    y = pred_df["true_event1_overlap"].to_numpy(dtype=bool)
    pred = pred_df["pred_event1"].to_numpy(dtype=bool)
    metrics = _metrics(y, pred)
    intervals = _merge_positive_windows(rows, dt)
    pred_df.to_csv(OUT_EVAL / "event1_sliding_window_predictions.csv", index=False)
    report = {
        "sequence_dir": str(OUT_PMU.resolve()),
        "window_seconds": window_seconds,
        "step_seconds": step_seconds,
        "dt_seconds": dt,
        "true_event1_interval": sequence["bounds"]["event1"],
        "detected_event1_intervals": intervals,
        "window_metrics_overlap_label": metrics,
        "n_windows": int(len(pred_df)),
    }
    (OUT_EVAL / "event1_sliding_window_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    report = evaluate_windows()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
