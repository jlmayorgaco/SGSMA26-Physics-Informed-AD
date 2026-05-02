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
from event3_generation_change_detector import (  # noqa: E402
    BUSES,
    PHASES,
    Event3GenerationChangeDetector,
    _metrics as event3_metrics,
    _nan_stat as event3_nan_stat,
    _passes_rule as event3_passes_rule,
)
from event4_load_change_detector import (  # noqa: E402
    Event4LoadChangeDetector,
    _metrics as event4_metrics,
    _nan_stat as event4_nan_stat,
    _passes_rule as event4_passes_rule,
)


OUT_ROOT = ROOT / "data" / "simulated" / "EVENT0_EVENT3_EVENT0_EVENT4_EVENT0_PROFILED"
OUT_PMU = OUT_ROOT / "pmu"
OUT_EVAL = OUT_ROOT / "detector_eval"


SEGMENTS = (
    ("event0_pre", ROOT / "data" / "chunked" / "chunk9_event0"),
    ("event3", ROOT / "data" / "chunked" / "chunk19_event3"),
    ("event0_mid", ROOT / "data" / "chunked" / "chunk17_event0"),
    ("event4", ROOT / "data" / "chunked" / "chunk22_event4"),
    ("event0_post", ROOT / "data" / "chunked" / "chunk15_event0"),
)


def _json_float(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


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


def build_sequence() -> dict[str, Any]:
    loaded = [(name, path, _load_bus_frames(path)) for name, path in SEGMENTS]
    common_buses = sorted(set.intersection(*(set(frames) for _, _, frames in loaded)), key=_bus_sort_key)
    dt = _median_dt(next(frames for name, _, frames in loaded if name == "event3"))
    lengths = {
        name: min(len(frames[bus]) for bus in common_buses)
        for name, _, frames in loaded
    }

    OUT_PMU.mkdir(parents=True, exist_ok=True)
    OUT_EVAL.mkdir(parents=True, exist_ok=True)

    bounds: dict[str, dict[str, float]] = {}
    sample_labels: list[dict[str, Any]] | None = None
    for bus in common_buses:
        cursor = 0.0
        parts: list[pd.DataFrame] = []
        labels: list[dict[str, Any]] = []
        for name, _, frames in loaded:
            part = _with_time(frames[bus].iloc[: lengths[name]].reset_index(drop=True), cursor, dt)
            start = float(part["TIMESTAMP"].iloc[0])
            end = float(part["TIMESTAMP"].iloc[-1])
            bounds[name] = {"start": start, "end": end}
            parts.append(part)
            if sample_labels is None:
                labels.extend(
                    {
                        "sample_index": int(len(labels) + idx),
                        "timestamp": float(ts),
                        "true_phase": name,
                        "true_event3": name == "event3",
                        "true_event4": name == "event4",
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
        "segments": [{"name": name, "source": str(path.resolve()), "n_samples": lengths[name]} for name, path, _ in loaded],
        "dt_seconds": dt,
        "bounds": bounds,
    }


def summarize_event3(features: pd.DataFrame, chunk_name: str) -> dict[str, Any]:
    bus_rows = []
    for bus in BUSES:
        v_cols = [f"V{p}_MAG_PU_FILTERED_{bus}" for p in PHASES]
        i_cols = [f"I{p}_MAG_PU_FILTERED_{bus}" for p in PHASES]
        di_cols = [f"{col}_D1_PER_S" for col in i_cols]
        freq_col = f"Freq_HZ_DEV_FILTERED_{bus}"
        rocof_col = f"ROCOF_HZ_PER_S_FILTERED_{bus}"
        row = {
            "bus": bus,
            "min_vabc_pu": event3_nan_stat(features[v_cols].to_numpy(dtype=float), "min"),
            "max_vabc_pu": event3_nan_stat(features[v_cols].to_numpy(dtype=float), "max"),
            "max_iabc_pu": event3_nan_stat(features[i_cols].to_numpy(dtype=float), "max"),
            "span_iabc_pu": event3_nan_stat(features[i_cols].to_numpy(dtype=float), "max") - event3_nan_stat(features[i_cols].to_numpy(dtype=float), "min"),
            "max_abs_diabc_dt": event3_nan_stat(np.abs(features[di_cols].to_numpy(dtype=float)), "max"),
            "freq_span_hz": event3_nan_stat(features[freq_col].to_numpy(dtype=float), "max") - event3_nan_stat(features[freq_col].to_numpy(dtype=float), "min"),
            "freq_abs_step_hz": abs(
                event3_nan_stat(features[freq_col].tail(max(1, len(features) // 5)).to_numpy(dtype=float), "median")
                - event3_nan_stat(features[freq_col].head(max(1, len(features) // 5)).to_numpy(dtype=float), "median")
            ),
            "rocof_abs_max": event3_nan_stat(np.abs(features[rocof_col].to_numpy(dtype=float)), "max"),
        }
        bus_rows.append(row)
    best = max(bus_rows, key=lambda row: (row["freq_span_hz"], row["rocof_abs_max"], row["max_abs_diabc_dt"]))
    return {
        "chunk_name": chunk_name,
        "event_label": None,
        "true_event3": None,
        "duration_seconds_est": float(len(features) / 30.0),
        "n_rows": int(len(features)),
        "best_bus": best["bus"],
        "bus_summaries": bus_rows,
        "max_freq_span_hz": float(max(row["freq_span_hz"] for row in bus_rows)),
        "max_freq_abs_step_hz": float(max(row["freq_abs_step_hz"] for row in bus_rows)),
        "max_rocof_abs": float(max(row["rocof_abs_max"] for row in bus_rows)),
        "max_abs_diabc_dt": float(max(row["max_abs_diabc_dt"] for row in bus_rows)),
        "max_iabc_pu": float(max(row["max_iabc_pu"] for row in bus_rows)),
        "min_vabc_pu": float(min(row["min_vabc_pu"] for row in bus_rows)),
        "bus2_freq_span_hz": next(row["freq_span_hz"] for row in bus_rows if row["bus"] == "BUS2"),
        "bus2_rocof_abs": next(row["rocof_abs_max"] for row in bus_rows if row["bus"] == "BUS2"),
        "bus2_max_abs_diabc_dt": next(row["max_abs_diabc_dt"] for row in bus_rows if row["bus"] == "BUS2"),
    }


def summarize_event4(features: pd.DataFrame, chunk_name: str) -> dict[str, Any]:
    bus_rows = []
    for bus in BUSES:
        v_cols = [f"V{p}_MAG_PU_FILTERED_{bus}" for p in PHASES]
        i_cols = [f"I{p}_MAG_PU_FILTERED_{bus}" for p in PHASES]
        di_cols = [f"{col}_D1_PER_S" for col in i_cols]
        freq_col = f"Freq_HZ_DEV_FILTERED_{bus}"
        rocof_col = f"ROCOF_HZ_PER_S_FILTERED_{bus}"
        vals_v = features[v_cols].to_numpy(dtype=float)
        vals_i = features[i_cols].to_numpy(dtype=float)
        bus_rows.append(
            {
                "bus": bus,
                "min_vabc_pu": event4_nan_stat(vals_v, "min"),
                "max_vabc_pu": event4_nan_stat(vals_v, "max"),
                "span_vabc_pu": event4_nan_stat(vals_v, "max") - event4_nan_stat(vals_v, "min"),
                "max_iabc_pu": event4_nan_stat(vals_i, "max"),
                "span_iabc_pu": event4_nan_stat(vals_i, "max") - event4_nan_stat(vals_i, "min"),
                "max_abs_diabc_dt": event4_nan_stat(np.abs(features[di_cols].to_numpy(dtype=float)), "max"),
                "freq_span_hz": event4_nan_stat(features[freq_col].to_numpy(dtype=float), "max") - event4_nan_stat(features[freq_col].to_numpy(dtype=float), "min"),
                "rocof_abs_max": event4_nan_stat(np.abs(features[rocof_col].to_numpy(dtype=float)), "max"),
            }
        )
    best = max(bus_rows, key=lambda row: (row["span_iabc_pu"], row["max_abs_diabc_dt"]))
    return {
        "chunk_name": chunk_name,
        "event_label": None,
        "true_event4": None,
        "duration_seconds_est": float(len(features) / 30.0),
        "n_rows": int(len(features)),
        "best_bus": best["bus"],
        "bus_summaries": bus_rows,
        "max_current_span_pu": float(max(row["span_iabc_pu"] for row in bus_rows)),
        "max_iabc_pu": float(max(row["max_iabc_pu"] for row in bus_rows)),
        "max_abs_diabc_dt": float(max(row["max_abs_diabc_dt"] for row in bus_rows)),
        "max_freq_span_hz": float(max(row["freq_span_hz"] for row in bus_rows)),
        "max_rocof_abs": float(max(row["rocof_abs_max"] for row in bus_rows)),
        "min_vabc_pu": float(min(row["min_vabc_pu"] for row in bus_rows)),
        "max_voltage_span_pu": float(max(row["span_vabc_pu"] for row in bus_rows)),
    }


def _merge(rows: list[dict[str, Any]], pred_key: str, dt: float) -> list[dict[str, float]]:
    positives = [row for row in rows if row[pred_key]]
    if not positives:
        return []
    intervals = []
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


def evaluate_windows(window_seconds: float = 30.0, step_seconds: float = 1.0) -> dict[str, Any]:
    sequence = build_sequence()
    frames = _load_bus_frames(OUT_PMU)
    e0 = Event0RangeDetector()
    e3 = Event3GenerationChangeDetector()
    e4 = Event4LoadChangeDetector()
    timestamps = pd.to_numeric(next(iter(frames.values()))["TIMESTAMP"], errors="coerce").reset_index(drop=True)
    dt = sequence["dt_seconds"]
    window_n = max(5, int(round(window_seconds / dt)))
    step_n = max(1, int(round(step_seconds / dt)))
    n_samples = min(len(frame) for frame in frames.values())
    e3_start, e3_end = sequence["bounds"]["event3"]["start"], sequence["bounds"]["event3"]["end"]
    e4_start, e4_end = sequence["bounds"]["event4"]["start"], sequence["bounds"]["event4"]["end"]

    rows: list[dict[str, Any]] = []
    for start_idx in range(0, max(1, n_samples - window_n + 1), step_n):
        end_idx = min(n_samples, start_idx + window_n)
        window_frames = {bus: frame.iloc[start_idx:end_idx].reset_index(drop=True) for bus, frame in frames.items()}
        features = e0.build_features(window_frames, include_derivatives=True)
        s3 = summarize_event3(features, f"window_{start_idx}_{end_idx}")
        s4 = summarize_event4(features, f"window_{start_idx}_{end_idx}")
        pred3 = event3_passes_rule(s3, e3.params)
        pred4 = event4_passes_rule(s4, e4.params)
        start_time = float(timestamps.iloc[start_idx])
        end_time = float(timestamps.iloc[end_idx - 1])
        rows.append(
            {
                "start_idx": start_idx,
                "end_idx": end_idx - 1,
                "start_time": start_time,
                "end_time": end_time,
                "center_time": 0.5 * (start_time + end_time),
                "true_event3_overlap": bool(start_time <= e3_end and end_time >= e3_start),
                "true_event4_overlap": bool(start_time <= e4_end and end_time >= e4_start),
                "pred_event3": bool(pred3),
                "pred_event4": bool(pred4),
                "event3_best_bus": s3["best_bus"],
                "event3_max_freq_span_hz": _json_float(s3["max_freq_span_hz"]),
                "event3_max_rocof_abs": _json_float(s3["max_rocof_abs"]),
                "event3_bus2_freq_span_hz": _json_float(s3["bus2_freq_span_hz"]),
                "event4_best_bus": s4["best_bus"],
                "event4_max_current_span_pu": _json_float(s4["max_current_span_pu"]),
                "event4_max_freq_span_hz": _json_float(s4["max_freq_span_hz"]),
                "event4_max_rocof_abs": _json_float(s4["max_rocof_abs"]),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_EVAL / "event3_event4_sliding_window_predictions.csv", index=False)
    m3 = event3_metrics(df["true_event3_overlap"].to_numpy(dtype=bool), df["pred_event3"].to_numpy(dtype=bool))
    m4 = event4_metrics(df["true_event4_overlap"].to_numpy(dtype=bool), df["pred_event4"].to_numpy(dtype=bool))
    first3 = df[df["pred_event3"]].head(1)
    first4 = df[df["pred_event4"]].head(1)
    report = {
        "sequence_dir": str(OUT_PMU.resolve()),
        "window_seconds": window_seconds,
        "step_seconds": step_seconds,
        **sequence,
        "true_event3_interval": sequence["bounds"]["event3"],
        "true_event4_interval": sequence["bounds"]["event4"],
        "detected_event3_intervals": _merge(rows, "pred_event3", dt),
        "detected_event4_intervals": _merge(rows, "pred_event4", dt),
        "event3_first_positive_window": None if first3.empty else first3.iloc[0].to_dict(),
        "event4_first_positive_window": None if first4.empty else first4.iloc[0].to_dict(),
        "event3_alarm_delay_seconds_from_true_start": None if first3.empty else float(first3.iloc[0]["end_time"] - e3_start),
        "event4_alarm_delay_seconds_from_true_start": None if first4.empty else float(first4.iloc[0]["end_time"] - e4_start),
        "event3_window_metrics_overlap_label": m3,
        "event4_window_metrics_overlap_label": m4,
        "n_windows": int(len(df)),
    }
    (OUT_EVAL / "event3_event4_temporal_detection_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    print(json.dumps(evaluate_windows(), indent=2))


if __name__ == "__main__":
    main()
