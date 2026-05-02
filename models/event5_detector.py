from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNK_ROOT = ROOT / "data" / "chunked"


def _bus_sort_key(text: str) -> tuple[int, str]:
    match = re.search(r"\d+", text)
    return (int(match.group(0)) if match else 10**9, text)


def _event_label(chunk_name: str) -> int | None:
    match = re.search(r"_event(\d+)$", chunk_name)
    return int(match.group(1)) if match else None


def _json_float(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _modal_dt_seconds(timestamps: pd.Series, precision: int = 6) -> float | None:
    clean = pd.to_numeric(timestamps, errors="coerce").dropna().sort_values()
    if len(clean) < 2:
        return None
    diffs = clean.diff().dropna()
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return None
    rounded = diffs.round(precision)
    modes = rounded.mode()
    if len(modes) > 0:
        return float(modes.iloc[0])
    return float(diffs.median())


class Event5TimestampGapDetector:
    """Detect event5-style missing data from timestamp gaps.

    This detector follows the requested rule:
    estimate the PMU sampling interval per bus and flag a bus when any
    timestamp delta is greater than one nominal dt.
    """

    def __init__(self, gap_multiplier: float = 1.0, tolerance_fraction: float = 0.05) -> None:
        self.gap_multiplier = float(gap_multiplier)
        self.tolerance_fraction = float(tolerance_fraction)

    def predict_bus_frame(self, frame: pd.DataFrame, bus: str) -> dict[str, Any]:
        if "TIMESTAMP" not in frame.columns:
            raise KeyError("Expected TIMESTAMP column")
        timestamps = pd.to_numeric(frame["TIMESTAMP"], errors="coerce")
        nominal_dt = _modal_dt_seconds(timestamps)
        if nominal_dt is None:
            return {
                "bus": bus,
                "pred_event5": False,
                "reason": "not_enough_timestamps",
                "nominal_dt_seconds": None,
                "max_dt_seconds": None,
                "gap_count": 0,
                "gap_threshold_seconds": None,
            }

        sorted_ts = timestamps.dropna().sort_values()
        diffs = sorted_ts.diff().dropna()
        threshold = nominal_dt * self.gap_multiplier * (1.0 + self.tolerance_fraction)
        gap_mask = diffs > threshold
        gap_diffs = diffs[gap_mask]
        return {
            "bus": bus,
            "pred_event5": bool(len(gap_diffs) > 0),
            "reason": "timestamp_gap" if len(gap_diffs) > 0 else "no_timestamp_gap",
            "nominal_dt_seconds": _json_float(nominal_dt),
            "max_dt_seconds": _json_float(diffs.max() if len(diffs) else None),
            "gap_count": int(len(gap_diffs)),
            "gap_threshold_seconds": _json_float(threshold),
            "gap_dts_seconds": [_json_float(value) for value in gap_diffs.head(20).tolist()],
        }

    def predict_chunk(self, chunk_dir: Path | str) -> dict[str, Any]:
        chunk_path = Path(chunk_dir)
        bus_results: dict[str, dict[str, Any]] = {}
        for csv_path in sorted(chunk_path.glob("Bus*_Competition_Data_nanmask.csv"), key=lambda path: _bus_sort_key(path.name)):
            bus = csv_path.name.split("_", 1)[0].upper()
            frame = pd.read_csv(csv_path, usecols=lambda col: col == "TIMESTAMP")
            bus_results[bus] = self.predict_bus_frame(frame, bus=bus)
        if not bus_results:
            raise FileNotFoundError(f"No Bus*_Competition_Data_nanmask.csv files found in {chunk_path}")

        event5_buses = [bus for bus, result in bus_results.items() if result["pred_event5"]]
        return {
            "chunk_name": chunk_path.name,
            "true_event_label_from_name": _event_label(chunk_path.name),
            "pred_event5": bool(event5_buses),
            "event5_buses": event5_buses,
            "bus_results": bus_results,
        }


class Event5NanMaskDetector(Event5TimestampGapDetector):
    """Fallback detector for this repository's nanmask encoding.

    RAW0001 keeps timestamp rows and encodes missing PMU samples with
    DATA_PRESENT=0 plus NaN measurements. This class detects that encoding.
    """

    def predict_chunk(self, chunk_dir: Path | str) -> dict[str, Any]:
        chunk_path = Path(chunk_dir)
        bus_results: dict[str, dict[str, Any]] = {}
        for csv_path in sorted(chunk_path.glob("Bus*_Competition_Data_nanmask.csv"), key=lambda path: _bus_sort_key(path.name)):
            bus = csv_path.name.split("_", 1)[0].upper()
            frame = pd.read_csv(csv_path)
            base = self.predict_bus_frame(frame[["TIMESTAMP"]], bus=bus)
            data_present_missing = 0
            nan_measurement_rows = 0
            if "DATA_PRESENT" in frame.columns:
                data_present = pd.to_numeric(frame["DATA_PRESENT"], errors="coerce")
                data_present_missing = int((data_present == 0).sum())
            measurement_cols = [
                col
                for col in frame.columns
                if col not in {"TIMESTAMP", "DATA_PRESENT", "Event"}
            ]
            if measurement_cols:
                nan_measurement_rows = int(frame[measurement_cols].isna().any(axis=1).sum())
            pred_event5 = data_present_missing > 0 or nan_measurement_rows > 0
            base.update(
                {
                    "pred_event5": bool(pred_event5),
                    "reason": "nanmask_missing_data" if pred_event5 else base["reason"],
                    "data_present_zero_count": data_present_missing,
                    "nan_measurement_row_count": nan_measurement_rows,
                }
            )
            bus_results[bus] = base

        if not bus_results:
            raise FileNotFoundError(f"No Bus*_Competition_Data_nanmask.csv files found in {chunk_path}")
        event5_buses = [bus for bus, result in bus_results.items() if result["pred_event5"]]
        return {
            "chunk_name": chunk_path.name,
            "true_event_label_from_name": _event_label(chunk_path.name),
            "pred_event5": bool(event5_buses),
            "event5_buses": event5_buses,
            "bus_results": bus_results,
        }


def evaluate_chunk_root(chunk_root: Path, detector: Event5TimestampGapDetector) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for chunk_dir in sorted([path for path in chunk_root.iterdir() if path.is_dir() and "_event" in path.name], key=lambda path: _bus_sort_key(path.name)):
        result = detector.predict_chunk(chunk_dir)
        true_event5 = result["true_event_label_from_name"] == 5
        rows.append(
            {
                "chunk_name": result["chunk_name"],
                "true_event_label": result["true_event_label_from_name"],
                "true_event5": true_event5,
                "pred_event5": result["pred_event5"],
                "event5_buses": "|".join(result["event5_buses"]),
            }
        )
    predictions = pd.DataFrame(rows)
    y = predictions["true_event5"].to_numpy(dtype=bool)
    p = predictions["pred_event5"].to_numpy(dtype=bool)
    tp = int(np.sum(y & p))
    fn = int(np.sum(y & ~p))
    fp = int(np.sum(~y & p))
    tn = int(np.sum(~y & ~p))
    metrics = {
        "tp_event5_as_event5": tp,
        "fn_event5_as_non5": fn,
        "fp_non5_as_event5": fp,
        "tn_non5_as_non5": tn,
        "accuracy": _json_float((tp + tn) / len(y) if len(y) else 0.0),
        "precision_event5": _json_float(tp / (tp + fp) if tp + fp else 0.0),
        "recall_event5": _json_float(tp / (tp + fn) if tp + fn else 0.0),
        "specificity_non5": _json_float(tn / (tn + fp) if tn + fp else 0.0),
    }
    return predictions, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect event5 from PMU timestamp gaps.")
    parser.add_argument("--chunk-dir", type=Path, default=None)
    parser.add_argument("--chunk-root", type=Path, default=DEFAULT_CHUNK_ROOT)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--mode", choices=["timestamp_gap", "nanmask"], default="timestamp_gap")
    parser.add_argument("--gap-multiplier", type=float, default=1.0)
    parser.add_argument("--tolerance-fraction", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector_cls = Event5NanMaskDetector if args.mode == "nanmask" else Event5TimestampGapDetector
    detector = detector_cls(gap_multiplier=args.gap_multiplier, tolerance_fraction=args.tolerance_fraction)
    if args.chunk_dir is not None:
        print(json.dumps(detector.predict_chunk(args.chunk_dir), indent=2))
        return

    predictions, metrics = evaluate_chunk_root(args.chunk_root, detector)
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(args.output_csv, index=False)
        print(f"Wrote {args.output_csv.resolve()}")
    print(predictions.to_string(index=False))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
