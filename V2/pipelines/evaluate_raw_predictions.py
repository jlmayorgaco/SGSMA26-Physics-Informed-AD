"""Evaluate V2 raw prediction JSON against visible raw Event columns."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pmu_features import PMU_BUSES, find_bus_csv  # noqa: E402
from src.ml.pmu_grid_pipeline import LABEL_PRIORITY  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--sample-stride", type=int, default=1, help="Evaluate every Nth inference window.")
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def centers_from_prediction(prediction: dict[str, Any], raw_dir: Path) -> np.ndarray:
    window_sec = float(prediction.get("window_sec", 1.0))
    stride_sec = float(prediction.get("stride_sec", 0.25))
    n_windows = int(prediction.get("n_windows", 0))
    first_csv = find_bus_csv(raw_dir, int(prediction.get("pmu_buses", PMU_BUSES)[0]))
    timestamps = pd.read_csv(first_csv, usecols=["TIMESTAMP"])["TIMESTAMP"].to_numpy(dtype=float)
    first = float(timestamps[0]) + window_sec / 2.0
    return first + np.arange(n_windows, dtype=float) * stride_sec


def prediction_labels(prediction: dict[str, Any], centers: np.ndarray) -> np.ndarray:
    labels = np.zeros(len(centers), dtype=int)
    for event in prediction.get("events", []):
        label = int(event.get("label", 0))
        start = float(event.get("start_sec", 0.0))
        end = float(event.get("end_sec", start))
        mask = (centers >= start) & (centers <= end)
        current = labels[mask]
        if current.size == 0:
            continue
        current_priority = np.array([LABEL_PRIORITY.get(int(value), int(value)) for value in current])
        take = current_priority < LABEL_PRIORITY.get(label, label)
        updated = current.copy()
        updated[take] = label
        labels[mask] = updated
    return labels


class RawTruth:
    """Fast visible-label lookup for raw PMU Event columns."""

    def __init__(self, raw_dir: Path, pmu_buses: list[int]) -> None:
        self.series: list[tuple[np.ndarray, np.ndarray]] = []
        for bus in pmu_buses:
            frame = pd.read_csv(find_bus_csv(raw_dir, int(bus)), usecols=["TIMESTAMP", "Event"])
            timestamps = pd.to_numeric(frame["TIMESTAMP"], errors="coerce").to_numpy(dtype=float)
            events = pd.to_numeric(frame["Event"], errors="coerce").fillna(0).astype(int).to_numpy()
            self.series.append((timestamps, events))

    def labels(self, centers: np.ndarray, window_sec: float) -> np.ndarray:
        half = float(window_sec) / 2.0
        labels = np.zeros(len(centers), dtype=int)
        for index, center in enumerate(centers):
            bus_labels: list[int] = []
            for timestamps, events in self.series:
                left = int(np.searchsorted(timestamps, center - half, side="left"))
                right = int(np.searchsorted(timestamps, center + half, side="right"))
                if right <= left:
                    nearest = int(np.nanargmin(np.abs(timestamps - center)))
                    values = [int(events[nearest])]
                else:
                    values = [int(value) for value in events[left:right]]
                bus_labels.append(max(values, key=lambda label: LABEL_PRIORITY.get(label, label)))
            labels[index] = max(bus_labels, key=lambda label: LABEL_PRIORITY.get(label, label))
        return labels


def evaluate(prediction_path: Path, raw_dir: Path, sample_stride: int) -> dict[str, Any]:
    prediction = load_json(prediction_path)
    centers = centers_from_prediction(prediction, raw_dir)
    if sample_stride > 1:
        centers = centers[::sample_stride]
    pred = prediction_labels(prediction, centers)
    truth = RawTruth(raw_dir, [int(bus) for bus in prediction.get("pmu_buses", PMU_BUSES)])
    true = truth.labels(centers, float(prediction.get("window_sec", 1.0)))
    labels = sorted(set(int(value) for value in true) | set(int(value) for value in pred))
    report = classification_report(true, pred, labels=labels, output_dict=True, zero_division=0)
    matrix = confusion_matrix(true, pred, labels=labels)
    result = {
        "prediction_path": str(prediction_path.resolve()),
        "raw_dir": str(raw_dir.resolve()),
        "n_windows_evaluated": int(len(centers)),
        "sample_stride": int(sample_stride),
        "labels": labels,
        "true_counts": {str(k): int(v) for k, v in sorted(Counter(true).items())},
        "pred_counts": {str(k): int(v) for k, v in sorted(Counter(pred).items())},
        "accuracy": float(accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(true, pred, labels=labels, average="weighted", zero_division=0)),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }
    positive = true != 0
    if np.any(positive):
        result["positive_accuracy"] = float(accuracy_score(true[positive], pred[positive]))
        result["positive_macro_f1"] = float(
            f1_score(
                true[positive],
                pred[positive],
                labels=[label for label in labels if label != 0],
                average="macro",
                zero_division=0,
            )
        )
    return result


def main() -> None:
    args = parse_args()
    prediction_path = project_path(args.predictions)
    raw_dir = project_path(args.raw_dir)
    out = project_path(args.out) if args.out else prediction_path.with_name(prediction_path.stem + "_raw_eval.json")
    result = evaluate(prediction_path, raw_dir, max(1, int(args.sample_stride)))
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
