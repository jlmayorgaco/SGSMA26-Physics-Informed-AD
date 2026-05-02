from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models"))

from event1_fault_detector import Event1FaultDetector  # noqa: E402
from event2_line_outage_detector import Event2LineOutageDetector  # noqa: E402
from event3_generation_change_detector import Event3GenerationChangeDetector  # noqa: E402
from event4_load_change_detector import Event4LoadChangeDetector  # noqa: E402
from event5_detector import DEFAULT_CHUNK_ROOT, Event5NanMaskDetector  # noqa: E402


DEFAULT_CONFIG = ROOT / "models" / "event6_concurrent_missing_physical_config.json"
DEFAULT_REPORT = ROOT / "models" / "event6_concurrent_missing_physical_report.json"
DEFAULT_PREDICTIONS = ROOT / "models" / "event6_concurrent_missing_physical_predictions.csv"


def _event_label(chunk_name: str) -> int | None:
    if "_event" not in chunk_name:
        return None
    try:
        return int(chunk_name.rsplit("_event", 1)[1])
    except ValueError:
        return None


def _json_float(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def _metrics(y: np.ndarray, pred: np.ndarray, prefix: str = "event6") -> dict[str, Any]:
    tp = int(np.sum(y & pred))
    fn = int(np.sum(y & ~pred))
    fp = int(np.sum(~y & pred))
    tn = int(np.sum(~y & ~pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        f"tp_{prefix}_as_{prefix}": tp,
        f"fn_{prefix}_as_non{prefix[-1] if prefix.startswith('event') else ''}": fn,
        f"fp_non{prefix[-1] if prefix.startswith('event') else ''}_as_{prefix}": fp,
        f"tn_non{prefix[-1] if prefix.startswith('event') else ''}_as_non{prefix[-1] if prefix.startswith('event') else ''}": tn,
        "accuracy": _json_float((tp + tn) / len(y) if len(y) else 0.0),
        f"precision_{prefix}": _json_float(precision),
        f"recall_{prefix}": _json_float(recall),
        f"specificity_non{prefix[-1] if prefix.startswith('event') else ''}": _json_float(specificity),
        f"f1_{prefix}": _json_float(f1),
    }


class Event6ConcurrentMissingPhysicalDetector:
    """Detect label-6 style concurrent missing data plus a physical event.

    Per the guideline, event5 is missing-data only. Event6 is missing data at
    a PMU concurrent with a physical event elsewhere. This detector therefore
    composes the missing-data detector with physical event detectors 1-4.
    """

    def __init__(self) -> None:
        self.missing_detector = Event5NanMaskDetector()
        self.physical_detectors = {
            "event1": Event1FaultDetector(),
            "event2": Event2LineOutageDetector(),
            "event3": Event3GenerationChangeDetector(),
            "event4": Event4LoadChangeDetector(),
        }

    def predict_chunk(self, chunk_dir: Path | str) -> dict[str, Any]:
        chunk_path = Path(chunk_dir)
        missing = self.missing_detector.predict_chunk(chunk_path)
        physical: dict[str, dict[str, Any]] = {}
        for name, detector in self.physical_detectors.items():
            physical[name] = detector.predict_chunk(chunk_path)
        physical_events = [
            name
            for name, result in physical.items()
            if bool(result.get(f"pred_{name}", False))
        ]
        pred_missing = bool(missing["pred_event5"])
        pred_event6 = pred_missing and bool(physical_events)
        return {
            "chunk_name": chunk_path.name,
            "true_event_label_from_name": _event_label(chunk_path.name),
            "pred_event6": bool(pred_event6),
            "pred_label": "event6" if pred_event6 else "non6",
            "pred_missing_data": pred_missing,
            "missing_buses": missing["event5_buses"],
            "physical_events": physical_events,
            "physical_predictions": physical,
        }


def evaluate_chunk_root(chunk_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    detector = Event6ConcurrentMissingPhysicalDetector()
    rows: list[dict[str, Any]] = []
    for chunk_dir in sorted([path for path in chunk_root.iterdir() if path.is_dir() and "_event" in path.name], key=lambda path: int(path.name.split("_", 1)[0].replace("chunk", ""))):
        result = detector.predict_chunk(chunk_dir)
        event_label = result["true_event_label_from_name"]
        guideline_proxy_event6 = bool(result["pred_missing_data"] and event_label in {1, 2, 3, 4})
        rows.append(
            {
                "chunk_name": result["chunk_name"],
                "event_label": event_label,
                "true_event6_from_label": event_label == 6,
                "guideline_proxy_event6_missing_plus_physical_label": guideline_proxy_event6,
                "pred_event6": result["pred_event6"],
                "pred_missing_data": result["pred_missing_data"],
                "missing_buses": "|".join(result["missing_buses"]),
                "physical_events": "|".join(result["physical_events"]),
            }
        )
    predictions = pd.DataFrame(rows)
    y_proxy = predictions["guideline_proxy_event6_missing_plus_physical_label"].to_numpy(dtype=bool)
    pred = predictions["pred_event6"].to_numpy(dtype=bool)
    report = {
        "note": "RAW0001 has no chunks named event6. The proxy target follows the guideline: missing data plus a physical event label 1-4.",
        "proxy_event6_metrics": _metrics(y_proxy, pred, prefix="event6"),
        "predicted_event6_chunks": predictions[pred]["chunk_name"].tolist(),
        "proxy_event6_chunks": predictions[y_proxy]["chunk_name"].tolist(),
    }
    return predictions, report


def write_default_config(report: dict[str, Any]) -> None:
    config = {
        "schema_version": 1,
        "model_name": "event6_concurrent_missing_physical_rule",
        "model_type": "composite_rule",
        "rule": {
            "missing_data": "DATA_PRESENT == 0 or any measurement NaN in any PMU stream",
            "physical_event": "any event1/event2/event3/event4 detector is positive",
            "event6": "missing_data AND physical_event",
            "event5_only": "missing_data AND NOT physical_event",
        },
        "validation": report,
    }
    DEFAULT_CONFIG.write_text(json.dumps(config, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect event6: missing data concurrent with physical event.")
    parser.add_argument("--chunk-dir", type=Path, default=None)
    parser.add_argument("--chunk-root", type=Path, default=DEFAULT_CHUNK_ROOT)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = Event6ConcurrentMissingPhysicalDetector()
    if args.chunk_dir is not None:
        print(json.dumps(detector.predict_chunk(args.chunk_dir), indent=2))
        return

    predictions, report = evaluate_chunk_root(args.chunk_root)
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.predictions, index=False)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_default_config(report)
    print(f"Wrote {args.predictions.resolve()}")
    print(f"Wrote {args.report.resolve()}")
    print(f"Wrote {DEFAULT_CONFIG.resolve()}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
