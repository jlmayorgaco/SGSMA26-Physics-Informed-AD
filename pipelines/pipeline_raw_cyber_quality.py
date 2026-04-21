from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.classes.report_models import RawCyberQualityReport
from src.estimators import Event5Estimator, Event7Estimator
from src.helpers.metrics import compute_binary_metrics
from src.helpers.raw_loader import load_and_align_raw_directory
from src.utils.plot_utils import plot_active_bus_count, plot_event7_score_histogram, plot_event_timeline


def _build_true_labels(aligned_frames: dict[str, pd.DataFrame]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bus_names = sorted(aligned_frames.keys())
    event_table = pd.DataFrame(
        {
            bus: pd.to_numeric(frame["Event"], errors="coerce").astype(float)
            for bus, frame in aligned_frames.items()
        }
    )
    observed_table = pd.DataFrame({bus: event_table[bus].notna().astype(int) for bus in bus_names})

    true_event5 = (
        (~observed_table.astype(bool)).any(axis=1)
        | event_table.isin([5.0, 6.0]).any(axis=1)
    ).astype(int)
    true_event7 = event_table.eq(7.0).any(axis=1).astype(int)
    true_cyber = ((true_event5 == 1) | (true_event7 == 1)).astype(int)

    true_event7_bus = np.full((len(event_table),), "", dtype=object)
    if int(true_event7.sum()) > 0:
        event7_rows = np.where(true_event7.to_numpy(dtype=int) == 1)[0]
        for row_idx in event7_rows:
            row = event_table.iloc[row_idx]
            buses = [bus for bus in bus_names if row.get(bus) == 7.0]
            true_event7_bus[row_idx] = buses[0] if buses else ""

    return (
        true_event5.to_numpy(dtype=int),
        true_event7.to_numpy(dtype=int),
        true_cyber.to_numpy(dtype=int),
        true_event7_bus,
    )


def _count_dict(**kwargs: int) -> dict[str, int]:
    return {key: int(value) for key, value in kwargs.items()}


def run_pipeline(raw_dir: Path, output_dir: Path, max_plot_points: int = 8000) -> dict[str, Any]:
    aligned_data = load_and_align_raw_directory(raw_dir=raw_dir)
    timeline = aligned_data.timeline
    aligned_frames = aligned_data.aligned_frames

    event5 = Event5Estimator()
    event5_result = event5.estimate(aligned_frames=aligned_frames)

    event7 = Event7Estimator()
    event7_result = event7.estimate(
        aligned_frames=aligned_frames,
        event5_prediction=event5_result.frame_prediction.to_numpy(dtype=int),
    )

    true_event5, true_event7, true_cyber, true_event7_bus = _build_true_labels(aligned_frames=aligned_frames)

    pred_event5 = event5_result.frame_prediction.to_numpy(dtype=int)
    pred_event7 = event7_result.frame_prediction.to_numpy(dtype=int)
    pred_cyber = ((pred_event5 == 1) | (pred_event7 == 1)).astype(int)

    event5_metrics = compute_binary_metrics(true_values=true_event5, pred_values=pred_event5)
    event7_metrics = compute_binary_metrics(true_values=true_event7, pred_values=pred_event7)
    cyber_metrics = compute_binary_metrics(true_values=true_cyber, pred_values=pred_cyber)

    pred_event7_bus = event7_result.frame_top_bus.to_numpy(dtype=object)
    true7_idx = np.where(true_event7 == 1)[0]
    if true7_idx.size > 0:
        event7_bus_acc = float(np.mean(pred_event7_bus[true7_idx] == true_event7_bus[true7_idx]))
    else:
        event7_bus_acc = 0.0

    output_dir.mkdir(parents=True, exist_ok=True)
    timeline_plot = output_dir / "timeline_event5_event7.png"
    score_plot = output_dir / "event7_score_histogram.png"
    active_plot = output_dir / "event7_active_bus_count.png"
    plot_event_timeline(
        timeline=timeline,
        true_event5=true_event5,
        pred_event5=pred_event5,
        true_event7=true_event7,
        pred_event7=pred_event7,
        output_path=timeline_plot,
        max_points=max_plot_points,
    )
    plot_event7_score_histogram(
        event7_score=event7_result.frame_max_score.to_numpy(dtype=float),
        true_event7=true_event7,
        output_path=score_plot,
    )
    plot_active_bus_count(
        active_bus_count=event7_result.frame_active_bus_count.to_numpy(dtype=int),
        true_event7=true_event7,
        output_path=active_plot,
    )

    predictions_path = output_dir / "frame_predictions.csv"
    per_frame = pd.DataFrame(
        {
            "TIMESTAMP": timeline.to_numpy(dtype=float),
            "true_event5": true_event5,
            "pred_event5": pred_event5,
            "true_event7": true_event7,
            "pred_event7": pred_event7,
            "true_cyber": true_cyber,
            "pred_cyber": pred_cyber,
            "true_event7_bus": true_event7_bus,
            "pred_event7_bus": pred_event7_bus,
            "event5_score": event5_result.frame_score.to_numpy(dtype=float),
            "event7_score": event7_result.frame_max_score.to_numpy(dtype=float),
            "event7_active_bus_count": event7_result.frame_active_bus_count.to_numpy(dtype=int),
        }
    )
    per_frame.to_csv(predictions_path, index=False)

    report = RawCyberQualityReport(
        input_raw_dir=str(raw_dir.resolve()),
        output_dir=str(output_dir.resolve()),
        buses=sorted(aligned_frames.keys()),
        n_frames=int(len(timeline)),
        event5_metrics=event5_metrics,
        event7_metrics=event7_metrics,
        cyber_metrics=cyber_metrics,
        event7_bus_localization_accuracy=event7_bus_acc,
        counts=_count_dict(
            true_event5=int(np.sum(true_event5)),
            pred_event5=int(np.sum(pred_event5)),
            true_event7=int(np.sum(true_event7)),
            pred_event7=int(np.sum(pred_event7)),
            true_cyber=int(np.sum(true_cyber)),
            pred_cyber=int(np.sum(pred_cyber)),
        ),
        event7_thresholds_by_bus=event7_result.thresholds_by_bus,
        artifact_paths={
            "json_report": str((output_dir / "raw_cyber_quality_report.json").resolve()),
            "frame_predictions_csv": str(predictions_path.resolve()),
            "timeline_plot": str(timeline_plot.resolve()),
            "event7_score_histogram_plot": str(score_plot.resolve()),
            "event7_active_bus_count_plot": str(active_plot.resolve()),
        },
    )

    report_payload = report.to_dict()
    json_report_path = output_dir / "raw_cyber_quality_report.json"
    json_report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    return report_payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate simple event5/event7 estimators over RAW PMU data.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/RAW0001"),
        help="Path containing RAW PMU CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("report/raw_cyber_quality"),
        help="Directory where report json, csv and plots will be written.",
    )
    parser.add_argument(
        "--max-plot-points",
        type=int,
        default=8000,
        help="Maximum number of points to draw in timeline plot.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    payload = run_pipeline(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        max_plot_points=max(1000, int(args.max_plot_points)),
    )
    summary = {
        "n_frames": payload["n_frames"],
        "event5_f1": payload["event5_metrics"]["f1"],
        "event7_f1": payload["event7_metrics"]["f1"],
        "cyber_f1": payload["cyber_metrics"]["f1"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
