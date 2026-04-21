from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.detectors.domain.models.detector_config import DetectorConfigV2
from src.detectors.domain.models.detection_input import DetectionInput
from src.detectors.domain.models.detection_output import DetectionOutput
from src.detectors.domain.services.hybrid_event_detector import HybridEventDetector
from src.detectors.pipelines.common import build_frame_output, write_json
from src.detectors.pipelines.raw_holdout_loader import (
    RawScenarioRef,
    event_family_from_event,
    load_raw_holdout_frames,
)
from src.detectors.training.datasets.detector_dataset_builder import DetectorDatasetBuilder, merge_detection_inputs
from src.detectors.training.datasets.detector_split_loader import SplitRecord
from src.detectors.training.evaluators.calibration_evaluator import CalibrationEvaluator
from src.detectors.training.evaluators.detector_evaluator import BinaryDetectorEvaluator
from src.detectors.training.evaluators.plotting import plot_confusion_matrix, plot_curve


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M10 raw holdout validation for the frozen detector")
    p.add_argument("--raw-input-root", type=Path, default=Path("data/RAW0001"))
    p.add_argument("--model-path", type=Path, default=Path("output/detector_m10_2_ready/models/detector_model.pkl"))
    p.add_argument("--threshold-config-path", type=Path, default=Path("output/detector_m10_2_ready/config/threshold_config_v4.json"))
    p.add_argument("--output-root", type=Path, default=Path("output/detector_raw_holdout_eval"))
    p.add_argument("--run-name", type=str, default="raw_holdout")
    return p.parse_args()


def _scenario_family(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "unknown"
    source_col = "event_coarse" if "event_coarse" in frame.columns else "event_max" if "event_max" in frame.columns else None
    if source_col is None:
        return "unknown"
    values = pd.to_numeric(frame[source_col], errors="coerce").fillna(0).astype(int).to_numpy()
    non_zero = values[values != 0]
    if len(non_zero) == 0:
        return "normal"
    families = {event_family_from_event(v) for v in np.unique(non_zero)}
    if len(families) == 1:
        return next(iter(families))
    if families <= {"physical_heavy", "cyber_heavy"}:
        return "concurrent_heavy"
    if "concurrent_heavy" in families:
        return "concurrent_heavy"
    return "mixed"


def _extract_intervals(labels: np.ndarray, timestamps: np.ndarray) -> list[dict[str, Any]]:
    labels = np.asarray(labels, dtype=int)
    timestamps = np.asarray(timestamps, dtype=float)
    rows: list[dict[str, Any]] = []
    start_idx: int | None = None
    for idx, value in enumerate(labels):
        if value == 1 and start_idx is None:
            start_idx = idx
        elif value == 0 and start_idx is not None:
            end_idx = idx - 1
            rows.append(_interval_row(labels, timestamps, start_idx, end_idx))
            start_idx = None
    if start_idx is not None:
        rows.append(_interval_row(labels, timestamps, start_idx, len(labels) - 1))
    return rows


def _interval_row(labels: np.ndarray, timestamps: np.ndarray, start_idx: int, end_idx: int) -> dict[str, Any]:
    start_ts = float(timestamps[start_idx])
    end_ts = float(timestamps[end_idx])
    duration_s = max(0.0, end_ts - start_ts)
    return {
        "start_idx": int(start_idx),
        "end_idx": int(end_idx),
        "start_timestamp": start_ts,
        "end_timestamp": end_ts,
        "duration_s": duration_s,
        "frame_count": int(end_idx - start_idx + 1),
        "positive_frames": int(np.asarray(labels[start_idx : end_idx + 1], dtype=int).sum()),
    }


def _interval_overlap(a: dict[str, Any], b: dict[str, Any]) -> float:
    start = max(float(a["start_timestamp"]), float(b["start_timestamp"]))
    end = min(float(a["end_timestamp"]), float(b["end_timestamp"]))
    return max(0.0, end - start)


def _interval_iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    overlap = _interval_overlap(a, b)
    union = max(float(a["duration_s"]), 0.0) + max(float(b["duration_s"]), 0.0) - overlap
    return float(overlap / union) if union > 0 else 0.0


def _align_intervals(
    *,
    scenario_id: str,
    true_intervals: list[dict[str, Any]],
    pred_intervals: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    alignment_rows: list[dict[str, Any]] = []
    true_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    matched_pred: set[int] = set()

    for true_index, true_item in enumerate(true_intervals):
        best_index: int | None = None
        best_overlap = 0.0
        best_iou = 0.0
        for pred_index, pred_item in enumerate(pred_intervals):
            overlap = _interval_overlap(true_item, pred_item)
            if overlap > best_overlap:
                best_overlap = overlap
                best_index = pred_index
                best_iou = _interval_iou(true_item, pred_item)
        matched = best_index is not None and best_overlap > 0.0
        if matched and best_index is not None:
            matched_pred.add(best_index)
        delay = None
        if matched and best_index is not None:
            delay = max(0.0, float(pred_intervals[best_index]["start_timestamp"]) - float(true_item["start_timestamp"]))
        row = {
            "scenario_id": scenario_id,
            "interval_role": "true",
            "interval_index": true_index,
            "start_timestamp": float(true_item["start_timestamp"]),
            "end_timestamp": float(true_item["end_timestamp"]),
            "duration_s": float(true_item["duration_s"]),
            "frame_count": int(true_item["frame_count"]),
            "status": "matched" if matched else "missed",
            "best_match_index": best_index,
            "overlap_s": float(best_overlap),
            "iou": float(best_iou),
            "delay_s": delay,
        }
        alignment_rows.append(row)
        true_rows.append(row)

    for pred_index, pred_item in enumerate(pred_intervals):
        best_index = None
        best_overlap = 0.0
        best_iou = 0.0
        for true_index, true_item in enumerate(true_intervals):
            overlap = _interval_overlap(true_item, pred_item)
            if overlap > best_overlap:
                best_overlap = overlap
                best_index = true_index
                best_iou = _interval_iou(true_item, pred_item)
        row = {
            "scenario_id": scenario_id,
            "interval_role": "predicted",
            "interval_index": pred_index,
            "start_timestamp": float(pred_item["start_timestamp"]),
            "end_timestamp": float(pred_item["end_timestamp"]),
            "duration_s": float(pred_item["duration_s"]),
            "frame_count": int(pred_item["frame_count"]),
            "status": "matched" if pred_index in matched_pred else "false_alarm",
            "best_match_index": best_index,
            "overlap_s": float(best_overlap),
            "iou": float(best_iou),
            "delay_s": None,
        }
        alignment_rows.append(row)
        pred_rows.append(row)

    return alignment_rows, {"true": true_rows, "predicted": pred_rows}


def _plot_timeline(frame: pd.DataFrame, output_path: Path, *, title: str, score_only: bool = False) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        fig, ax = plt.subplots(figsize=(8.0, 3.0))
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        fig.tight_layout()
        fig.savefig(output_path, dpi=140)
        plt.close(fig)
        return

    scenario_ids = frame["scenario_id"].astype(str).tolist() if "scenario_id" in frame.columns else ["scenario"]
    unique_ids = list(dict.fromkeys(scenario_ids))
    rows = max(1, len(unique_ids))
    fig, axes = plt.subplots(rows, 1, figsize=(12.0, max(3.2, 2.8 * rows)), sharex=False)
    axes = np.atleast_1d(axes)
    for ax, scenario_id in zip(axes, unique_ids):
        group = frame.loc[frame["scenario_id"].astype(str) == scenario_id].sort_values("timestamp").reset_index(drop=True)
        ts = group["timestamp"].to_numpy(dtype=float)
        ax.set_title(str(scenario_id))
        if not score_only:
            ax.step(ts, group["y_true"].to_numpy(dtype=int), where="mid", label="true", alpha=0.85, linewidth=1.8)
            ax.step(ts, group["y_pred_stable"].to_numpy(dtype=int), where="mid", label="pred_stable", alpha=0.75, linewidth=1.6)
            ax.fill_between(ts, 0.0, group["p_abnormal"].to_numpy(dtype=float), step="mid", alpha=0.18, label="p_abnormal")
        else:
            ax.plot(ts, group["p_abnormal"].to_numpy(dtype=float), label="p_abnormal", linewidth=1.8)
            ax.plot(ts, group["p_cyber"].to_numpy(dtype=float), label="p_cyber", linewidth=1.2, alpha=0.8)
            ax.plot(ts, group["p_physical"].to_numpy(dtype=float), label="p_physical", linewidth=1.2, alpha=0.8)
            ax.step(ts, group["y_true"].to_numpy(dtype=int), where="mid", label="true", alpha=0.6, linewidth=1.4)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def _load_detector(detector_path: Path) -> HybridEventDetector:
    detector = HybridEventDetector.load(detector_path)
    if not isinstance(detector.config, DetectorConfigV2):
        detector.config = DetectorConfigV2()
    return detector


def _build_raw_scenario_records(raw_input_root: Path) -> list[RawScenarioRef]:
    return load_raw_holdout_frames(raw_input_root)


def _empty_detection_input(*, scenario_id: str, split: str, window_size: int, feature_names: list[str]) -> DetectionInput:
    return DetectionInput(
        scenario_id=scenario_id,
        split=split,
        x_windows=np.zeros((0, int(window_size), max(len(feature_names), 1)), dtype=float),
        timestamps=np.zeros((0,), dtype=float),
        feature_names=list(feature_names),
        metadata=pd.DataFrame(),
    )


def main() -> int:
    args = parse_args()
    output_root = args.output_root
    metrics_dir = output_root / "metrics"
    plots_dir = output_root / "plots"
    report_dir = output_root / "report"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    detector = _load_detector(args.model_path)
    builder = DetectorDatasetBuilder(detector.config)
    if detector.preprocessing_state is None:
        raise RuntimeError(f"Detector at {args.model_path} does not expose a preprocessing state")
    builder.normalization_state = detector.preprocessing_state  # type: ignore[assignment]
    builder.feature_columns = list(detector.feature_names)

    scenario_refs = _build_raw_scenario_records(args.raw_input_root.resolve())
    scenario_frames: list[pd.DataFrame] = []
    interval_rows: list[dict[str, Any]] = []
    scenario_metrics_rows: list[dict[str, Any]] = []
    raw_scenario_summaries: list[dict[str, Any]] = []

    for ref in scenario_refs:
        record = SplitRecord(
            scenario_id=ref.scenario_id,
            scenario_dir=ref.scenario_dir,
            split="raw_holdout",
            template_name="RAW_HOLDOUT",
            event_coarse=None,
            difficulty_level="external_holdout",
            scenario_family="raw_holdout",
            seed_family="",
        )
        dataset = builder.build_for_records([record], fit=False)
        if ref.scenario_id not in dataset.by_scenario:
            continue
        scenario_input = dataset.by_scenario[ref.scenario_id]
        if scenario_input.x_windows.shape[0] == 0:
            raw_scenario_summaries.append(
                {
                    "scenario_id": ref.scenario_id,
                    "scenario_family": "raw_holdout",
                    "true_intervals": 0,
                    "predicted_intervals": 0,
                    "missed_events": 0,
                    "false_alarm_intervals": 0,
                    "windows": 0,
                }
            )
            continue
        inputs = merge_detection_inputs([scenario_input], scenario_id=ref.scenario_id, split="raw_holdout")
        output: DetectionOutput = detector.predict(inputs)
        if inputs.metadata.empty:
            continue

        y_true_source = inputs.metadata.get("event_coarse", inputs.metadata.get("event_max", pd.Series([0] * len(inputs.metadata))))
        y_true = pd.to_numeric(y_true_source, errors="coerce").fillna(0).astype(int).gt(0).to_numpy(dtype=int)
        frame = build_frame_output(output, y_true, inputs.timestamps, inputs.metadata)
        frame["scenario_id"] = ref.scenario_id
        frame["raw_input_root"] = str(args.raw_input_root.resolve())
        frame["event_binary_truth"] = y_true
        frame["event_family_truth"] = frame.get("event_coarse", pd.Series(0, index=frame.index)).apply(event_family_from_event)
        frame["raw_event_family"] = frame["event_family_truth"]
        frame["scenario_family"] = _scenario_family(frame)
        scenario_frames.append(frame)

        evaluator = BinaryDetectorEvaluator()
        metrics = evaluator.metric_bundle(
            y_true=y_true,
            y_pred=output.y_pred_stable,
            y_score=output.p_abnormal,
            timestamps=inputs.timestamps,
            scenario_ids=inputs.metadata["scenario_id"].to_numpy() if "scenario_id" in inputs.metadata.columns else None,
        )
        scenario_metrics_rows.append(
            {
                "scenario_id": ref.scenario_id,
                "scenario_dir": str(ref.scenario_dir),
                "scenario_family": _scenario_family(frame),
                "windows": int(len(frame)),
                "abnormal_windows": int(y_true.sum()),
                "normal_windows": int((y_true == 0).sum()),
                **metrics,
            }
        )

        true_intervals = _extract_intervals(y_true, inputs.timestamps)
        pred_intervals = _extract_intervals(output.y_pred_stable.astype(int), inputs.timestamps)
        aligned_rows, aligned_groups = _align_intervals(
            scenario_id=ref.scenario_id,
            true_intervals=true_intervals,
            pred_intervals=pred_intervals,
        )
        interval_rows.extend(aligned_rows)
        raw_scenario_summaries.append(
            {
                "scenario_id": ref.scenario_id,
                "scenario_family": _scenario_family(frame),
                "true_intervals": len(true_intervals),
                "predicted_intervals": len(pred_intervals),
                "missed_events": int(sum(1 for row in aligned_groups["true"] if row["status"] == "missed")),
                "false_alarm_intervals": int(sum(1 for row in aligned_groups["predicted"] if row["status"] == "false_alarm")),
                "windows": int(len(frame)),
            }
        )

    if scenario_frames:
        raw_frame = pd.concat(scenario_frames, ignore_index=True)
        raw_frame = raw_frame.sort_values(["scenario_id", "timestamp"]).reset_index(drop=True)
    else:
        raw_frame = pd.DataFrame(
            columns=[
                "timestamp",
                "y_true",
                "p_abnormal",
                "p_cyber",
                "p_physical",
                "y_pred_frame",
                "y_pred_stable",
                "is_abnormal",
                "confidence",
                "scenario_id",
                "split",
                "window_index",
                "start_idx",
                "end_idx",
                "event_coarse",
                "y_binary",
                "event_family",
                "is_cyber_event",
                "is_physical_event",
                "is_concurrent_event",
                "data_present_ratio",
                "subtype",
                "origin",
                "difficulty_level",
                "window_size",
                "scenario_family",
                "raw_input_root",
                "event_binary_truth",
                "event_family_truth",
                "raw_event_family",
            ]
        )

    evaluator = BinaryDetectorEvaluator()
    global_metrics = evaluator.metric_bundle(
        y_true=raw_frame["y_true"].to_numpy(dtype=int),
        y_pred=raw_frame["y_pred_stable"].to_numpy(dtype=int),
        y_score=raw_frame["p_abnormal"].to_numpy(dtype=float),
        timestamps=raw_frame["timestamp"].to_numpy(dtype=float),
        scenario_ids=raw_frame["scenario_id"].to_numpy() if "scenario_id" in raw_frame.columns else None,
    )
    familywise = evaluator.familywise_metrics(raw_frame)
    per_scenario = evaluator.per_scenario_metrics(raw_frame)
    roc_frame, pr_frame = evaluator.curve_points(raw_frame["y_true"].to_numpy(dtype=int), raw_frame["p_abnormal"].to_numpy(dtype=float))
    cal = CalibrationEvaluator(n_bins=10).evaluate(raw_frame["y_true"].to_numpy(dtype=int), raw_frame["p_abnormal"].to_numpy(dtype=float))
    raw_ece = float(cal.ece)
    raw_brier = float(cal.brier)

    raw_frame.to_csv(metrics_dir / "raw_frame_predictions.csv", index=False)
    per_scenario.to_csv(metrics_dir / "raw_per_scenario_metrics.csv", index=False)
    pd.DataFrame(interval_rows).to_csv(metrics_dir / "raw_interval_alignment.csv", index=False)
    roc_frame.to_csv(metrics_dir / "raw_roc_curve.csv", index=False)
    pr_frame.to_csv(metrics_dir / "raw_pr_curve.csv", index=False)

    plot_confusion_matrix(global_metrics["confusion"], plots_dir / "raw_confusion_matrix.png")
    plot_curve(pr_frame, x="recall", y="precision", output_path=plots_dir / "raw_pr_curve.png", title="RAW PR Curve", xlabel="Recall", ylabel="Precision")
    plot_curve(roc_frame, x="fpr", y="tpr", output_path=plots_dir / "raw_roc_curve.png", title="RAW ROC Curve", xlabel="FPR", ylabel="TPR")
    _plot_timeline(raw_frame, plots_dir / "raw_timeline_overlay.png", title="RAW Timeline Overlay")
    _plot_timeline(raw_frame, plots_dir / "raw_score_over_time.png", title="RAW Score Over Time", score_only=True)

    raw_true_intervals = [row for row in interval_rows if row.get("interval_role") == "true"]
    raw_pred_intervals = [row for row in interval_rows if row.get("interval_role") == "predicted"]
    missed_events = [row for row in raw_true_intervals if row.get("status") == "missed"]
    false_alarm_intervals = [row for row in raw_pred_intervals if row.get("status") == "false_alarm"]

    synthetic_reference_path = args.model_path.parent.parent / "detector_training_report_v4.json"
    sim_reference = json.loads(synthetic_reference_path.read_text(encoding="utf-8")) if synthetic_reference_path.exists() else {}
    sim_test = sim_reference.get("metrics", {}).get("test", {})
    sim_f1 = float(sim_test.get("f1_abnormal", 0.0) or 0.0)
    sim_fp = float(sim_test.get("false_positives_per_minute", 0.0) or 0.0)
    sim_recall = float(sim_test.get("recall_abnormal", 0.0) or 0.0)

    sim_vs_raw_drop_notes = [
        f"Synthetic test F1 was {sim_f1:.4f}; raw holdout F1 is {float(global_metrics['f1_abnormal']):.4f}.",
        f"Synthetic test FP/min was {sim_fp:.4f}; raw holdout FP/min is {float(global_metrics['false_positives_per_minute']):.4f}.",
        f"Synthetic test recall was {sim_recall:.4f}; raw holdout recall is {float(global_metrics['recall_abnormal']):.4f}.",
    ]
    likely_failure_modes = []
    if float(global_metrics["false_positives_per_minute"]) > 0.5:
        likely_failure_modes.append("normal intervals may still trigger due to holdover tails or benign PMU drift")
    if float(global_metrics["recall_abnormal"]) < 0.8:
        likely_failure_modes.append("abnormal RAW events are being missed or delayed")
    if len(missed_events) > 0:
        likely_failure_modes.append("some abnormal intervals have no overlapping predicted abnormal window")
    if len(false_alarm_intervals) > 0:
        likely_failure_modes.append("some predicted abnormal intervals do not overlap any raw abnormal interval")
    if not likely_failure_modes:
        likely_failure_modes.append("no major failure mode surfaced in this holdout slice")

    detects_non_zero = bool(float(global_metrics["recall_abnormal"]) > 0.0 and int(global_metrics["support_abnormal"]) > 0)
    usable_on_raw = bool(
        detects_non_zero
        and float(global_metrics["f1_abnormal"]) >= 0.50
        and float(global_metrics["false_positives_per_minute"]) <= 2.0
        and raw_ece <= 0.15
    )

    verdict_main_strengths = [
        "recovers non-zero events on RAW if abnormal support is present",
        "uses the frozen detector without any RAW training or threshold retuning",
    ]
    verdict_main_failures = []
    if float(global_metrics["false_positives_per_minute"]) > 1.0:
        verdict_main_failures.append("false alarms remain elevated on RAW")
    if float(global_metrics["recall_abnormal"]) < 0.8:
        verdict_main_failures.append("some non-zero RAW events are missed")
    if float(global_metrics["precision_abnormal"]) < 0.8:
        verdict_main_failures.append("precision drops on RAW")
    if not verdict_main_failures:
        verdict_main_failures.append("no major RAW failure observed in this holdout slice")

    verdict_next_actions = [
        "expand RAW coverage with additional external holdout files if available",
        "inspect the false alarm intervals and missed-event intervals in raw_interval_alignment.csv",
    ]

    report_json: dict[str, Any] = {
        "run_metadata": {
            "raw_input_root": str(args.raw_input_root.resolve()),
            "model_path": str(args.model_path.resolve()),
            "threshold_config_path": str(args.threshold_config_path.resolve()),
            "window_size": int(detector.config.window.size),
            "window_stride": int(detector.config.window.stride),
            "scenarios_evaluated": int(len(scenario_frames)),
            "windows_evaluated": int(len(raw_frame)),
        },
        "evaluation_policy": {
            "mode": "inference_only_holdout",
            "binary_mapping": {
                "0": "normal",
                "1-8": "abnormal",
            },
            "no_retuning_on_raw": True,
            "no_retraining_on_raw": True,
            "frozen_model_used": True,
        },
        "metrics": {
            "global": {**global_metrics, "calibration_ece_test": raw_ece, "brier_score": raw_brier},
            "per_scenario": {
                "rows": int(len(per_scenario)),
                "records": per_scenario.to_dict(orient="records"),
            },
            "familywise": familywise,
        },
        "interval_alignment": {
            "true_abnormal_intervals": raw_true_intervals,
            "predicted_abnormal_intervals": raw_pred_intervals,
            "missed_events": missed_events,
            "false_alarm_intervals": false_alarm_intervals,
            "summary": {
                "true_intervals": int(len(raw_true_intervals)),
                "predicted_intervals": int(len(raw_pred_intervals)),
                "missed_events": int(len(missed_events)),
                "false_alarm_intervals": int(len(false_alarm_intervals)),
            },
        },
        "domain_shift_assessment": {
            "sim_vs_raw_drop_notes": sim_vs_raw_drop_notes,
            "likely_failure_modes": likely_failure_modes,
            "synthetic_reference": {
                "synthetic_test_f1_abnormal": sim_f1,
                "synthetic_test_false_positives_per_minute": sim_fp,
                "synthetic_test_recall_abnormal": sim_recall,
            },
        },
        "verdict": {
            "detects_non_zero_events_on_raw": detects_non_zero,
            "usable_on_raw": usable_on_raw,
            "main_strengths": verdict_main_strengths,
            "main_failures": verdict_main_failures,
            "next_actions": verdict_next_actions,
        },
        "artifacts": {
            "raw_frame_predictions": str(metrics_dir / "raw_frame_predictions.csv"),
            "raw_per_scenario_metrics": str(metrics_dir / "raw_per_scenario_metrics.csv"),
            "raw_interval_alignment": str(metrics_dir / "raw_interval_alignment.csv"),
            "raw_confusion_matrix": str(plots_dir / "raw_confusion_matrix.png"),
            "raw_pr_curve": str(plots_dir / "raw_pr_curve.png"),
            "raw_roc_curve": str(plots_dir / "raw_roc_curve.png"),
            "raw_timeline_overlay": str(plots_dir / "raw_timeline_overlay.png"),
            "raw_score_over_time": str(plots_dir / "raw_score_over_time.png"),
        },
        "scenario_summaries": raw_scenario_summaries,
    }
    write_json(metrics_dir / "raw_holdout_report.json", report_json)
    (metrics_dir / "raw_holdout_report.md").write_text(
        "\n".join(
            [
                "# RAW Holdout Validation",
                f"- raw_input_root: `{args.raw_input_root.resolve()}`",
                f"- model_path: `{args.model_path.resolve()}`",
                f"- windows_evaluated: `{len(raw_frame)}`",
                f"- detects_non_zero_events_on_raw: `{detects_non_zero}`",
                f"- usable_on_raw: `{usable_on_raw}`",
                f"- f1_abnormal: `{float(global_metrics['f1_abnormal']):.4f}`",
                f"- recall_abnormal: `{float(global_metrics['recall_abnormal']):.4f}`",
                f"- false_positives_per_minute: `{float(global_metrics['false_positives_per_minute']):.4f}`",
                f"- calibration_ece_test: `{raw_ece:.4f}`",
                f"- missed_events: `{len(missed_events)}`",
                f"- false_alarm_intervals: `{len(false_alarm_intervals)}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (report_dir / "raw_holdout_summary.md").write_text(
        "\n".join(
            [
                "# RAW Holdout Summary",
                f"- verdict: `{report_json['verdict']['usable_on_raw']}`",
                f"- detects_non_zero_events_on_raw: `{report_json['verdict']['detects_non_zero_events_on_raw']}`",
                f"- main_failures: `{'; '.join(verdict_main_failures)}`",
                f"- false alarms: `{len(false_alarm_intervals)}`",
                f"- missed events: `{len(missed_events)}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
