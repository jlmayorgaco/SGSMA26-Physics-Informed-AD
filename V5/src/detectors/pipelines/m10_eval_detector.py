from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.detectors.domain.models.detector_config import DetectorConfigV2
from src.detectors.domain.services.hybrid_event_detector import HybridEventDetector
from src.detectors.pipelines.common import build_frame_output, write_json
from src.detectors.training.datasets.detector_dataset_builder import DetectorDatasetBuilder, merge_detection_inputs
from src.detectors.training.datasets.detector_split_loader import load_split_csv
from src.detectors.training.evaluators.calibration_evaluator import CalibrationEvaluator
from src.detectors.training.evaluators.detector_evaluator import BinaryDetectorEvaluator
from src.detectors.training.evaluators.plotting import plot_calibration_curve, plot_confusion_matrix, plot_curve, plot_familywise_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M10 evaluate detector")
    p.add_argument("--split-csv", type=Path, default=Path("test_scenarios_v2.csv"))
    p.add_argument("--workspace-root", type=Path, default=Path("."))
    p.add_argument("--model-path", type=Path, default=Path("output/detector_m10_2_ready/models/detector_model.pkl"))
    p.add_argument("--output-root", type=Path, default=Path("output/detector_m10_2_ready"))
    p.add_argument("--run-name", type=str, default="hardened_test")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root
    metrics_dir = output_root / "metrics"
    plots_dir = output_root / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    detector = HybridEventDetector.load(args.model_path)
    builder = DetectorDatasetBuilder(detector.config if isinstance(detector.config, DetectorConfigV2) else DetectorConfigV2())
    if detector.preprocessing_state is not None:
        builder.normalization_state = detector.preprocessing_state  # type: ignore[assignment]
        builder.feature_columns = list(detector.feature_names)

    records = load_split_csv(args.split_csv, split_name=args.run_name, workspace_root=args.workspace_root.resolve())
    dataset = builder.build_for_records(records, fit=False)
    inputs = merge_detection_inputs(dataset.by_scenario.values(), scenario_id=f"{args.run_name}_MERGED", split=args.run_name)
    output = detector.predict(inputs)

    if "y_binary" in inputs.metadata.columns:
        y_true = inputs.metadata["y_binary"].to_numpy(dtype=int)
    else:
        y_true = np.zeros((inputs.x_windows.shape[0],), dtype=int)
    frame = build_frame_output(output, y_true, inputs.timestamps, inputs.metadata)

    evaluator = BinaryDetectorEvaluator()
    metrics = evaluator.metric_bundle(
        y_true=y_true,
        y_pred=output.y_pred_stable,
        y_score=output.p_abnormal,
        timestamps=inputs.timestamps,
        scenario_ids=inputs.metadata["scenario_id"].to_numpy() if "scenario_id" in inputs.metadata.columns else None,
    )
    familywise = evaluator.familywise_metrics(frame)
    per_scenario = evaluator.per_scenario_metrics(frame)
    roc_frame, pr_frame = evaluator.curve_points(y_true, output.p_abnormal)
    cal = CalibrationEvaluator(n_bins=10).evaluate(y_true, output.p_abnormal)

    frame.to_csv(metrics_dir / f"{args.run_name}_frame_predictions_v2.csv", index=False)
    frame.to_csv(metrics_dir / f"{args.run_name}_frame_predictions_v4.csv", index=False)
    per_scenario.to_csv(metrics_dir / f"{args.run_name}_per_scenario_metrics_v2.csv", index=False)
    per_scenario.to_csv(metrics_dir / f"{args.run_name}_per_scenario_metrics_v4.csv", index=False)
    roc_frame.to_csv(metrics_dir / f"{args.run_name}_roc_curve_points_v2.csv", index=False)
    pr_frame.to_csv(metrics_dir / f"{args.run_name}_pr_curve_points_v2.csv", index=False)
    cal.bins.to_csv(metrics_dir / f"{args.run_name}_calibration_bins_v2.csv", index=False)

    plot_confusion_matrix(dict(metrics.get("confusion", {})), plots_dir / f"{args.run_name}_confusion_matrix_v2.png")
    plot_curve(pr_frame, x="recall", y="precision", output_path=plots_dir / f"{args.run_name}_pr_curve_v2.png", title="PR Curve", xlabel="Recall", ylabel="Precision")
    plot_curve(roc_frame, x="fpr", y="tpr", output_path=plots_dir / f"{args.run_name}_roc_curve_v2.png", title="ROC Curve", xlabel="FPR", ylabel="TPR")
    plot_calibration_curve(cal.bins, plots_dir / f"{args.run_name}_calibration_curve_v2.png")
    plot_familywise_metrics(familywise, plots_dir / f"{args.run_name}_familywise_metrics_v2.png")

    report = {
        "run_name": args.run_name,
        "split_csv": str(args.split_csv),
        "windows": int(inputs.x_windows.shape[0]),
        "metrics": {**metrics, "calibration_ece": float(cal.ece), "brier_score": float(cal.brier)},
        "familywise": familywise,
        "artifacts": {
            "frame_predictions": str(metrics_dir / f"{args.run_name}_frame_predictions_v2.csv"),
            "per_scenario_metrics": str(metrics_dir / f"{args.run_name}_per_scenario_metrics_v2.csv"),
        },
    }
    write_json(metrics_dir / f"{args.run_name}_eval_report_v2.json", report)
    write_json(metrics_dir / f"{args.run_name}_eval_report_v4.json", report)
    # backward-compatible files
    write_json(output_root / f"{args.run_name}_eval_report.json", report)
    frame.to_csv(output_root / f"{args.run_name}_frame_predictions.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
