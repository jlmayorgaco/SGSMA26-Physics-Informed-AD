from __future__ import annotations

from pathlib import Path
import json


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_detector_training_report(payload: dict, *, output_root: Path, report_dir: Path | None = None) -> tuple[Path, Path]:
    metrics_dir = output_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    json_path = metrics_dir / "detector_training_report.json"
    md_path = metrics_dir / "detector_training_report.md"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    test_metrics = payload.get("metrics", {}).get("test", {})
    val_metrics = payload.get("metrics", {}).get("validation", {})
    selected = payload.get("threshold_tuning_validation", {}).get("selected", {})
    readiness = payload.get("downstream_readiness", {})
    family = payload.get("metrics", {}).get("cyber_vs_physical", {})
    nan_cfg = payload.get("nan_and_data_present_handling", {})

    lines = [
        "# Detector Training Report",
        "",
        "## Model",
        "- Hybrid detector: cyber heuristics + tabular ML, physical temporal model + heuristics, rule-gated fusion, state-machine postprocessing.",
        "",
        "## NaN / DATA_PRESENT",
        f"- strategy: `{nan_cfg.get('strategy', '')}`",
        f"- data_present_feature_used: `{nan_cfg.get('data_present_feature_used', False)}`",
        f"- nan_count_before_preprocessing_train: `{nan_cfg.get('nan_count_before_preprocessing_train', 0)}`",
        f"- nan_count_before_preprocessing_val: `{nan_cfg.get('nan_count_before_preprocessing_val', 0)}`",
        f"- nan_count_before_preprocessing_test: `{nan_cfg.get('nan_count_before_preprocessing_test', 0)}`",
        f"- nan_count_after_preprocessing_train: `{nan_cfg.get('nan_count_after_preprocessing_train', 0)}`",
        f"- nan_count_after_preprocessing_val: `{nan_cfg.get('nan_count_after_preprocessing_val', 0)}`",
        f"- nan_count_after_preprocessing_test: `{nan_cfg.get('nan_count_after_preprocessing_test', 0)}`",
        "",
        "## Threshold Policy",
        f"- threshold/start/stop: `{_fmt(selected.get('threshold', 0.5))}` / `{_fmt(selected.get('start_threshold', 0.5))}` / `{_fmt(selected.get('stop_threshold', 0.4))}`",
        f"- margin: `{_fmt(selected.get('margin', 0.1))}`",
        f"- min_on_frames: `{selected.get('min_on_frames', 2)}`",
        f"- min_off_frames: `{selected.get('min_off_frames', 2)}`",
        "",
        "## Metrics",
        f"- validation_f1_abnormal: `{_fmt(float(val_metrics.get('f1_abnormal', 0.0)))}`",
        f"- validation_recall_abnormal: `{_fmt(float(val_metrics.get('recall_abnormal', 0.0)))}`",
        f"- test_f1_abnormal: `{_fmt(float(test_metrics.get('f1_abnormal', 0.0)))}`",
        f"- test_precision_abnormal: `{_fmt(float(test_metrics.get('precision_abnormal', 0.0)))}`",
        f"- test_recall_abnormal: `{_fmt(float(test_metrics.get('recall_abnormal', 0.0)))}`",
        f"- test_false_positives_per_minute: `{_fmt(float(test_metrics.get('false_positives_per_minute', 0.0)))}`",
        f"- test_detection_delay_s: `{test_metrics.get('detection_delay_s')}`",
        f"- test_roc_auc: `{test_metrics.get('roc_auc')}`",
        f"- test_pr_auc: `{test_metrics.get('pr_auc')}`",
        f"- test_brier_score: `{_fmt(float(test_metrics.get('brier_score', 0.0)))}`",
        f"- test_calibration_ece: `{payload.get('metrics', {}).get('calibration_ece_test')}`",
        "",
        "## Cyber vs Physical",
        f"- cyber_heavy_recall: `{family.get('cyber_heavy', {}).get('recall_abnormal')}`",
        f"- physical_heavy_recall: `{family.get('physical_heavy', {}).get('recall_abnormal')}`",
        f"- concurrent_heavy_recall: `{family.get('concurrent_heavy', {}).get('recall_abnormal')}`",
        "",
        "## Readiness",
        f"- binary_detector_ready_for_classifier_localizer: `{readiness.get('binary_detector_ready_for_classifier_localizer', False)}`",
        f"- verdict: `{readiness.get('verdict', 'not_ready')}`",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if report_dir is not None:
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "detector_training_report.json").write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
        (report_dir / "detector_training_report.md").write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    return json_path, md_path
