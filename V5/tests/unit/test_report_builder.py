from __future__ import annotations

import json
from pathlib import Path

from src.detectors.training.reports.detector_report_builder import write_detector_training_report


def test_report_builder_schema_fields(tmp_path: Path) -> None:
    payload = {
        "run_metadata": {"train_split": "train.csv", "val_split": "val.csv", "test_split": "test.csv", "reference_pmu_dir": "data", "output_root": str(tmp_path), "window_size": 16, "window_stride": 8, "train_windows": 10, "val_windows": 5, "test_windows": 5},
        "nan_and_data_present_handling": {"strategy": "x", "data_present_feature_used": True, "nan_count_after_preprocessing_train": 0, "nan_count_after_preprocessing_val": 0, "nan_count_after_preprocessing_test": 0},
        "threshold_tuning_validation": {"selected": {"threshold": 0.5, "start_threshold": 0.5, "stop_threshold": 0.4, "margin": 0.1, "min_on_frames": 2, "min_off_frames": 2}, "top_candidates": [], "rows": 1},
        "metrics": {"train": {}, "validation": {}, "test": {"f1_abnormal": 0.5, "precision_abnormal": 0.5, "recall_abnormal": 0.5, "false_positives_per_minute": 1.0, "brier_score": 0.2}, "confusion_matrix_test": {"tp": 1, "tn": 1, "fp": 1, "fn": 1}, "calibration_ece_test": 0.1, "cyber_vs_physical": {}, "raw_reference_normal_baseline": {}},
        "branch_summary": {"train": {}, "validation": {}, "test": {}},
        "downstream_readiness": {"binary_detector_ready_for_classifier_localizer": False, "verdict": "needs_hardening", "criteria": {}},
    }
    json_path, md_path = write_detector_training_report(payload, output_root=tmp_path, report_dir=tmp_path / "report")
    assert json_path.exists()
    assert md_path.exists()
    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert "run_metadata" in saved
    assert "threshold_tuning_validation" in saved
    assert "downstream_readiness" in saved
    assert (tmp_path / "report" / "detector_training_report.json").exists()
