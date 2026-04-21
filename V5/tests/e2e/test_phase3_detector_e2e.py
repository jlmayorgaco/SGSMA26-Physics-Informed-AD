from __future__ import annotations

import json
from pathlib import Path
import subprocess

from tests.helpers.detector_phase3_utils import build_phase3_splits


def test_phase3_detector_e2e(tmp_path: Path) -> None:
    train_csv, val_csv, test_csv, scenario_for_infer = build_phase3_splits(tmp_path)
    out = tmp_path / "out"
    cwd = Path(__file__).resolve().parents[2]

    subprocess.run(
        [
            "python",
            "-m",
            "src.detectors.pipelines.m10_train_detector",
            "--train-split",
            str(train_csv),
            "--val-split",
            str(val_csv),
            "--test-split",
            str(test_csv),
            "--workspace-root",
            str(cwd),
            "--output-root",
            str(out),
            "--window-size",
            "16",
            "--window-stride",
            "8",
        ],
        cwd=cwd,
        check=True,
    )
    subprocess.run(
        [
            "python",
            "-m",
            "src.detectors.pipelines.m10_eval_detector",
            "--split-csv",
            str(test_csv),
            "--workspace-root",
            str(cwd),
            "--model-path",
            str(out / "models" / "detector_model.pkl"),
            "--threshold-config",
            str(out / "config" / "threshold_config.json"),
            "--output-root",
            str(out),
            "--run-name",
            "phase3_e2e",
        ],
        cwd=cwd,
        check=True,
    )
    subprocess.run(
        [
            "python",
            "-m",
            "src.detectors.pipelines.m10_infer_detector",
            "--scenario-dir",
            str(scenario_for_infer),
            "--scenario-id",
            "P3E2EINF",
            "--model-path",
            str(out / "models" / "detector_model.pkl"),
            "--threshold-config",
            str(out / "config" / "threshold_config.json"),
            "--output-root",
            str(out),
        ],
        cwd=cwd,
        check=True,
    )

    required = [
        out / "metrics" / "detector_training_report.json",
        out / "metrics" / "detector_training_report.md",
        out / "metrics" / "per_scenario_metrics.csv",
        out / "metrics" / "threshold_sweep.csv",
        out / "metrics" / "test_frame_predictions.csv",
        out / "metrics" / "training_summary.json",
        out / "plots" / "confusion_matrix.png",
        out / "plots" / "pr_curve.png",
        out / "plots" / "roc_curve.png",
        out / "plots" / "calibration_curve.png",
    ]
    for path in required:
        assert path.exists(), f"missing required artifact: {path}"

    payload = json.loads((out / "metrics" / "detector_training_report.json").read_text(encoding="utf-8"))
    assert payload.get("metrics", {}).get("test")
    assert payload.get("threshold_tuning_validation", {}).get("rows", 0) > 0
