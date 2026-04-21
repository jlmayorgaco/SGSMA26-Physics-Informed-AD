from __future__ import annotations

from pathlib import Path
import subprocess

from tests.helpers.detector_phase3_utils import build_phase3_splits


def test_phase3_eval_pipeline(tmp_path: Path) -> None:
    train_csv, val_csv, test_csv, _ = build_phase3_splits(tmp_path)
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
            "phase3_eval",
        ],
        cwd=cwd,
        check=True,
    )
    assert (out / "metrics" / "phase3_eval_eval_report.json").exists()
    assert (out / "plots" / "phase3_eval_roc_curve.png").exists()
    assert (out / "plots" / "phase3_eval_calibration_curve.png").exists()
