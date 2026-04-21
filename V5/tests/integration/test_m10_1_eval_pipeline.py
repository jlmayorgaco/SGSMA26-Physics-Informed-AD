from __future__ import annotations

from pathlib import Path
import subprocess

from tests.helpers.detector_m10_1_utils import create_m10_1_family_split_inputs


def test_m10_1_eval_pipeline(tmp_path: Path) -> None:
    train_csv, val_csv, test_csv, _ = create_m10_1_family_split_inputs(tmp_path)
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
            "--rebuild-splits-v3",
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
            str(out / "metadata" / "test_scenarios_v3.csv"),
            "--workspace-root",
            str(cwd),
            "--model-path",
            str(out / "models" / "detector_model.pkl"),
            "--output-root",
            str(out),
            "--run-name",
            "m10_1_eval",
        ],
        cwd=cwd,
        check=True,
    )
    assert (out / "metrics" / "m10_1_eval_eval_report_v2.json").exists()
    assert (out / "plots" / "m10_1_eval_calibration_curve_v2.png").exists()
