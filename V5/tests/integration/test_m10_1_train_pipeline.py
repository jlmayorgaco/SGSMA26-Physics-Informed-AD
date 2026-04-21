from __future__ import annotations

from pathlib import Path
import subprocess

from tests.helpers.detector_m10_1_utils import create_m10_1_family_split_inputs


def test_m10_1_train_pipeline(tmp_path: Path) -> None:
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
    assert (out / "models" / "detector_model.pkl").exists()
    assert (out / "metrics" / "detector_training_report_v2.json").exists()
    assert (out / "report" / "m10_1_hardening_report.json").exists()
