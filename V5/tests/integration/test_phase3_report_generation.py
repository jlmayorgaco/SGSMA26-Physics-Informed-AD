from __future__ import annotations

import json
from pathlib import Path
import subprocess

from tests.helpers.detector_phase3_utils import build_phase3_splits


def test_phase3_report_generation(tmp_path: Path) -> None:
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
    json_path = out / "metrics" / "detector_training_report.json"
    md_path = out / "metrics" / "detector_training_report.md"
    assert json_path.exists()
    assert md_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert "run_metadata" in payload
    assert "threshold_tuning_validation" in payload
    assert "downstream_readiness" in payload
    assert (Path("report") / "detector_training_report.json").exists()
    assert (Path("report") / "detector_training_report.md").exists()
