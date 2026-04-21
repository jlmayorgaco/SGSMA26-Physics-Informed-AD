from __future__ import annotations

import json
from pathlib import Path
import subprocess

from tests.helpers.detector_m10_1_utils import create_m10_1_family_split_inputs


def test_m10_1_hardening_e2e(tmp_path: Path) -> None:
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
    report_path = out / "report" / "m10_1_hardening_report.json"
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload.get("metrics", {}).get("test")
    assert (out / "metrics" / "threshold_sweep_v2.csv").exists()
    if not payload.get("readiness", {}).get("binary_detector_ready_for_classifier_localizer", False):
        assert payload.get("readiness", {}).get("verdict") in {"needs_more_data", "needs_more_hardening", "not_ready"}
