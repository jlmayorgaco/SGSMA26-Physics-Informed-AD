from __future__ import annotations

import json
from pathlib import Path
import subprocess

from tests.helpers.detector_m10_1_utils import create_m10_1_family_split_inputs


def test_m10_1_end_to_end(tmp_path: Path) -> None:
    train_csv, val_csv, test_csv, scenario_for_infer = create_m10_1_family_split_inputs(tmp_path)
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
            "m10_1_e2e",
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
            "M10_1_INF",
            "--model-path",
            str(out / "models" / "detector_model.pkl"),
            "--output-root",
            str(out),
        ],
        cwd=cwd,
        check=True,
    )

    report = json.loads((out / "report" / "m10_1_hardening_report.json").read_text(encoding="utf-8"))
    assert "readiness" in report
    if not report["readiness"]["binary_detector_ready_for_classifier_localizer"]:
        assert len(report["readiness"].get("failed_criteria", [])) > 0
