from __future__ import annotations

from pathlib import Path
import subprocess
import json

import pandas as pd

from tests.helpers.detector_phase1_utils import create_synthetic_scenario


def test_phase2_detector_e2e(tmp_path: Path) -> None:
    s0 = create_synthetic_scenario(tmp_path / "data", scenario_id="SIME2E0", abnormal_start_idx=58)
    s1 = create_synthetic_scenario(tmp_path / "data", scenario_id="SIME2E1", abnormal_start_idx=20)
    s2 = create_synthetic_scenario(tmp_path / "data", scenario_id="SIME2E2", abnormal_start_idx=30)
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    test_csv = tmp_path / "test.csv"
    pd.DataFrame(
        [
            {"scenario_id": "SIME2E0", "scenario_dir": str(s0), "split": "train", "template_name": "TEMPLATE_EVENT0_NORMAL", "event_coarse": 0, "difficulty_level": "easy", "scenario_family": "A", "seed_family": "A1"},
            {"scenario_id": "SIME2E1", "scenario_dir": str(s1), "split": "train", "template_name": "TEMPLATE_EVENT5_MISSING_ONLY", "event_coarse": 5, "difficulty_level": "medium", "scenario_family": "B", "seed_family": "B1"},
        ]
    ).to_csv(train_csv, index=False)
    pd.DataFrame(
        [{"scenario_id": "SIME2E2", "scenario_dir": str(s2), "split": "val", "template_name": "TEMPLATE_EVENT6_CONCURRENT_MISSING_PLUS_PHYSICAL", "event_coarse": 6, "difficulty_level": "hard", "scenario_family": "C", "seed_family": "C1"}]
    ).to_csv(val_csv, index=False)
    pd.DataFrame(
        [{"scenario_id": "SIME2E1", "scenario_dir": str(s1), "split": "test", "template_name": "TEMPLATE_EVENT5_MISSING_ONLY", "event_coarse": 5, "difficulty_level": "medium", "scenario_family": "B", "seed_family": "B1"}]
    ).to_csv(test_csv, index=False)
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
            str(out / "hybrid_detector.pkl"),
            "--output-root",
            str(out),
            "--run-name",
            "e2e",
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
            str(s2),
            "--scenario-id",
            "SIME2E2",
            "--model-path",
            str(out / "hybrid_detector.pkl"),
            "--output-root",
            str(out),
        ],
        cwd=cwd,
        check=True,
    )
    report_path = out / "e2e_eval_report.json"
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert "metrics" in payload
