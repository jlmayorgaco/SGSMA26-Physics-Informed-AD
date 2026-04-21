from __future__ import annotations

from pathlib import Path
import subprocess

import pandas as pd

from tests.helpers.detector_phase1_utils import create_synthetic_scenario


def test_phase2_eval_smoke(tmp_path: Path) -> None:
    s0 = create_synthetic_scenario(tmp_path / "data", scenario_id="SIMA", abnormal_start_idx=55)
    s1 = create_synthetic_scenario(tmp_path / "data", scenario_id="SIMB", abnormal_start_idx=24)
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    test_csv = tmp_path / "test.csv"
    rows = [
        {"scenario_id": "SIMA", "scenario_dir": str(s0), "split": "train", "template_name": "TEMPLATE_EVENT0_NORMAL", "event_coarse": 0, "difficulty_level": "easy", "scenario_family": "A", "seed_family": "A1"},
        {"scenario_id": "SIMB", "scenario_dir": str(s1), "split": "train", "template_name": "TEMPLATE_EVENT5_MISSING_ONLY", "event_coarse": 5, "difficulty_level": "medium", "scenario_family": "B", "seed_family": "B1"},
    ]
    pd.DataFrame(rows).to_csv(train_csv, index=False)
    pd.DataFrame([rows[1] | {"split": "val"}]).to_csv(val_csv, index=False)
    pd.DataFrame([rows[0] | {"split": "test"}, rows[1] | {"split": "test"}]).to_csv(test_csv, index=False)
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
            "smoke",
        ],
        cwd=cwd,
        check=True,
    )
    assert (out / "smoke_eval_report.json").exists()
    assert (out / "smoke_frame_predictions.csv").exists()

