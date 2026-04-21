from __future__ import annotations

from pathlib import Path
import subprocess

import pandas as pd

from tests.helpers.detector_phase1_utils import create_synthetic_scenario


def test_phase2_infer_smoke(tmp_path: Path) -> None:
    s0 = create_synthetic_scenario(tmp_path / "data", scenario_id="SIMA", abnormal_start_idx=60)
    s1 = create_synthetic_scenario(tmp_path / "data", scenario_id="SIMB", abnormal_start_idx=20)
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    pd.DataFrame(
        [
            {"scenario_id": "SIMA", "scenario_dir": str(s0), "split": "train", "template_name": "TEMPLATE_EVENT0_NORMAL", "event_coarse": 0, "difficulty_level": "easy", "scenario_family": "A", "seed_family": "A1"},
            {"scenario_id": "SIMB", "scenario_dir": str(s1), "split": "train", "template_name": "TEMPLATE_EVENT5_MISSING_ONLY", "event_coarse": 5, "difficulty_level": "medium", "scenario_family": "B", "seed_family": "B1"},
        ]
    ).to_csv(train_csv, index=False)
    pd.DataFrame(
        [{"scenario_id": "SIMB", "scenario_dir": str(s1), "split": "val", "template_name": "TEMPLATE_EVENT5_MISSING_ONLY", "event_coarse": 5, "difficulty_level": "medium", "scenario_family": "B", "seed_family": "B1"}]
    ).to_csv(val_csv, index=False)
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
            "src.detectors.pipelines.m10_infer_detector",
            "--scenario-dir",
            str(s1),
            "--scenario-id",
            "SIMB",
            "--model-path",
            str(out / "hybrid_detector.pkl"),
            "--output-root",
            str(out),
        ],
        cwd=cwd,
        check=True,
    )
    assert (out / "infer_SIMB_summary.json").exists()
    assert (out / "infer_SIMB_chunks.csv").exists()

