from __future__ import annotations

from pathlib import Path
import subprocess

import pandas as pd

from tests.helpers.m9_test_utils import PMU_LOCATION_PATH, RAW_PATH, REFERENCE_PMU_DIR
from src.simulation.m9.generator import generate_scenario


def _gen(tmp_path: Path, template: str, scenario_id: str, seed: int) -> Path:
    root = tmp_path / "scenarios"
    generate_scenario(
        raw_path=RAW_PATH,
        pmu_location_path=PMU_LOCATION_PATH,
        reference_pmu_dir=REFERENCE_PMU_DIR,
        output_root=root,
        scenario_template=template,
        scenario_id=scenario_id,
        seed=seed,
        use_andes=False,
        save_plots=False,
    )
    return (root / scenario_id).resolve()


def test_m10_train_eval_infer_pipeline(tmp_path: Path) -> None:
    s0 = _gen(tmp_path, "TEMPLATE_EVENT0_NORMAL", "SIMA", 41)
    s1 = _gen(tmp_path, "TEMPLATE_EVENT5_MISSING_ONLY", "SIMB", 42)
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    test_csv = tmp_path / "test.csv"
    frame = pd.DataFrame(
        [
            {"scenario_id": "SIMA", "scenario_dir": str(s0), "template_name": "TEMPLATE_EVENT0_NORMAL", "event_coarse": 0, "difficulty_level": "easy", "scenario_family": "A", "seed_family": "A1", "split": "train"},
            {"scenario_id": "SIMB", "scenario_dir": str(s1), "template_name": "TEMPLATE_EVENT5_MISSING_ONLY", "event_coarse": 5, "difficulty_level": "easy", "scenario_family": "B", "seed_family": "B1", "split": "train"},
        ]
    )
    frame.to_csv(train_csv, index=False)
    frame.iloc[[0]].assign(split="val").to_csv(val_csv, index=False)
    frame.iloc[[1]].assign(split="test").to_csv(test_csv, index=False)
    out_dir = tmp_path / "detector_out"
    cwd = Path(__file__).resolve().parents[2]
    subprocess.run(
        [
            "python",
            "-m",
            "src.pipelines.m10_train_detector",
            "--train-split",
            str(train_csv),
            "--val-split",
            str(val_csv),
            "--workspace-root",
            str(cwd),
            "--output-root",
            str(out_dir),
        ],
        cwd=cwd,
        check=True,
    )
    subprocess.run(
        [
            "python",
            "-m",
            "src.pipelines.m10_eval_detector",
            "--split-csv",
            str(test_csv),
            "--workspace-root",
            str(cwd),
            "--model-path",
            str(out_dir / "detector_model.pkl"),
            "--output-root",
            str(out_dir),
            "--run-name",
            "itest",
        ],
        cwd=cwd,
        check=True,
    )
    subprocess.run(
        [
            "python",
            "-m",
            "src.pipelines.m10_infer_detector",
            "--scenario-dir",
            str(s1),
            "--scenario-id",
            "SIMB",
            "--model-path",
            str(out_dir / "detector_model.pkl"),
            "--output-root",
            str(out_dir),
        ],
        cwd=cwd,
        check=True,
    )
    assert (out_dir / "detector_model.pkl").exists()
    assert (out_dir / "training_summary.json").exists()
    assert (out_dir / "itest_metrics.csv").exists()
    assert (out_dir / "infer_SIMB_summary.json").exists()

