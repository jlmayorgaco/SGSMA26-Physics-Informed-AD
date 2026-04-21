from __future__ import annotations

from pathlib import Path
import subprocess

from tests.helpers.detector_phase3_utils import build_phase3_splits


def test_phase3_infer_pipeline(tmp_path: Path) -> None:
    train_csv, val_csv, test_csv, scenario_for_infer = build_phase3_splits(tmp_path)
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
            "src.detectors.pipelines.m10_infer_detector",
            "--scenario-dir",
            str(scenario_for_infer),
            "--scenario-id",
            "P3INF",
            "--model-path",
            str(out / "models" / "detector_model.pkl"),
            "--threshold-config",
            str(out / "config" / "threshold_config.json"),
            "--output-root",
            str(out),
        ],
        cwd=cwd,
        check=True,
    )
    assert (out / "metrics" / "infer_P3INF_frame_predictions.csv").exists()
    assert (out / "metrics" / "infer_P3INF_chunks.csv").exists()
    assert (out / "infer_P3INF_summary.json").exists()
