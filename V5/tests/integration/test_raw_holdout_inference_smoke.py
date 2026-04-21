from __future__ import annotations

import json
from pathlib import Path
import subprocess


def test_raw_holdout_inference_smoke(tmp_path: Path) -> None:
    cwd = Path(__file__).resolve().parents[2]
    out = tmp_path / "raw_holdout_out"
    subprocess.run(
        [
            "python",
            "-m",
            "src.pipelines.m10_raw_holdout_eval",
            "--raw-input-root",
            str(Path("tests/fixtures/raw_small")),
            "--model-path",
            str(Path("output/detector_m10_2_ready/models/detector_model.pkl")),
            "--threshold-config-path",
            str(Path("output/detector_m10_2_ready/config/threshold_config_v4.json")),
            "--output-root",
            str(out),
            "--run-name",
            "raw_smoke",
        ],
        cwd=cwd,
        check=True,
    )
    report_path = out / "metrics" / "raw_holdout_report.json"
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["evaluation_policy"]["mode"] == "inference_only_holdout"
    assert "global" in payload["metrics"]
    assert (out / "metrics" / "raw_frame_predictions.csv").exists()
    assert (out / "metrics" / "raw_interval_alignment.csv").exists()
    assert (out / "plots" / "raw_confusion_matrix.png").exists()
