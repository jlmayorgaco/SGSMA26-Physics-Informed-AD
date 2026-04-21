from __future__ import annotations

import json
from pathlib import Path
import subprocess


def test_raw_holdout_report_schema(tmp_path: Path) -> None:
    cwd = Path(__file__).resolve().parents[2]
    out = tmp_path / "raw_holdout_schema_out"
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
            "raw_schema",
        ],
        cwd=cwd,
        check=True,
    )
    payload = json.loads((out / "metrics" / "raw_holdout_report.json").read_text(encoding="utf-8"))
    assert {"run_metadata", "evaluation_policy", "metrics", "interval_alignment", "domain_shift_assessment", "verdict"}.issubset(payload)
    assert payload["run_metadata"]["window_size"] > 0
    assert payload["evaluation_policy"]["no_retuning_on_raw"] is True
    assert "global" in payload["metrics"]
    assert "summary" in payload["interval_alignment"]
    assert "detects_non_zero_events_on_raw" in payload["verdict"]
