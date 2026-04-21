from __future__ import annotations

import json
import shutil
from pathlib import Path

from src.detectors.v6.raw_eval import run_v6_cyber_raw_eval


def test_v6_raw_eval_writes_reports(tmp_path) -> None:
    fixture_chunks = Path("tests/fixtures/m2_chunks_small")
    m0_root = tmp_path / "m0_like"
    chunks_root = m0_root / "chunks"
    shutil.copytree(fixture_chunks, chunks_root)

    metadata = {
        "chunks": [
            {"chunk_dir": "chunk01_event_0_normal_operation", "label": 0},
            {"chunk_dir": "chunk02_event_5_missing_data", "label": 5},
        ]
    }
    (m0_root / "chunks_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    output_root = tmp_path / "out"
    report = run_v6_cyber_raw_eval(m0_root=m0_root, output_root=output_root)

    assert report["summary"]["n_chunks"] == 2
    assert report["summary"]["n_true_event5"] == 1
    assert "event5_binary" in report["metrics"]
    assert (output_root / "v6_cyber_raw_eval.json").exists()
    assert (output_root / "v6_cyber_raw_eval.md").exists()
    assert (output_root / "v6_cyber_raw_predictions.csv").exists()
