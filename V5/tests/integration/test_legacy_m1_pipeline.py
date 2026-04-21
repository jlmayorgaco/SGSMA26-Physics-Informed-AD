from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.infrastructure.legacy.m1_adapter import run_normalization_pipeline


@pytest.mark.integration
def test_legacy_m1_pipeline_creates_core_artifacts(raw_small_dir: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "SCENARIO_RAW0001_NORMALIZED"
    run_normalization_pipeline(input_dir=str(raw_small_dir), output_dir=str(out_dir), generate_plots=False)

    assert (out_dir / "normalization_baselines.csv").exists()
    assert (out_dir / "normalized_chunk_index.json").exists()

    payload = json.loads((out_dir / "normalized_chunk_index.json").read_text(encoding="utf-8"))
    assert "chunks" in payload
    assert isinstance(payload["chunks"], list)
