from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.infrastructure.legacy.m0_adapter import run_chunk_pipeline


@pytest.mark.integration
def test_legacy_m0_pipeline_creates_core_artifacts(raw_small_dir: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "SCENARIO_RAW0001"
    run_chunk_pipeline(input_dir=str(raw_small_dir), output_dir=str(out_dir), enable_sanity_check=True, generate_plots=False)

    assert (out_dir / "chunks_metadata.json").exists()
    assert (out_dir / "chunk0_index.json").exists()

    payload = json.loads((out_dir / "chunks_metadata.json").read_text(encoding="utf-8"))
    assert "chunks" in payload
    assert len(payload["chunks"]) >= 1
