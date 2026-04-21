from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.application.use_cases.chunk_raw_data import run_chunk_raw_data_use_case


@pytest.mark.integration
def test_m0_chunk_use_case_creates_expected_artifacts(raw_small_dir: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "SCENARIO_RAW0001"

    run_chunk_raw_data_use_case(
        input_dir=str(raw_small_dir),
        output_dir=str(out_dir),
        enable_sanity_check=True,
        generate_plots=False,
    )

    assert (out_dir / "chunks").exists()
    assert (out_dir / "chunks_metadata.json").exists()
    assert (out_dir / "chunk0_index.json").exists()
    assert (out_dir / "raw_signal_sanity_summary.csv").exists()

    payload = json.loads((out_dir / "chunks_metadata.json").read_text(encoding="utf-8"))
    assert "chunks" in payload
    assert len(payload["chunks"]) >= 1

    sanity_df = pd.read_csv(out_dir / "raw_signal_sanity_summary.csv")
    assert not sanity_df.empty
