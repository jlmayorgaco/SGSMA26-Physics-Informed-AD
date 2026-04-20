from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from src.application.use_cases.normalize_data import run_normalize_data_use_case


def _workspace_test_dir(prefix: str) -> Path:
    out_dir = Path("output") / "_test_runs" / f"{prefix}_{uuid4().hex[:8]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


@pytest.mark.integration
def test_m1_normalize_use_case_creates_expected_artifacts(raw_small_dir: Path) -> None:
    out_dir = _workspace_test_dir("m1_integration")

    run_normalize_data_use_case(
        input_dir=raw_small_dir,
        output_dir=out_dir,
        generate_plots=False,
        feature_mode="legacy_replace_angles",
    )

    assert out_dir.exists()
    assert (out_dir / "normalization_baselines.csv").exists()
    assert (out_dir / "chunks_metadata.json").exists()
    assert (out_dir / "normalized_chunk_index.json").exists()
    assert list((out_dir / "chunks").glob("chunk*/Bus*_normalized.csv"))
