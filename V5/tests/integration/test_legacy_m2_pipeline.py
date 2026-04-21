from __future__ import annotations

from pathlib import Path

import pytest

from src.infrastructure.legacy.m0_adapter import run_chunk_pipeline
from src.infrastructure.legacy.m2_adapter import run_noise_profile_pipeline


@pytest.mark.integration
def test_legacy_m2_pipeline_generates_profile_json(raw_small_dir: Path, tmp_path: Path) -> None:
    m0_out = tmp_path / "SCENARIO_RAW0001"
    run_chunk_pipeline(input_dir=str(raw_small_dir), output_dir=str(m0_out), enable_sanity_check=False, generate_plots=False)

    profile_out = tmp_path / "raw_noise_profiles.json"
    profiles = run_noise_profile_pipeline(chunks_dir=str(m0_out / "chunks"), out_path=str(profile_out))

    assert profile_out.exists()
    assert isinstance(profiles, dict)
