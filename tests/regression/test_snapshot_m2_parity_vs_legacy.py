from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from src.application.use_cases.chunk_raw_data import run_chunk_raw_data_use_case
from src.application.use_cases.profile_noise import run_profile_noise_use_case
from src.infrastructure.legacy.m2_adapter import run_noise_profile_pipeline


def _workspace_test_dir(prefix: str) -> Path:
    root = Path("output") / "_test_runs" / f"{prefix}_{uuid4().hex[:8]}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_payload(payload: dict) -> dict:
    # Key order in dicts is irrelevant; semantic dict equality is parity criterion.
    return payload


@pytest.mark.regression
def test_m2_migrated_pipeline_matches_legacy_profiles(raw_small_dir: Path) -> None:
    root = _workspace_test_dir("m2_regression")
    m0_out = root / "m0_chunks"
    migrated_out = root / "migrated_profiles.json"
    legacy_out = root / "legacy_profiles.json"

    run_chunk_raw_data_use_case(
        input_dir=str(raw_small_dir),
        output_dir=str(m0_out),
        enable_sanity_check=False,
        generate_plots=False,
    )

    run_profile_noise_use_case(chunks_dir=m0_out / "chunks", output_path=migrated_out)
    run_noise_profile_pipeline(chunks_dir=str(m0_out / "chunks"), out_path=str(legacy_out))

    assert migrated_out.exists()
    assert legacy_out.exists()

    migrated_payload = _load_json(migrated_out)
    legacy_payload = _load_json(legacy_out)
    assert _normalize_payload(migrated_payload) == _normalize_payload(legacy_payload)
