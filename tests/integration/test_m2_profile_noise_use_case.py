from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from src.application.use_cases.profile_noise import run_profile_noise_use_case


def _workspace_test_path(prefix: str) -> Path:
    out_dir = Path("output") / "_test_runs" / f"{prefix}_{uuid4().hex[:8]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "dataset_profiles_raw.json"


@pytest.mark.integration
def test_m2_profile_noise_use_case_generates_expected_structure() -> None:
    chunks_dir = Path("tests/fixtures/m2_chunks_small")
    output_path = _workspace_test_path("m2_integration")
    profiles = run_profile_noise_use_case(chunks_dir=chunks_dir, output_path=output_path)

    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert payload == profiles

    # event_type -> bus_id -> signal -> profile
    assert payload
    event_type = next(iter(payload.keys()))
    bus_id = next(iter(payload[event_type].keys()))
    signal = next(iter(payload[event_type][bus_id].keys()))
    profile = payload[event_type][bus_id][signal]
    assert "eda_stats" in profile
    assert "noise_model" in profile
    assert "diagnostics" in profile
    assert "sample_count" in profile
