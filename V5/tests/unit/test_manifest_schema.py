from __future__ import annotations

import json

from tests.helpers.m9_test_utils import generate_single


def test_manifest_schema(tmp_path) -> None:
    manifest, scenario_dir = generate_single(tmp_path, "TEMPLATE_OFFICIAL_STYLE_MULTI_EVENT", scenario_id="SIM0001")
    manifest_path = scenario_dir / "scenario_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "scenario_id",
        "generated_at_utc",
        "seed",
        "base_system",
        "mva_base",
        "rated_frequency_hz",
        "fps",
        "duration_s",
        "pmu_buses",
        "physical_events",
        "cyber_events",
        "concurrent_events",
        "noise_model",
        "calibration_reference",
        "output_files",
    }
    assert required.issubset(payload.keys())
    assert payload["scenario_id"] == "SIM0001"
    assert payload["physical_events"]
    assert payload["cyber_events"]
    assert manifest["scenario_template"] == "TEMPLATE_OFFICIAL_STYLE_MULTI_EVENT"
