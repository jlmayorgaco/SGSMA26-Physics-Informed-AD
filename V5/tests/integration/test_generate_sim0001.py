from __future__ import annotations

import json

from tests.helpers.m9_test_utils import generate_single


def test_generate_sim0001(tmp_path) -> None:
    manifest, scenario_dir = generate_single(tmp_path, "TEMPLATE_OFFICIAL_STYLE_MULTI_EVENT", scenario_id="SIM0001")
    assert manifest["scenario_id"] == "SIM0001"
    assert (scenario_dir / "scenario_manifest.json").exists()
    assert (scenario_dir / "all_buses" / "all_bus_truth.csv").exists()
    assert (scenario_dir / "all_buses" / "full_state_target.csv").exists()
    assert len(list((scenario_dir / "pmu").glob("Bus*_Competition_Data_sim.csv"))) == 8
    payload = json.loads((scenario_dir / "scenario_manifest.json").read_text(encoding="utf-8"))
    assert payload["physical_events"]
    assert payload["cyber_events"]
