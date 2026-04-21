from __future__ import annotations

import json

from tests.helpers.m9_test_utils import generate_single
from src.simulation.m9.validation import validate_scenario


def test_andes_plus_cyber_pipeline(tmp_path) -> None:
    _, scenario_dir = generate_single(tmp_path, "TEMPLATE_EVENT6_CONCURRENT_MISSING_PLUS_PHYSICAL", scenario_id="SIM0006")
    result = validate_scenario(scenario_dir, reference_pmu_dir="data/RAW0001", save_json=True)
    assert result["schema_checks"]["pmu_csvs_match_expected_schema"] is True
    assert result["missing_data_semantics_checks"]["BUS29"]["pass"] is True
    validation_path = scenario_dir / "metadata" / "scenario_validation.json"
    assert validation_path.exists()
    payload = json.loads(validation_path.read_text(encoding="utf-8"))
    assert payload["explicit_answers"]["physical_and_cyber_events_represented"] is True
