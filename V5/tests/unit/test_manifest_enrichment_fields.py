from __future__ import annotations

from src.simulation.m9.hardening import enrich_scenario_labels_and_manifest
from tests.helpers.m9_test_utils import generate_single


def test_manifest_enrichment_fields(tmp_path) -> None:
    _, scenario_dir = generate_single(tmp_path, "TEMPLATE_EVENT1_FAULT", scenario_id="SIMX")
    m = enrich_scenario_labels_and_manifest(scenario_dir, difficulty_level="hard")
    for key in [
        "difficulty_level",
        "expected_detector_challenge",
        "expected_classifier_challenge",
        "expected_localizer_challenge",
        "expected_estimator_challenge",
        "event_visibility_rank",
        "electrical_distance_to_nearest_pmu",
        "scenario_family",
        "template_name",
        "realism_score",
        "training_value_score",
    ]:
        assert key in m

