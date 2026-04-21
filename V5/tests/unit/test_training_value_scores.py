from __future__ import annotations

from src.simulation.m9.hardening import score_scenario_for_estimator
from tests.helpers.m9_test_utils import generate_single


def test_training_value_scores(tmp_path) -> None:
    _, scenario_dir = generate_single(tmp_path, "TEMPLATE_EVENT6_CONCURRENT_MISSING_PLUS_PHYSICAL", scenario_id="SIMTV")
    s = score_scenario_for_estimator(scenario_dir)
    assert 0.0 <= s["overall_training_value_score"] <= 1.0
    assert 0.0 <= s["estimator_difficulty_score"] <= 1.0

