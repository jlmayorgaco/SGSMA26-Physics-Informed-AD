from __future__ import annotations

from src.simulation.m9.hardening import run_estimator_scoring
from tests.helpers.m9_test_utils import generate_templates


def test_estimator_in_the_loop_scoring(tmp_path) -> None:
    _, scenarios_root = generate_templates(tmp_path)
    dirs = sorted([p for p in scenarios_root.iterdir() if p.is_dir()])
    payload = run_estimator_scoring(dirs, tmp_path)
    assert payload["scenario_count"] >= 1
    assert (tmp_path / "metrics" / "scenario_scoring_table.csv").exists()

