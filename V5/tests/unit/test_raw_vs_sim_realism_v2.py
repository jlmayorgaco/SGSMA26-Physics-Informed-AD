from __future__ import annotations

from src.simulation.m9.hardening import run_realism_validation_v2
from tests.helpers.m9_test_utils import REFERENCE_PMU_DIR, generate_templates


def test_raw_vs_sim_realism_v2(tmp_path) -> None:
    _, scenarios_root = generate_templates(tmp_path)
    scenario_dirs = sorted([p for p in scenarios_root.iterdir() if p.is_dir()])
    payload = run_realism_validation_v2(scenario_dirs, REFERENCE_PMU_DIR, tmp_path)
    assert "summary" in payload
    assert "pass_realism_v2" in payload
    assert (tmp_path / "metadata" / "raw_vs_sim_comparison_v2.json").exists()

