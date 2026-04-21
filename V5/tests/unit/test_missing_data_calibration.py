from __future__ import annotations

from src.simulation.m9.hardening import build_cyber_calibration_profile, evaluate_missing_data_patterns
from tests.helpers.m9_test_utils import REFERENCE_PMU_DIR, generate_templates


def test_missing_data_calibration_outputs(tmp_path) -> None:
    _, scenarios_root = generate_templates(tmp_path)
    dirs = sorted([p for p in scenarios_root.iterdir() if p.is_dir()])
    profile = build_cyber_calibration_profile(REFERENCE_PMU_DIR, tmp_path)
    metrics = evaluate_missing_data_patterns(dirs, tmp_path)
    assert "overall_missing_fraction" in profile
    assert "mean_missing_fraction" in metrics
    assert (tmp_path / "metadata" / "cyber_calibration_profile.json").exists()
    assert (tmp_path / "metrics" / "missing_data_pattern_metrics.csv").exists()

