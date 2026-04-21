from __future__ import annotations

from tests.helpers.m9_test_utils import generate_validation_suite


def test_m9_e2e_sim0001_and_readiness_gate(tmp_path) -> None:
    report = generate_validation_suite(tmp_path)
    official = report["scenario_checks"]["OFFICIAL_STYLE_MULTI_EVENT"]
    assert official["schema_checks"]["pmu_csvs_match_expected_schema"]
    assert official["all_bus_export_checks"]["pass"]
    assert official["label_consistency_checks"]["pass"]
    assert official["raw_vs_sim_comparison"] is not None
    # Assertion acts as nonzero exit gate for CI if downstream readiness fails.
    assert report["overall_verdict"]["m9_simulator_working"]
