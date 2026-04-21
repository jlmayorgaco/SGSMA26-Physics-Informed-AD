from __future__ import annotations

from src.estimation.state_estimation.estimator_variants import build_estimator_registry


def test_estimator_registry_contains_expected_variants() -> None:
    reg = build_estimator_registry()
    expected = {
        "PRIOR_ONLY_BASELINE",
        "PMU_VOLTAGE_WLS",
        "PMU_VOLTAGE_CURRENT_WLS",
        "PMU_VOLTAGE_CURRENT_ADAPTIVE_REG",
        "PMU_VOLTAGE_CURRENT_SMOOTHED",
    }
    assert expected.issubset(set(reg.keys()))

