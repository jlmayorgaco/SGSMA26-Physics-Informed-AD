from __future__ import annotations

from src.application.use_cases.m7_state_estimator_benchmark import _verdict


def test_verdict_good_case() -> None:
    metrics = {
        "global_metrics": {"rmse_v_mag_all": 0.01, "rmse_v_mag_nonpmu_buses": 0.02},
        "robustness_metrics": {
            "mean_pmus_used_clean": 8.0,
            "pmu_voltage_fit_rmse": 0.01,
            "frozen_nonpmu_fraction": 0.1,
            "missing_data_degradation_ratio": 1.2,
            "solver_failure_rate": 0.0,
        },
    }
    th = {
        "min_clean_pmu_usage": 7.5,
        "max_pmu_fit_rmse": 0.03,
        "max_frozen_nonpmu_fraction": 0.7,
        "max_missing_data_degradation": 2.0,
        "max_solver_failure_rate": 0.2,
        "min_baseline_improvement": 0.01,
    }
    v = _verdict("PMU_VOLTAGE_CURRENT_WLS", metrics, baseline_nonpmu_rmse=0.05, thresholds=th)
    assert v["overall_status"] is True

