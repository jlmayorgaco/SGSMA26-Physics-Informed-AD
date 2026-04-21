from __future__ import annotations

import numpy as np

from src.application.use_cases.m8_hybrid_dynamic_benchmark import _verdict


def test_verdict_logic_m8_good_case() -> None:
    payload = {
        "global_metrics": {"RMSE_V_MAG_NONPMU": 0.02, "RMSE_V_MAG_ALL": 0.015},
        "robustness_metrics": {
            "mean_pmus_used_clean": 8.0,
            "frozen_nonpmu_fraction": 0.2,
            "pmu_voltage_fit_rmse": 0.01,
            "missing_data_degradation_ratio": 1.2,
            "solver_failure_rate": 0.01,
        },
        "dynamic_metrics": {"dvdt_corr_mean": 0.4},
    }
    thresholds = {
        "min_clean_pmu_usage": 7.5,
        "max_frozen_nonpmu_fraction": 0.7,
        "max_pmu_fit_rmse": 0.05,
        "max_missing_data_degradation": 2.0,
        "max_solver_failure_rate": 0.2,
        "min_dvdt_corr": 0.1,
        "prior_baseline_rmse_nonpmu": 0.05,
    }
    v = _verdict(payload, thresholds, m7_reference_nonpmu_rmse=0.03)
    assert v["passes_truth_validation"] is True
    assert v["uses_all_valid_pmus"] is True
    assert v["better_than_m7_reference"] is True
    assert np.isfinite(float(payload["global_metrics"]["RMSE_V_MAG_NONPMU"]))

