from __future__ import annotations

from src.application.use_cases.m7_state_estimator_benchmark import _score


def test_ranking_logic_deterministic() -> None:
    a = {
        "global_metrics": {"rmse_v_mag_nonpmu_buses": 0.02, "rmse_ang_nonpmu_buses": 5.0},
        "robustness_metrics": {"pmu_voltage_fit_rmse": 0.01, "missing_data_degradation_ratio": 1.2},
        "runtime": {"mean_runtime_per_frame_s": 0.001},
    }
    b = {
        "global_metrics": {"rmse_v_mag_nonpmu_buses": 0.03, "rmse_ang_nonpmu_buses": 6.0},
        "robustness_metrics": {"pmu_voltage_fit_rmse": 0.02, "missing_data_degradation_ratio": 1.5},
        "runtime": {"mean_runtime_per_frame_s": 0.001},
    }
    assert _score(a) < _score(b)

