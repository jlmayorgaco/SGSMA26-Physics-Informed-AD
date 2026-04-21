"""Estimator variants and registry for M7 benchmarking."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import time
from typing import Callable

import numpy as np

from src.estimation.state_estimation.measurement_model import FrameMeasurements
from src.estimation.state_estimation.models import EstimationConfig, NetworkModel
from src.estimation.state_estimation.pmu_state_estimator import PmuStateEstimator


EstimatorRunner = Callable[[NetworkModel, np.ndarray, list[FrameMeasurements], list[str]], dict]


@dataclass(slots=True)
class EstimatorSpec:
    name: str
    description: str
    runner: EstimatorRunner
    config: dict


def _to_frame(frame: FrameMeasurements, include_current: bool) -> FrameMeasurements:
    return FrameMeasurements(
        timestamp=frame.timestamp,
        voltage_by_bus=dict(frame.voltage_by_bus),
        current_by_bus=dict(frame.current_by_bus if include_current else {}),
        data_present_count=frame.data_present_count,
        expected_pmu_buses=frame.expected_pmu_buses,
        valid_pmu_buses=frame.valid_pmu_buses,
        dropped_reasons=frame.dropped_reasons,
    )


def run_prior_only(network: NetworkModel, prior: np.ndarray, frames: list[FrameMeasurements], window_types: list[str]) -> dict:
    t0 = time.perf_counter()
    states = np.repeat(np.asarray(prior, dtype=complex)[None, :], len(frames), axis=0)
    runtime = time.perf_counter() - t0
    diagnostics = [
        {
            "TIMESTAMP": float(f.timestamp),
            "FRAME_INDEX": i,
            "SOLVER_STATUS": "prior_only",
            "RESIDUAL_NORM": 0.0,
            "N_PMUS_USED_IN_SOLVER": len(f.valid_pmu_buses or []),
            "WINDOW_TYPE": window_types[i],
        }
        for i, f in enumerate(frames)
    ]
    return {
        "states": states,
        "diagnostics": diagnostics,
        "runtime_s": runtime,
    }


def run_voltage_wls(network: NetworkModel, prior: np.ndarray, frames: list[FrameMeasurements], window_types: list[str]) -> dict:
    t0 = time.perf_counter()
    model = PmuStateEstimator(network, EstimationConfig(lambda_reg=5e-2, mu_reg=1e-3, current_weight=0.0))
    result = model.estimate_timeseries([_to_frame(f, include_current=False) for f in frames], x_prior=prior)
    runtime = time.perf_counter() - t0
    diagnostics = []
    for i, d in enumerate(result.diagnostics):
        row = asdict(d) if is_dataclass(d) else (dict(d) if isinstance(d, dict) else {})
        row.update({"WINDOW_TYPE": window_types[i]})
        if not row:
            row = {"TIMESTAMP": float(frames[i].timestamp), "FRAME_INDEX": i, "WINDOW_TYPE": window_types[i]}
        diagnostics.append(row)
    return {"states": result.voltage_estimates_pu, "diagnostics": diagnostics, "runtime_s": runtime}


def run_voltage_current_wls(network: NetworkModel, prior: np.ndarray, frames: list[FrameMeasurements], window_types: list[str]) -> dict:
    t0 = time.perf_counter()
    model = PmuStateEstimator(network, EstimationConfig(lambda_reg=5e-2, mu_reg=1e-3, current_weight=0.25))
    result = model.estimate_timeseries([_to_frame(f, include_current=True) for f in frames], x_prior=prior)
    runtime = time.perf_counter() - t0
    diagnostics = []
    for i, d in enumerate(result.diagnostics):
        row = asdict(d) if is_dataclass(d) else (dict(d) if isinstance(d, dict) else {})
        row.update({"WINDOW_TYPE": window_types[i]})
        diagnostics.append(row if row else {"TIMESTAMP": float(frames[i].timestamp), "FRAME_INDEX": i, "WINDOW_TYPE": window_types[i]})
    return {"states": result.voltage_estimates_pu, "diagnostics": diagnostics, "runtime_s": runtime}


def run_adaptive_reg(network: NetworkModel, prior: np.ndarray, frames: list[FrameMeasurements], window_types: list[str]) -> dict:
    t0 = time.perf_counter()
    model = PmuStateEstimator(network, EstimationConfig(lambda_reg=8e-2, mu_reg=2e-3, current_weight=0.25))
    x_prev = prior.copy()
    states = []
    diagnostics = []
    for i, frame in enumerate(frames):
        wt = window_types[i]
        if wt == "event":
            model.config.lambda_reg = 1e-2
            model.config.mu_reg = 5e-4
        elif wt == "missing_data":
            model.config.lambda_reg = 1.2e-1
            model.config.mu_reg = 4e-3
        else:
            model.config.lambda_reg = 8e-2
            model.config.mu_reg = 2e-3
        x_hat, d = model.estimate_frame(frame, x_prev=x_prev if i > 0 else None, x_prior=prior, frame_index=i)
        x_prev = x_hat
        states.append(x_hat)
        row = asdict(d) if is_dataclass(d) else (dict(d) if isinstance(d, dict) else {})
        row["WINDOW_TYPE"] = wt
        diagnostics.append(row)
    runtime = time.perf_counter() - t0
    return {"states": np.vstack(states), "diagnostics": diagnostics, "runtime_s": runtime}


def run_voltage_current_smoothed(network: NetworkModel, prior: np.ndarray, frames: list[FrameMeasurements], window_types: list[str]) -> dict:
    base = run_voltage_current_wls(network=network, prior=prior, frames=frames, window_types=window_types)
    states = np.asarray(base["states"], dtype=complex)
    if len(states) > 2:
        smooth = states.copy()
        smooth[1:-1, :] = (states[:-2, :] + states[1:-1, :] + states[2:, :]) / 3.0
        base["states"] = smooth
    return base


def build_estimator_registry() -> dict[str, EstimatorSpec]:
    """Build deterministic estimator registry for benchmarking."""
    return {
        "PRIOR_ONLY_BASELINE": EstimatorSpec(
            name="PRIOR_ONLY_BASELINE",
            description="No measurement fitting; prior-only state sequence.",
            runner=run_prior_only,
            config={"mode": "prior_only"},
        ),
        "PMU_VOLTAGE_WLS": EstimatorSpec(
            name="PMU_VOLTAGE_WLS",
            description="Regularized WLS using PMU voltages only.",
            runner=run_voltage_wls,
            config={"use_voltage": True, "use_current": False},
        ),
        "PMU_VOLTAGE_CURRENT_WLS": EstimatorSpec(
            name="PMU_VOLTAGE_CURRENT_WLS",
            description="Regularized WLS using PMU voltages and currents.",
            runner=run_voltage_current_wls,
            config={"use_voltage": True, "use_current": True},
        ),
        "PMU_VOLTAGE_CURRENT_ADAPTIVE_REG": EstimatorSpec(
            name="PMU_VOLTAGE_CURRENT_ADAPTIVE_REG",
            description="Voltage+current WLS with window-dependent regularization.",
            runner=run_adaptive_reg,
            config={"adaptive_lambda_mu": True},
        ),
        "PMU_VOLTAGE_CURRENT_SMOOTHED": EstimatorSpec(
            name="PMU_VOLTAGE_CURRENT_SMOOTHED",
            description="Voltage+current WLS with temporal post-smoothing.",
            runner=run_voltage_current_smoothed,
            config={"smoother": "3-point-moving-average"},
        ),
    }
