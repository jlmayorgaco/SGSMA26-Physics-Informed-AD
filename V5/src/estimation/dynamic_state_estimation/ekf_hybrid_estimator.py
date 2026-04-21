"""Hybrid EKF estimator for M8."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.estimation.dynamic_state_estimation.covariance_builder import build_process_covariance
from src.estimation.dynamic_state_estimation.dynamic_measurement_model import (
    build_measurement_matrices,
    voltage_vector_from_state,
)
from src.estimation.dynamic_state_estimation.graph_regularization import build_rectangular_laplacian_penalty
from src.estimation.dynamic_state_estimation.hybrid_state_definition import (
    HybridStateLayout,
    hybrid_state_from_voltage_prior,
)
from src.estimation.dynamic_state_estimation.swing_dynamics import SwingParams, predict_hybrid_state
from src.estimation.state_estimation.measurement_model import FrameMeasurements
from src.estimation.state_estimation.models import NetworkModel


def run_hybrid_ekf(
    network: NetworkModel,
    prior: np.ndarray,
    frames: list[FrameMeasurements],
    window_types: list[str],
    layout: HybridStateLayout,
    graph_laplacian: np.ndarray,
    dt: float = 1.0 / 30.0,
    lambda_reg: float = 5e-2,
    mu_reg: float = 1e-3,
    alpha_graph: float = 3e-2,
    voltage_sigma: float = 0.01,
    current_sigma: float = 0.02,
    swing_params: SwingParams | None = None,
) -> dict[str, Any]:
    """Run reduced-order hybrid EKF."""
    params = swing_params or SwingParams()
    n = layout.total_dim
    n_bus = len(layout.bus_order)
    x_prior = np.asarray(hybrid_state_from_voltage_prior(layout, list(np.asarray(prior, dtype=complex))), dtype=float)
    x = x_prior.copy()
    p = np.eye(n, dtype=float) * 0.05
    q = build_process_covariance(n_state=n, voltage_dim=layout.voltage_dim, voltage_q=2e-4, dynamic_q=7e-4)
    l_rect = build_rectangular_laplacian_penalty(np.asarray(graph_laplacian, dtype=float))
    t0 = time.perf_counter()

    states: list[np.ndarray] = []
    diagnostics: list[dict[str, Any]] = []
    for i, frame in enumerate(frames):
        x_pred = predict_hybrid_state(x=x, layout=layout, dt=dt, params=params)
        f = np.eye(n, dtype=float)
        p_pred = f @ p @ f.T + q

        meas = build_measurement_matrices(
            frame=frame,
            network=network,
            layout=layout,
            x_pred=x_pred,
            voltage_sigma=voltage_sigma,
            current_sigma=current_sigma,
        )
        if meas.H.shape[0] == 0:
            x = x_pred
            p = p_pred
            vhat = voltage_vector_from_state(x, layout)
            states.append(vhat)
            graph_term = float(alpha_graph * np.real(x[: 2 * n_bus].T @ l_rect @ x[: 2 * n_bus]))
            diagnostics.append(
                {
                    "timestamp": float(frame.timestamp),
                    "frame_index": i,
                    "solver_status": "no_measurements",
                    "n_pmus_used": 0,
                    "n_valid_pmus": len(frame.valid_pmu_buses or []),
                    "pmu_buses_used": [],
                    "pmu_buses_excluded": sorted(frame.expected_pmu_buses or []),
                    "residual_norm": 0.0,
                    "data_term_value": 0.0,
                    "temporal_prior_term_value": 0.0,
                    "loadflow_prior_term_value": float(mu_reg * np.sum((x - x_prior) ** 2)),
                    "graph_term_value": graph_term,
                    "total_objective_value": float(mu_reg * np.sum((x - x_prior) ** 2) + graph_term),
                    "window_type": window_types[i],
                }
            )
            continue

        y = meas.z - meas.h
        h = meas.H
        r = meas.R + np.eye(meas.R.shape[0]) * 1e-10
        s = h @ p_pred @ h.T + r
        try:
            k = p_pred @ h.T @ np.linalg.inv(s)
            status = "ok"
        except np.linalg.LinAlgError:
            k = p_pred @ h.T @ np.linalg.pinv(s)
            status = "pinv_fallback"
        x_upd = x_pred + k @ y

        # Soft graph and prior correction in voltage subspace.
        x_volt = x_upd[: 2 * n_bus]
        prior_volt = x_prior[: 2 * n_bus]
        lhs = np.eye(2 * n_bus) + alpha_graph * l_rect + mu_reg * np.eye(2 * n_bus)
        rhs = x_volt + mu_reg * prior_volt
        try:
            x_upd[: 2 * n_bus] = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            x_upd[: 2 * n_bus] = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
            status = "graph_lstsq_fallback"

        i_kh = np.eye(n) - k @ h
        p = i_kh @ p_pred @ i_kh.T + k @ r @ k.T
        x = x_upd
        vhat = voltage_vector_from_state(x, layout)
        states.append(vhat)

        resid = y - h @ (x - x_pred)
        data_term = float(resid.T @ np.linalg.pinv(r) @ resid)
        temporal_term = float(lambda_reg * np.sum((x - x_pred) ** 2))
        loadflow_term = float(mu_reg * np.sum((x - x_prior) ** 2))
        graph_term = float(alpha_graph * np.real(x[: 2 * n_bus].T @ l_rect @ x[: 2 * n_bus]))
        diagnostics.append(
            {
                "timestamp": float(frame.timestamp),
                "frame_index": i,
                "solver_status": status,
                "n_pmus_used": len(meas.pmu_buses_used),
                "n_valid_pmus": len(frame.valid_pmu_buses or []),
                "pmu_buses_used": meas.pmu_buses_used,
                "pmu_buses_excluded": meas.pmu_buses_excluded,
                "residual_norm": float(np.linalg.norm(resid)),
                "data_term_value": data_term,
                "temporal_prior_term_value": temporal_term,
                "loadflow_prior_term_value": loadflow_term,
                "graph_term_value": graph_term,
                "total_objective_value": data_term + temporal_term + loadflow_term + graph_term,
                "window_type": window_types[i],
            }
        )

    return {"states": np.vstack(states), "diagnostics": diagnostics, "runtime_s": float(time.perf_counter() - t0)}

