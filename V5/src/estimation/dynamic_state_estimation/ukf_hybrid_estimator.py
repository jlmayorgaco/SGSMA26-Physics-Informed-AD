"""Hybrid UKF estimator for M8."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.estimation.dynamic_state_estimation.covariance_builder import build_process_covariance
from src.estimation.dynamic_state_estimation.dynamic_measurement_model import (
    build_measurement_matrices,
    measurement_function,
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


def _sigma_points(x: np.ndarray, p: np.ndarray, alpha: float, beta: float, kappa: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(len(x))
    lam = alpha**2 * (n + kappa) - n
    w_m = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)), dtype=float)
    w_c = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)), dtype=float)
    w_m[0] = lam / (n + lam)
    w_c[0] = lam / (n + lam) + (1.0 - alpha**2 + beta)
    jitter = 1e-9 * np.eye(n)
    s = np.linalg.cholesky((n + lam) * (p + jitter))
    pts = [x]
    for i in range(n):
        pts.append(x + s[:, i])
        pts.append(x - s[:, i])
    return np.vstack(pts), w_m, w_c


def run_hybrid_ukf(
    network: NetworkModel,
    prior: np.ndarray,
    frames: list[FrameMeasurements],
    window_types: list[str],
    layout: HybridStateLayout,
    graph_laplacian: np.ndarray,
    dt: float = 1.0 / 30.0,
    alpha_graph: float = 3e-2,
    mu_reg: float = 1e-3,
    voltage_sigma: float = 0.01,
    current_sigma: float = 0.02,
    ukf_alpha: float = 0.3,
    ukf_beta: float = 2.0,
    ukf_kappa: float = 0.0,
    swing_params: SwingParams | None = None,
) -> dict[str, Any]:
    """Run reduced-order hybrid UKF."""
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
        pts, w_m, w_c = _sigma_points(x=x, p=p, alpha=ukf_alpha, beta=ukf_beta, kappa=ukf_kappa)
        pts_pred = np.vstack([predict_hybrid_state(pt, layout=layout, dt=dt, params=params) for pt in pts])
        x_pred = np.sum(pts_pred * w_m[:, None], axis=0)
        p_pred = q.copy()
        for j in range(pts_pred.shape[0]):
            dx = (pts_pred[j] - x_pred).reshape(-1, 1)
            p_pred += w_c[j] * (dx @ dx.T)

        meas0 = build_measurement_matrices(
            frame=frame,
            network=network,
            layout=layout,
            x_pred=x_pred,
            voltage_sigma=voltage_sigma,
            current_sigma=current_sigma,
        )
        if meas0.H.shape[0] == 0:
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

        z_sigma = np.vstack([measurement_function(pt, frame=frame, network=network, layout=layout) for pt in pts_pred])
        z_pred = np.sum(z_sigma * w_m[:, None], axis=0)
        s = meas0.R.copy()
        c = np.zeros((n, z_pred.shape[0]), dtype=float)
        for j in range(z_sigma.shape[0]):
            dz = (z_sigma[j] - z_pred).reshape(-1, 1)
            dx = (pts_pred[j] - x_pred).reshape(-1, 1)
            s += w_c[j] * (dz @ dz.T)
            c += w_c[j] * (dx @ dz.T)
        try:
            k = c @ np.linalg.inv(s)
            status = "ok"
        except np.linalg.LinAlgError:
            k = c @ np.linalg.pinv(s)
            status = "pinv_fallback"

        y = meas0.z - z_pred
        x_upd = x_pred + k @ y
        p_upd = p_pred - k @ s @ k.T

        # Soft graph/prior correction for voltage block.
        xv = x_upd[: 2 * n_bus]
        pv = x_prior[: 2 * n_bus]
        lhs = np.eye(2 * n_bus) + alpha_graph * l_rect + mu_reg * np.eye(2 * n_bus)
        rhs = xv + mu_reg * pv
        try:
            x_upd[: 2 * n_bus] = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            x_upd[: 2 * n_bus] = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
            status = "graph_lstsq_fallback"

        x = x_upd
        p = p_upd
        vhat = voltage_vector_from_state(x, layout)
        states.append(vhat)

        data_term = float(y.T @ np.linalg.pinv(s) @ y)
        temporal_term = float(np.sum((x - x_pred) ** 2))
        loadflow_term = float(mu_reg * np.sum((x - x_prior) ** 2))
        graph_term = float(alpha_graph * np.real(x[: 2 * n_bus].T @ l_rect @ x[: 2 * n_bus]))
        diagnostics.append(
            {
                "timestamp": float(frame.timestamp),
                "frame_index": i,
                "solver_status": status,
                "n_pmus_used": len(meas0.pmu_buses_used),
                "n_valid_pmus": len(frame.valid_pmu_buses or []),
                "pmu_buses_used": meas0.pmu_buses_used,
                "pmu_buses_excluded": meas0.pmu_buses_excluded,
                "residual_norm": float(np.linalg.norm(y)),
                "data_term_value": data_term,
                "temporal_prior_term_value": temporal_term,
                "loadflow_prior_term_value": loadflow_term,
                "graph_term_value": graph_term,
                "total_objective_value": data_term + temporal_term + loadflow_term + graph_term,
                "window_type": window_types[i],
            }
        )

    return {"states": np.vstack(states), "diagnostics": diagnostics, "runtime_s": float(time.perf_counter() - t0)}

