"""Graph-regularized dynamic-prior WLS estimator for M8."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.estimation.dynamic_state_estimation.graph_regularization import graph_penalty_value
from src.estimation.state_estimation.measurement_model import FrameMeasurements, build_linear_frame
from src.estimation.state_estimation.models import EstimationConfig, NetworkModel


def run_graph_regularized_dynamic_wls(
    network: NetworkModel,
    prior: np.ndarray,
    frames: list[FrameMeasurements],
    window_types: list[str],
    graph_laplacian: np.ndarray,
    lambda_reg: float = 5e-2,
    mu_reg: float = 1e-3,
    alpha_graph: float = 3e-2,
    current_weight: float = 0.25,
) -> dict[str, Any]:
    """Run graph-regularized WLS with temporal dynamic prior."""
    n = len(network.bus_order)
    l = np.asarray(graph_laplacian, dtype=float)
    prior_c = np.asarray(prior, dtype=complex)
    x_prev = prior_c.copy()
    states: list[np.ndarray] = []
    diagnostics: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    cfg = EstimationConfig(lambda_reg=lambda_reg, mu_reg=mu_reg, current_weight=current_weight)
    for idx, frame in enumerate(frames):
        lf = build_linear_frame(frame=frame, network=network, config=cfg)
        if lf.A.shape[0] == 0:
            x_hat = x_prev.copy()
            diagnostics.append(
                {
                    "timestamp": float(frame.timestamp),
                    "frame_index": idx,
                    "solver_status": "no_measurements",
                    "n_pmus_used": 0,
                    "n_valid_pmus": len(frame.valid_pmu_buses or []),
                    "pmu_buses_used": [],
                    "pmu_buses_excluded": sorted(frame.expected_pmu_buses or []),
                    "residual_norm": 0.0,
                    "data_term_value": 0.0,
                    "temporal_prior_term_value": 0.0,
                    "loadflow_prior_term_value": 0.0,
                    "graph_term_value": graph_penalty_value(x_prev, l),
                    "total_objective_value": graph_penalty_value(x_prev, l),
                    "window_type": window_types[idx],
                }
            )
            states.append(x_hat)
            x_prev = x_hat
            continue

        w2 = np.diag(np.asarray(lf.w, dtype=float) ** 2)
        h = lf.A.conj().T @ w2 @ lf.A
        g = lf.A.conj().T @ w2 @ lf.b
        reg = (lambda_reg + mu_reg + 1e-9) * np.eye(n, dtype=complex)
        lhs = h + reg + alpha_graph * l.astype(complex)
        rhs = g + lambda_reg * x_prev + mu_reg * prior_c
        try:
            x_hat = np.linalg.solve(lhs, rhs)
            status = "ok"
        except np.linalg.LinAlgError:
            x_hat = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
            status = "lstsq_fallback"

        resid = lf.A @ x_hat - lf.b
        data_term = float(np.real(np.vdot(resid, w2 @ resid)))
        temporal_term = float(lambda_reg * np.real(np.vdot(x_hat - x_prev, x_hat - x_prev)))
        loadflow_term = float(mu_reg * np.real(np.vdot(x_hat - prior_c, x_hat - prior_c)))
        graph_term = float(alpha_graph * graph_penalty_value(x_hat, l))
        diagnostics.append(
            {
                "timestamp": float(frame.timestamp),
                "frame_index": idx,
                "solver_status": status,
                "n_pmus_used": len(lf.pmu_buses_used),
                "n_valid_pmus": len(frame.valid_pmu_buses or []),
                "pmu_buses_used": lf.pmu_buses_used,
                "pmu_buses_excluded": lf.pmu_buses_excluded,
                "residual_norm": float(np.sqrt(np.mean(np.abs(resid) ** 2))),
                "data_term_value": data_term,
                "temporal_prior_term_value": temporal_term,
                "loadflow_prior_term_value": loadflow_term,
                "graph_term_value": graph_term,
                "total_objective_value": data_term + temporal_term + loadflow_term + graph_term,
                "window_type": window_types[idx],
            }
        )
        states.append(x_hat)
        x_prev = x_hat

    return {"states": np.vstack(states), "diagnostics": diagnostics, "runtime_s": float(time.perf_counter() - t0)}

