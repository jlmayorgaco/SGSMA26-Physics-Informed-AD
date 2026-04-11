"""Compute power-flow sensitivity columns J_k = ∂θ_PMU / ∂P_k for each bus k.

Uses finite-difference linearization around the .raw base-case operating point.
J ∈ R^{8 × N}: J[:, k] = voltage-angle perturbation at 8 PMU buses when bus k
injects 1 p.u. of additional real power.

For the T1 cosine-match localizer, only the relative pattern matters (not scale).
"""
from __future__ import annotations

import logging

import numpy as np

from src.grid.load_case import GridCase

log = logging.getLogger(__name__)


def compute_jacobians(grid: GridCase) -> np.ndarray:
    """Compute (8, N) sensitivity matrix from Ybus and base-case voltages.

    Uses the DC power-flow approximation:
      ΔP = B' Δθ  →  Δθ = B'^{-1} ΔP
    where B' is the imaginary part of Ybus with the slack row/col removed.

    Returns J[:, k] = B'^{-1}[pmu_rows, k] (angles in radians per p.u. MW).
    """
    N = len(grid.buses)

    # Identify slack bus (type 3)
    slack_idx = next(
        (i for i, b in enumerate(grid.buses) if b.bus_type == 3), 0
    )

    # B' matrix: imaginary part of Ybus
    B = grid.Ybus.imag.copy()  # (N, N)

    # Remove slack row and column (DC PF formulation)
    non_slack = [i for i in range(N) if i != slack_idx]
    B_red = B[np.ix_(non_slack, non_slack)]  # (N-1, N-1)

    try:
        B_inv = np.linalg.inv(B_red)
    except np.linalg.LinAlgError:
        log.warning("B' singular — using pseudo-inverse")
        B_inv = np.linalg.pinv(B_red)

    # Full (N-1, N) sensitivity: each column k gives ∂θ_reduced / ∂P_k
    # Only the non-slack columns of B_red are present; slack column is zero
    # dTheta_full[reduced_row, k] for non-slack row i and all buses k
    # ∂θ_i / ∂P_k = B_inv[i_r, k_r]  where i_r = position of i in non_slack
    #                                         k_r = position of k in non_slack

    # Map PMU indices to reduced indices
    pmu_red = []
    for pi in grid.pmu_bus_indices:
        if pi in non_slack:
            pmu_red.append(non_slack.index(pi))
        else:
            pmu_red.append(None)

    # Build (8, N) Jacobian
    J = np.zeros((8, N))
    for row_out, pi_red in enumerate(pmu_red):
        if pi_red is None:
            continue  # PMU bus is slack — sensitivity is 0 by definition
        for col_full, k in enumerate(range(N)):
            if k == slack_idx:
                J[row_out, col_full] = 0.0  # slack column
            elif k in non_slack:
                k_red = non_slack.index(k)
                J[row_out, col_full] = B_inv[pi_red, k_red]

    log.info("Jacobian J shape: %s (8 PMU buses × %d network buses)", J.shape, N)
    return J


def bus_sensitivity_columns(J: np.ndarray, grid: GridCase) -> dict[int, np.ndarray]:
    """Return {competition_bus_number: J_k column (length 8)}.

    Only returns entries for buses that have a competition bus number.
    """
    result: dict[int, np.ndarray] = {}
    for col, ext_num in enumerate(grid.ext_bus_order):
        result[ext_num] = J[:, col]
    return result
