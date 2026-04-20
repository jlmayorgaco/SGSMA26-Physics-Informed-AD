"""Temporal-regularized YBUS estimator helpers for m4."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def _legacy_m4():
    workspace_root = Path(__file__).resolve().parents[2]
    root_str = str(workspace_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return import_module("m4_andes_faults_type1")


def solve_unknown_voltages_temporal(
    Yuu: np.ndarray,
    Yuk: np.ndarray,
    Vk: np.ndarray,
    Vu_prev: np.ndarray | None,
    Vu_prior: np.ndarray | None,
    lambda_time: float,
    lambda_prior: float,
) -> np.ndarray:
    A = Yuu
    b = -Yuk @ Vk
    n = A.shape[1]
    H = A.conj().T @ A + (1e-8 + lambda_time + lambda_prior) * np.eye(n, dtype=complex)
    rhs = A.conj().T @ b
    if Vu_prev is not None:
        rhs = rhs + lambda_time * Vu_prev
    if Vu_prior is not None:
        rhs = rhs + lambda_prior * Vu_prior
    try:
        return np.linalg.solve(H, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(H, rhs, rcond=None)[0]


def rolling_complex_mean(x: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(x, dtype=complex)
    if window <= 1 or len(values) == 0:
        return values.copy()
    real = pd.Series(np.real(values)).rolling(window=window, center=True, min_periods=1).mean().to_numpy(float)
    imag = pd.Series(np.imag(values)).rolling(window=window, center=True, min_periods=1).mean().to_numpy(float)
    return real + 1j * imag


def smooth_estimated_voltage_matrix(V_est: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1:
        return np.asarray(V_est, dtype=complex)
    out = np.zeros_like(V_est, dtype=complex)
    for bi in range(V_est.shape[1]):
        out[:, bi] = rolling_complex_mean(V_est[:, bi], window)
    return out


def build_estimated_voltage_matrix_from_pmuses(simulation_dfs: dict[str, pd.DataFrame], meta: dict) -> np.ndarray:
    return _legacy_m4().build_estimated_voltage_matrix_from_pmuses(simulation_dfs, meta)


def build_estimated_bus_dataframes(simulation_dfs: dict[str, pd.DataFrame], meta: dict, fault_bus: str) -> dict[str, pd.DataFrame]:
    return _legacy_m4().build_estimated_bus_dataframes(simulation_dfs, meta, str(fault_bus))
