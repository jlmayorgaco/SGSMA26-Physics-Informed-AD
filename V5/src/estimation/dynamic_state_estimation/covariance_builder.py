"""Covariance builders for hybrid dynamic estimators."""

from __future__ import annotations

import numpy as np


def build_process_covariance(
    n_state: int,
    voltage_q: float = 1e-4,
    dynamic_q: float = 5e-4,
    voltage_dim: int | None = None,
) -> np.ndarray:
    """Build diagonal process covariance."""
    q = np.full(int(n_state), float(dynamic_q), dtype=float)
    if voltage_dim is not None:
        q[: int(voltage_dim)] = float(voltage_q)
    return np.diag(q)


def build_measurement_covariance(n_meas: int, base_sigma: float = 0.02) -> np.ndarray:
    """Build diagonal measurement covariance."""
    return np.diag(np.full(int(n_meas), float(base_sigma) ** 2, dtype=float))

