"""Electrical distance helpers based on Zbus."""

from __future__ import annotations

import numpy as np


def electrical_distance(zbus: np.ndarray, i: int, j: int) -> float:
    """Compute |Zii + Zjj - 2*Zij| distance."""
    z = np.asarray(zbus, dtype=complex)
    return float(np.abs(z[i, i] + z[j, j] - 2.0 * z[i, j]))


def build_electrical_distance_matrix(zbus: np.ndarray) -> np.ndarray:
    """Build symmetric electrical distance matrix from Zbus."""
    z = np.asarray(zbus, dtype=complex)
    if z.ndim != 2 or z.shape[0] != z.shape[1]:
        raise ValueError("Zbus must be square.")
    n = z.shape[0]
    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            d = electrical_distance(z, i, j)
            dist[i, j] = d
            dist[j, i] = d
    return dist
