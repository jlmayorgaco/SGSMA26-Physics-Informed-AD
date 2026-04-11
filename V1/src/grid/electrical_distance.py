"""Compute the electrical distance matrix from Zbus.

d_ij = |Z_ii + Z_jj - 2*Z_ij|  (Eq. from CLAUDE.md §1)
"""
from __future__ import annotations

import numpy as np


def electrical_distance(Zbus: np.ndarray) -> np.ndarray:
    """Return the N×N real-valued electrical distance matrix.

    Args:
        Zbus: (N, N) complex impedance matrix

    Returns:
        D: (N, N) real symmetric matrix where D[i,j] = |Z_ii + Z_jj - 2*Z_ij|
    """
    N = Zbus.shape[0]
    # Extract diagonal as column vector
    diag = np.diag(Zbus).reshape(-1, 1)  # (N, 1)
    # Broadcasting: D[i,j] = Z_ii + Z_jj - 2*Z_ij
    D_complex = diag + diag.T - 2 * Zbus
    D = np.abs(D_complex)
    # Enforce exact symmetry and zero diagonal
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0.0)
    return D


def pmu_submatrix(D: np.ndarray, pmu_indices: list[int]) -> np.ndarray:
    """Extract the 8×8 sub-matrix for PMU buses only."""
    idx = np.array(pmu_indices)
    return D[np.ix_(idx, idx)]
