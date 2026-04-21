"""Graph/electrical-distance regularization helpers."""

from __future__ import annotations

import numpy as np


def build_distance_weights(distance_matrix: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Build symmetric proximity weights from electrical distance."""
    d = np.asarray(distance_matrix, dtype=float)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("Distance matrix must be square.")
    n = d.shape[0]
    w = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            w[i, j] = 1.0 / max(float(d[i, j]), eps)
    w = 0.5 * (w + w.T)
    np.fill_diagonal(w, 0.0)
    return w


def build_graph_laplacian(weights: np.ndarray) -> np.ndarray:
    """Build unnormalized graph Laplacian L = D - W."""
    w = np.asarray(weights, dtype=float)
    if w.ndim != 2 or w.shape[0] != w.shape[1]:
        raise ValueError("Weight matrix must be square.")
    d = np.diag(np.sum(w, axis=1))
    return d - w


def build_rectangular_laplacian_penalty(laplacian: np.ndarray) -> np.ndarray:
    """Build block-diagonal Laplacian for [Vr, Vi] rectangular voltage states."""
    l = np.asarray(laplacian, dtype=float)
    if l.ndim != 2 or l.shape[0] != l.shape[1]:
        raise ValueError("Laplacian must be square.")
    return np.block(
        [
            [l, np.zeros_like(l)],
            [np.zeros_like(l), l],
        ]
    )


def graph_penalty_value(v_complex: np.ndarray, laplacian: np.ndarray) -> float:
    """Compute V^H L V graph smoothness value."""
    v = np.asarray(v_complex, dtype=complex)
    l = np.asarray(laplacian, dtype=float)
    return float(np.real(np.vdot(v, l @ v)))

