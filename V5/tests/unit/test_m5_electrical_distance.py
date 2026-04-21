from __future__ import annotations

import numpy as np

from src.metadata.electrical_distance import build_electrical_distance_matrix, electrical_distance


def test_electrical_distance_diagonal_zero() -> None:
    z = np.array([[2 + 0j, 1 + 0j], [1 + 0j, 3 + 0j]], dtype=complex)
    assert np.isclose(electrical_distance(z, 0, 0), 0.0)


def test_distance_matrix_symmetric() -> None:
    z = np.array([[2 + 0j, 1 + 0.2j], [1 + 0.2j, 3 + 0j]], dtype=complex)
    d = build_electrical_distance_matrix(z)
    assert np.allclose(d, d.T)


def test_distance_matrix_finite_for_invertible_zbus() -> None:
    z = np.array([[2 + 0j, 1 + 0.2j], [1 + 0.2j, 3 + 0j]], dtype=complex)
    d = build_electrical_distance_matrix(z)
    assert np.all(np.isfinite(d))
