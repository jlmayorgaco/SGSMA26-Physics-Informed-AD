from __future__ import annotations

import numpy as np

from src.estimation.temporal_regularized_estimator import smooth_estimated_voltage_matrix, solve_unknown_voltages_temporal
from src.estimation.ybus_estimator import compute_estimated_currents_from_voltage, solve_unknown_voltages_static


def test_solve_unknown_voltages_static_returns_finite_result() -> None:
    Yuu = np.array([[2 + 0j]])
    Yuk = np.array([[1 + 0j]])
    Vk = np.array([1 + 0j])
    Vu = solve_unknown_voltages_static(Yuu, Yuk, Vk)
    assert Vu.shape == (1,)
    assert np.all(np.isfinite(np.real(Vu)))


def test_solve_unknown_voltages_temporal_returns_finite_result() -> None:
    Yuu = np.array([[2 + 0j]])
    Yuk = np.array([[1 + 0j]])
    Vk = np.array([1 + 0j])
    Vu = solve_unknown_voltages_temporal(Yuu, Yuk, Vk, np.array([0.5 + 0j]), np.array([0.4 + 0j]), 5e-2, 1e-3)
    assert Vu.shape == (1,)
    assert np.isfinite(np.real(Vu[0]))


def test_smoothing_preserves_shape() -> None:
    mat = np.ones((10, 3), dtype=complex)
    sm = smooth_estimated_voltage_matrix(mat, window=5)
    assert sm.shape == mat.shape


def test_compute_estimated_currents_from_voltage_preserves_shape() -> None:
    v = np.ones((4, 2), dtype=complex)
    y = np.eye(2, dtype=complex)
    i = compute_estimated_currents_from_voltage(v, y)
    assert i.shape == v.shape
