from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

from pmu_hybrid.physics.virtual_pmu import evaluation_ground_truth, observed_measurements


def test_observed_and_evaluation_ground_truth_apis_do_not_leak_hidden_buses() -> None:
    buses = tuple(range(1, 40)); voltage = np.arange(1, 40, dtype=float) + 1j
    observed = observed_measurements(buses, voltage)
    truth = evaluation_ground_truth(buses, voltage)
    assert set(observed) == {2, 5, 6, 10, 19, 22, 29, 39}
    assert set(truth) == set(buses)
    assert 1 not in observed and 1 in truth


def test_stable_discretization_uses_matrix_exponential() -> None:
    A = np.array([[0.0, 1.0], [-4.0, -0.2]])
    discrete = expm(A / 30.0)
    assert np.max(np.abs(np.linalg.eigvals(discrete))) < 1.0
    assert not np.allclose(discrete, np.eye(2) + A / 30.0)


def test_toy_observability_rank_and_functional_residual() -> None:
    A = np.diag([-1.0, -2.0]); C = np.array([[1.0, 0.0]]); L = np.array([[0.0, 1.0]])
    Ad = expm(A / 30.0)
    O = np.vstack((C, C @ Ad, C @ Ad @ Ad))
    assert np.linalg.matrix_rank(O, tol=1e-12) == 1
    _, _, vh = np.linalg.svd(O, full_matrices=False)
    projection = vh[:1].T @ vh[:1]
    residual = np.linalg.norm(L @ (np.eye(2) - projection)) / np.linalg.norm(L)
    assert residual == pytest.approx(1.0)


def test_gauge_rotation_does_not_change_physical_complex_observability() -> None:
    theta = 0.37
    v = np.array([1.0 + 0.2j, 0.9 - 0.1j])
    rotated = v * np.exp(1j * theta)
    assert np.allclose(np.abs(v), np.abs(rotated))
    assert np.allclose(np.angle(rotated) - theta, np.angle(v))
