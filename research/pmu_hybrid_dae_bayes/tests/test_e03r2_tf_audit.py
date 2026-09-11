import numpy as np
import pytest

from pmu_hybrid.experiments.e03r2_tf_audit import (
    apply_tf, augmented_step, consistent_dy, slope_estimate,
)


def test_tf_application_and_zero_guard():
    raw = np.array([[2.0, 0.0], [0.0, 6.0]])
    np.testing.assert_allclose(apply_tf(raw, np.array([2.0, 3.0])), [[1.0, 0.0], [0.0, 2.0]])
    with pytest.raises(ValueError):
        apply_tf(raw, np.array([0.0, 3.0]))


def test_simple_state_normalization():
    # T xdot = -k x, so lambda=-k/T (not -k).
    np.testing.assert_allclose(apply_tf(np.array([[-4.0]]), np.array([2.0])), [[-2.0]])


def test_consistent_ic_formula():
    gy = np.array([[2.0]]); gx = np.array([[3.0]]); dx = np.array([0.1])
    np.testing.assert_allclose(consistent_dy(gy, gx, dx), [-0.15])


def test_augmented_exact_step_and_slope():
    # xdot=-x+u, exact unit step at t=1 is 1-exp(-1).
    np.testing.assert_allclose(augmented_step(np.array([[-1.0]]), np.array([1.0]), 1.0), [1-np.exp(-1)], rtol=1e-10)
    assert 1.9 < slope_estimate(np.array([1e-1, 1e-2, 1e-3]), np.array([1e-2, 1e-4, 1e-6])) < 2.1


def test_reduced_b_and_direct_feedthrough_formula():
    tf = np.array([2.0]); fx = np.array([[-1.0]]); fy = np.array([[0.5]]); gy = np.array([[2.0]]); gx = np.array([[1.0]])
    fu = np.array([0.0]); gu = np.array([1.0]); b = (fu - fy @ np.linalg.solve(gy, gu)) / tf
    d = -np.array([[3.0]]) @ np.linalg.solve(gy, gu)
    np.testing.assert_allclose(b, [-0.125]); np.testing.assert_allclose(d, [-1.5])
