from __future__ import annotations

import numpy as np

from src.estimation.estimation_metrics import angle_diff_deg, complex_phase_deg, wrap_deg


def test_wrap_deg_range() -> None:
    vals = wrap_deg(np.array([-540.0, -181.0, -180.0, 0.0, 180.0, 540.0]))
    assert np.all(vals >= -180.0)
    assert np.all(vals < 180.0)


def test_complex_phase_deg_expected_output() -> None:
    z = np.array([1 + 0j, 1j, -1 + 0j])
    ph = complex_phase_deg(z)
    assert np.allclose(ph, np.array([0.0, 90.0, -180.0]))


def test_angle_diff_deg_wrapped_subtraction() -> None:
    a = np.array([179.0])
    b = np.array([-179.0])
    diff = angle_diff_deg(a, b)
    assert np.isclose(diff[0], -2.0)
