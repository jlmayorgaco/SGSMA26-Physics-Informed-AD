from __future__ import annotations

import numpy as np

from src.estimation.state_estimation.positive_sequence import positive_sequence_from_mag_angle


def test_positive_sequence_balanced_case() -> None:
    v1 = positive_sequence_from_mag_angle(
        np.array([100.0]),
        np.array([0.0]),
        np.array([100.0]),
        np.array([-120.0]),
        np.array([100.0]),
        np.array([120.0]),
    )[0]
    assert np.isclose(np.abs(v1), 100.0, rtol=1e-6)
    assert np.isclose(np.rad2deg(np.angle(v1)), 0.0, atol=1e-6)

