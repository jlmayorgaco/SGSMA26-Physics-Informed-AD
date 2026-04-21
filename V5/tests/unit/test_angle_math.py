from __future__ import annotations

import numpy as np

from src.signals.transforms import angle_diff_deg, wrap_deg


def test_wrap_deg_maps_to_minus180_180() -> None:
    values = np.array([-540.0, -181.0, -180.0, -179.0, 179.0, 180.0, 540.0])
    wrapped = wrap_deg(values)
    assert np.all(wrapped >= -180.0)
    assert np.all(wrapped < 180.0)
    assert wrapped.tolist() == [-180.0, 179.0, -180.0, -179.0, 179.0, -180.0, -180.0]


def test_angle_diff_handles_wraparound() -> None:
    diff = angle_diff_deg(np.array([179.0]), np.array([-179.0]))
    assert np.isclose(diff[0], -2.0)
