from __future__ import annotations

import numpy as np

from src.calibration.calibration_metrics import wrap_deg


def test_wrap_deg_preserves_range() -> None:
    vals = wrap_deg(np.array([-720, -181, -180, 0, 180, 181, 720], dtype=float))
    assert np.all(vals >= -180.0)
    assert np.all(vals < 180.0)


def test_wrap_deg_handles_boundary_cases() -> None:
    vals = wrap_deg(np.array([-180.0, 180.0], dtype=float))
    assert vals.tolist() == [-180.0, -180.0]


def test_wrapped_delta_symmetry_cases() -> None:
    a = wrap_deg(np.array([179.0 - (-179.0)]))[0]
    b = wrap_deg(np.array([-179.0 - 179.0]))[0]
    assert np.isclose(a, -2.0)
    assert np.isclose(b, 2.0)
