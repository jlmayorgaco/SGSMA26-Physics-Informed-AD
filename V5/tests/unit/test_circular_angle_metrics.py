from __future__ import annotations

import numpy as np

from src.simulation.m9.hardening import _circular_mean_deg, _circular_variance, _wrap_deg


def test_circular_metrics_basic() -> None:
    x = np.array([179.0, -179.0, 180.0, -180.0])
    m = _circular_mean_deg(x)
    v = _circular_variance(x)
    assert abs(_wrap_deg(np.array([m]))[0]) <= 180.0
    assert 0.0 <= v <= 1.0

