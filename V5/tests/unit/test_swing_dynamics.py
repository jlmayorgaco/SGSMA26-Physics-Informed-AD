from __future__ import annotations

import numpy as np

from src.estimation.dynamic_state_estimation.hybrid_state_definition import build_hybrid_state_layout
from src.estimation.dynamic_state_estimation.swing_dynamics import SwingParams, predict_hybrid_state, step_swing


def test_swing_step_returns_same_shape() -> None:
    d = np.asarray([0.1, -0.05])
    w = np.asarray([0.02, -0.01])
    d1, w1 = step_swing(d, w, dt=0.033, params=SwingParams())
    assert d1.shape == d.shape
    assert w1.shape == w.shape


def test_predict_hybrid_state_shape_consistent() -> None:
    layout = build_hybrid_state_layout(["BUS1", "BUS2", "BUS3"], ["BUS1"])
    x = np.zeros(layout.total_dim, dtype=float)
    x[layout.vr_offset + 0] = 1.0
    x1 = predict_hybrid_state(x, layout=layout, dt=0.033, params=SwingParams())
    assert x1.shape == x.shape

