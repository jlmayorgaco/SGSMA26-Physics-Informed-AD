from __future__ import annotations

import numpy as np

from src.estimation.dynamic_state_estimation.electrical_distance import build_electrical_distance_from_ybus


def test_electrical_distance_matrix_is_square_and_finite() -> None:
    y = np.asarray([[10 - 20j, -10 + 20j], [-10 + 20j, 10 - 20j]], dtype=complex)
    d, meta = build_electrical_distance_from_ybus(y + np.eye(2) * 1e-3)
    assert d.shape == (2, 2)
    assert np.all(np.isfinite(d))
    assert meta["distance_finite"] is True

