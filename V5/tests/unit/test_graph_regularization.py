from __future__ import annotations

import numpy as np

from src.estimation.dynamic_state_estimation.graph_regularization import (
    build_distance_weights,
    build_graph_laplacian,
    build_rectangular_laplacian_penalty,
)


def test_graph_regularization_builds_valid_laplacian() -> None:
    d = np.asarray([[0.0, 2.0, 3.0], [2.0, 0.0, 4.0], [3.0, 4.0, 0.0]])
    w = build_distance_weights(d)
    l = build_graph_laplacian(w)
    rect = build_rectangular_laplacian_penalty(l)
    assert w.shape == (3, 3)
    assert np.allclose(l.sum(axis=1), 0.0, atol=1e-10)
    assert rect.shape == (6, 6)

