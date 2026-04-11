from __future__ import annotations

import numpy as np

from poc.estimators.estimator_e4_gnn import GraphNeuralImputerEstimator


def test_gnn_imputer_shapes_and_parameter_cap():
    rng = np.random.default_rng(10)
    x = rng.normal(size=(36, 112))
    est = GraphNeuralImputerEstimator()
    est.fit(x[:18])
    out = est.estimate(x)
    assert out.state_estimate.shape == (36, 39 * 14)
    assert out.innovation.shape == x.shape
    assert est.count_parameters() < 20_000

