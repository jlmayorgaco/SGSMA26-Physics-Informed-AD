from __future__ import annotations

import numpy as np

from poc.estimators.estimator_e2_ekf import ExtendedKalmanEstimator


def test_ekf_shapes_covariance_positive_diagonal():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(40, 112))
    est = ExtendedKalmanEstimator()
    est.fit(x[:20])
    out = est.estimate(x)
    assert out.state_estimate.shape == (40, 30)
    assert out.innovation.shape == x.shape
    assert out.innovation_covariance is not None
    assert np.all(np.diag(out.innovation_covariance[0]) > 0)
    assert est.count_parameters() > 0

