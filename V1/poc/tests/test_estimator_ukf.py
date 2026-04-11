from __future__ import annotations

import numpy as np

from poc.estimators.estimator_e3_ukf import UnscentedKalmanEstimator


def test_ukf_augmented_state_and_covariance():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(42, 112))
    est = UnscentedKalmanEstimator()
    est.fit(x[:20])
    out = est.estimate(x)
    assert out.state_estimate.shape == (42, 50)
    assert out.innovation.shape == x.shape
    assert out.innovation_covariance is not None
    assert np.all(np.diag(out.innovation_covariance[0]) > 0)
    assert est.count_parameters() > 0

