from __future__ import annotations

import numpy as np

from poc.estimators.estimator_e1_basic_lse import BasicLSEEstimator


def _normal_and_fault():
    rng = np.random.default_rng(7)
    normal = rng.normal(0.0, 1.0, size=(180, 112))
    normal[:, 1::14] += 200_000.0
    normal[:, 7::14] += 500.0
    fault = normal.copy()
    fault[90:98, 1::14] -= 20_000.0
    fault[90:98, 7::14] += 120.0
    return normal, fault


def test_lse_shapes_and_zero_params():
    normal, fault = _normal_and_fault()
    est = BasicLSEEstimator()
    est.fit(normal)
    out = est.estimate(fault)
    assert out.state_estimate.shape == (len(fault), 78)
    assert out.innovation.shape == fault.shape
    assert out.innovation_covariance is None
    assert out.normalized_score.shape == (len(fault),)
    assert est.count_parameters() == 0


def test_lse_fault_exceeds_normal_99th_percentile():
    normal, fault = _normal_and_fault()
    est = BasicLSEEstimator()
    est.fit(normal)
    normal_out = est.estimate(normal)
    fault_out = est.estimate(fault)
    assert float(np.max(fault_out.normalized_score[90:98])) > float(np.quantile(normal_out.normalized_score, 0.99))

