from __future__ import annotations

import numpy as np

from poc.classifiers.classifier_c3_tcn import TemporalConvNetClassifier


def test_tcn_api_and_parameter_target():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(18, 90, 112))
    y = np.array([1, 2, 3] * 6)
    clf = TemporalConvNetClassifier(seed=12)
    clf.fit(x, y)
    assert clf.predict(x).shape == (18,)
    assert clf.predict_proba(x).shape == (18, 9)
    assert 10_000 <= clf.count_parameters() <= 15_000

