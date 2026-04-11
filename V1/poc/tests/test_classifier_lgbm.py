from __future__ import annotations

import numpy as np

from poc.classifiers.classifier_c2_lgbm import LightGBMClassifierPOC


def test_lgbm_predict_proba_shape():
    rng = np.random.default_rng(11)
    x = rng.normal(size=(24, 42))
    y = np.array([0, 1, 2, 3, 4, 5] * 4)
    clf = LightGBMClassifierPOC(n_estimators=10, seed=11)
    clf.fit(x, y)
    pred = clf.predict(x)
    proba = clf.predict_proba(x)
    assert pred.shape == (24,)
    assert proba.shape == (24, 9)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert clf.count_parameters() > 0

