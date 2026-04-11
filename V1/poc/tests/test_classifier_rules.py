from __future__ import annotations

import numpy as np

from poc.classifiers.classifier_c1_rules import RuleBasedClassifier


def test_rules_identify_dropout_and_fault_patterns():
    windows = np.zeros((2, 90, 112), dtype=float)
    windows[0, :, :14] = np.nan
    windows[1, 40:46, 1::14] = -25_000.0
    windows[1, 40:46, 7::14] = 150.0
    clf = RuleBasedClassifier()
    clf.fit(windows, np.array([5, 1]))
    pred = clf.predict(windows)
    assert pred[0] in (5, 6)
    assert pred[1] == 1
    assert clf.count_parameters() == 0

