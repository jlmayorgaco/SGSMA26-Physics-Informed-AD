from __future__ import annotations

import numpy as np

from src.detectors.training.evaluators.calibration_evaluator import CalibrationEvaluator


def test_calibration_evaluator_outputs_valid_values() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1], dtype=int)
    y_prob = np.array([0.10, 0.30, 0.45, 0.55, 0.75, 0.90], dtype=float)
    result = CalibrationEvaluator(n_bins=5).evaluate(y_true=y_true, y_prob=y_prob)
    assert len(result.bins) == 5
    assert 0.0 <= result.ece <= 1.0
    assert result.brier >= 0.0
    assert {"bin_start", "bin_end", "count", "mean_pred", "empirical"}.issubset(set(result.bins.columns))
