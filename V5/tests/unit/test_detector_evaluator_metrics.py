from __future__ import annotations

import numpy as np

from src.detectors.training.evaluators.detector_evaluator import BinaryDetectorEvaluator


def test_detector_evaluator_metrics_known_values() -> None:
    y_true = np.array([0, 0, 1, 1], dtype=int)
    y_pred = np.array([0, 1, 1, 0], dtype=int)
    y_score = np.array([0.10, 0.70, 0.80, 0.20], dtype=float)
    ts = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)

    metrics = BinaryDetectorEvaluator().metric_bundle(
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
        timestamps=ts,
        scenario_ids=np.array(["A", "A", "A", "A"]),
    )
    assert metrics["false_positives"] == 1
    assert abs(float(metrics["precision_abnormal"]) - 0.5) < 1e-9
    assert abs(float(metrics["recall_abnormal"]) - 0.5) < 1e-9
    assert abs(float(metrics["f1_abnormal"]) - 0.5) < 1e-9
    assert abs(float(metrics["balanced_accuracy"]) - 0.5) < 1e-9
    assert metrics["support_abnormal"] == 2
    assert metrics["support_normal"] == 2
