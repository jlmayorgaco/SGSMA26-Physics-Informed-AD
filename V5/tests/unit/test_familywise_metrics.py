from __future__ import annotations

import numpy as np
import pandas as pd

from src.detectors.training.evaluators.detector_evaluator import BinaryDetectorEvaluator


def test_familywise_metrics_grouping() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": np.arange(8, dtype=float),
            "scenario_id": ["S0", "S0", "S1", "S1", "S2", "S2", "S3", "S3"],
            "scenario_family": ["A", "A", "B", "B", "C", "C", "D", "D"],
            "event_coarse": [0, 0, 1, 2, 5, 7, 6, 8],
            "y_true": [0, 0, 1, 1, 1, 1, 1, 1],
            "y_pred_stable": [0, 0, 1, 1, 1, 0, 1, 1],
            "p_abnormal": [0.1, 0.2, 0.8, 0.9, 0.7, 0.4, 0.85, 0.9],
        }
    )
    family = BinaryDetectorEvaluator().familywise_metrics(frame)
    assert int(family["normal"]["windows"]) == 2
    assert int(family["physical_heavy"]["windows"]) == 2
    assert int(family["cyber_heavy"]["windows"]) == 2
    assert int(family["concurrent_heavy"]["windows"]) == 2
    assert "scenario_family" in family
    assert "A" in family["scenario_family"]
