from __future__ import annotations

import numpy as np

from src.detectors.training.evaluators.threshold_tuner import ThresholdTuner


def test_threshold_tuner_v2_runs() -> None:
    y = np.array([0, 0, 1, 1, 1, 0], dtype=int)
    p = np.array([0.1, 0.2, 0.7, 0.8, 0.9, 0.4], dtype=float)
    p_c = np.array([0.2, 0.2, 0.85, 0.88, 0.89, 0.3], dtype=float)
    p_p = np.array([0.1, 0.1, 0.2, 0.25, 0.3, 0.1], dtype=float)
    ts = np.arange(len(y), dtype=float)
    best, sweep = ThresholdTuner().tune(
        y_true=y,
        p_abnormal=p,
        p_cyber=p_c,
        p_physical=p_p,
        frame_predictions=np.array([0, 0, 1, 1, 1, 0], dtype=int),
        timestamps=ts,
        scenario_ids=None,
    )
    assert not sweep.empty
    assert "objective" in sweep.columns
    assert "threshold" in best
    assert "quiet_cyber_max" in best
