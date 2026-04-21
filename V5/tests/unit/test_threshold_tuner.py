from __future__ import annotations

import numpy as np

from src.detectors.training.evaluators.threshold_tuner import ThresholdTuner, ThresholdTuningConfig


def test_threshold_tuner_deterministic_selection() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 0, 1], dtype=int)
    p = np.array([0.10, 0.22, 0.30, 0.55, 0.62, 0.75, 0.80, 0.40, 0.90], dtype=float)
    ts = np.arange(len(y_true), dtype=float) * 0.02
    tuner = ThresholdTuner(
        ThresholdTuningConfig(
            thresholds=(0.45, 0.50, 0.55),
            margins=(0.10, 0.15),
            min_on_values=(1, 2),
            min_off_values=(1, 2),
        )
    )
    best_1, sweep_1 = tuner.tune(y_true=y_true, p_abnormal=p, timestamps=ts, scenario_ids=None)
    best_2, sweep_2 = tuner.tune(y_true=y_true, p_abnormal=p, timestamps=ts, scenario_ids=None)
    assert not sweep_1.empty
    assert best_1 == best_2
    assert float(sweep_1.iloc[0]["objective"]) >= float(sweep_1.iloc[-1]["objective"])
    assert sweep_1[["threshold", "start_threshold", "stop_threshold", "margin", "min_on_frames", "min_off_frames"]].equals(
        sweep_2[["threshold", "start_threshold", "stop_threshold", "margin", "min_on_frames", "min_off_frames"]]
    )
