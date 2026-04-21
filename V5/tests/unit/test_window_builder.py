from __future__ import annotations

import numpy as np
import pandas as pd

from src.detectors.shared.preprocessing.pmu_window_builder import PMUWindowBuilder, PMUWindowConfig


def test_window_builder_length_stride_and_labels() -> None:
    n = 20
    frame = pd.DataFrame(
        {
            "TIMESTAMP": np.arange(n, dtype=float) * 0.1,
            "EVENT": [0] * 10 + [5] * 10,
            "DATA_PRESENT": np.ones(n),
            "F1": np.linspace(0.0, 1.0, n),
            "F2": np.linspace(1.0, 2.0, n),
        }
    )
    builder = PMUWindowBuilder(PMUWindowConfig(window_size=8, stride=4, positive_ratio_threshold=0.25))
    samples = builder.build(frame, feature_columns=["F1", "F2", "DATA_PRESENT"], scenario_id="SIM1", split="train")
    assert len(samples) == 4
    assert samples[0].x.shape == (8, 3)
    assert samples[-1].y_binary == 1

