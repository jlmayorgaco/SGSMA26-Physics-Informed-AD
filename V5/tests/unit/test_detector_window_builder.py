from __future__ import annotations

import numpy as np
import pandas as pd

from src.detectors.configs import WindowConfig
from src.detectors.data.window_builder import SlidingWindowBuilder


def test_window_builder_binary_labeling() -> None:
    n = 40
    frame = pd.DataFrame(
        {
            "TIMESTAMP": np.arange(n) * 0.033,
            "EVENT": [0] * 20 + [5] * 20,
            "DATA_PRESENT": np.ones(n),
            "BUS10_VA_MAG": np.linspace(0.0, 1.0, n),
            "BUS10_Freq": np.linspace(0.0, 1.0, n),
        }
    )
    builder = SlidingWindowBuilder(WindowConfig(size=10, stride=5, positive_ratio_threshold=0.2))
    batch = builder.build(frame, feature_columns=["BUS10_VA_MAG", "BUS10_Freq", "DATA_PRESENT"], scenario_id="SIMX")
    assert batch.x.shape[0] > 0
    assert set(np.unique(batch.y)).issubset({0, 1})
    assert int(batch.y[-1]) == 1

