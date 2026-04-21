from __future__ import annotations

import numpy as np
import pandas as pd

from src.detectors.domain.models.detection_input import DetectionInput
from src.detectors.physical.physical_change_detector import PhysicalChangeDetector


def test_physical_change_detector_reacts_to_freq_rocof_shift() -> None:
    x = np.zeros((5, 10, 3), dtype=float)
    x[:, :, 0] = 1.0
    x[:, :, 1] = 0.0
    x[:, :, 2] = 0.0
    x[4, :, 1] = np.linspace(0.0, 0.9, 10)  # FREQ change
    x[4, :, 2] = np.linspace(0.0, 1.0, 10)  # ROCOF change
    inputs = DetectionInput(
        scenario_id="s",
        split="test",
        x_windows=x,
        timestamps=np.arange(5, dtype=float),
        feature_names=["BUS10_VA_MAG", "BUS10_FREQ", "BUS10_ROCOF"],
        metadata=pd.DataFrame({"y_binary": [0, 0, 0, 0, 1]}),
    )
    out = PhysicalChangeDetector().detect(inputs)
    assert float(out.p_physical[4]) > float(out.p_physical[0])

