from __future__ import annotations

import numpy as np
import pandas as pd

from src.detectors.cyber.event5_detector import Event5Detector
from src.detectors.domain.models.detection_input import DetectionInput


def test_event5_detector_high_dropout_probability() -> None:
    x = np.zeros((4, 6, 3), dtype=float)
    # DATA_PRESENT
    x[:, :, 0] = 1.0
    x[2:, :, 0] = 0.0
    # nan masks
    x[2:, :, 1] = 1.0
    x[2:, :, 2] = 1.0
    inputs = DetectionInput(
        scenario_id="s",
        split="test",
        x_windows=x,
        timestamps=np.arange(4, dtype=float),
        feature_names=["DATA_PRESENT", "f0__is_nan", "f1__is_nan"],
        metadata=pd.DataFrame({"y_binary": [0, 0, 1, 1]}),
    )
    out = Event5Detector().detect(inputs)
    assert len(out.p_event5) == 4
    assert float(out.p_event5[3]) >= 0.9

