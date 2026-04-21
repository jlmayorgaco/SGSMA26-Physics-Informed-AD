from __future__ import annotations

import numpy as np
import pandas as pd

from src.detectors.cyber.event7_detector import Event7Detector
from src.detectors.domain.models.detection_input import DetectionInput


def _input(x: np.ndarray) -> DetectionInput:
    return DetectionInput(
        scenario_id="s",
        split="test",
        x_windows=x,
        timestamps=np.arange(x.shape[0], dtype=float),
        feature_names=["f0", "f1"],
        metadata=pd.DataFrame({"y_binary": [0] * x.shape[0]}),
    )


def test_event7_detector_detects_spiky_window() -> None:
    normal = np.ones((8, 8, 2), dtype=float) * 0.1
    detector = Event7Detector()
    detector.fit(_input(normal))
    test_x = normal.copy()
    test_x[6, 4, 0] = 10.0
    out = detector.detect(_input(test_x))
    assert len(out.p_event7) == 8
    assert float(out.p_event7[6]) > float(out.p_event7[0])

