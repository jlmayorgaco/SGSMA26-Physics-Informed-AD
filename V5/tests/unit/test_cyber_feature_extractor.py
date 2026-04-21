from __future__ import annotations

import numpy as np
import pandas as pd

from src.detectors.cyber.features.cyber_feature_extractor import CyberFeatureExtractor
from src.detectors.domain.models.detection_input import DetectionInput


def test_cyber_feature_extractor_deterministic() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(0.0, 1.0, size=(6, 12, 4))
    x[:, :, 3] = np.linspace(1.0, 0.2, 12)[None, :]
    inp = DetectionInput(
        scenario_id="S",
        split="train",
        x_windows=x,
        timestamps=np.arange(6, dtype=float) * 0.1,
        feature_names=["F1", "F2", "F3", "DATA_PRESENT"],
        metadata=pd.DataFrame({"y_binary": [0, 0, 1, 1, 1, 0]}),
    )
    ext = CyberFeatureExtractor()
    out1 = ext.extract(inp)
    out2 = ext.extract(inp)
    assert out1.x.shape[0] == 6
    assert out1.x.shape == out2.x.shape
    assert np.allclose(out1.x, out2.x)
    assert len(out1.feature_names) == out1.x.shape[1]

