from __future__ import annotations

import numpy as np
import pandas as pd

from src.detectors.cyber.features.cyber_feature_extractor import CyberFeatureExtractor
from src.detectors.domain.models.detection_input import DetectionInput


def test_cyber_branch_hardening_features_present() -> None:
    n, t, f = 4, 8, 4
    x = np.random.default_rng(7).normal(size=(n, t, f))
    x[:, :, 1] = np.array([[1,1,1,0,0,1,1,1]] * n)
    x[:, 2:5, 3] = 1.0
    feature_names = ["sig1", "DATA_PRESENT", "sig2", "sig2__is_nan"]
    meta = pd.DataFrame({"y_binary": [0, 1, 1, 0], "scenario_id": ["S"] * n})
    inp = DetectionInput(scenario_id="S", split="train", x_windows=x, timestamps=np.arange(n, dtype=float), feature_names=feature_names, metadata=meta)
    batch = CyberFeatureExtractor().extract(inp)
    needed = {"data_present_transition_rate", "nan_burst_count", "partial_dropout_ratio", "stuck_after_missing"}
    assert needed.issubset(set(batch.feature_names))
    assert batch.x.shape[0] == n
