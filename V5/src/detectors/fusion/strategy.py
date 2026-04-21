from __future__ import annotations

import numpy as np
import pandas as pd

from src.detectors.configs import FusionConfig


class WeightedFusionStrategy:
    def __init__(self, config: FusionConfig) -> None:
        self.config = config

    def fuse(self, cyber_probabilities: np.ndarray, physical_probabilities: np.ndarray, metadata: pd.DataFrame) -> np.ndarray:
        cyber_w = self.config.cyber_weight
        physical_w = self.config.physical_weight
        total = max(cyber_w + physical_w, 1e-6)
        fused = (cyber_w * cyber_probabilities + physical_w * physical_probabilities) / total
        if "estimator_difficulty_score" in metadata.columns:
            est = metadata["estimator_difficulty_score"].to_numpy(dtype=float)
            fused = np.clip(fused + self.config.estimator_assist_weight * (est - 0.5), 0.0, 1.0)
        return fused

