from __future__ import annotations

import numpy as np

from src.detectors.configs import PhysicalModelConfig
from src.detectors.models import BranchScores, WindowedBatch
from src.detectors.physical.baseline import BaselinePhysicalDetector
from src.detectors.physical.tcn import TemporalConvDetector


class PhysicalTemporalBranch:
    def __init__(self, config: PhysicalModelConfig) -> None:
        self.config = config
        self.primary = TemporalConvDetector(config)
        self.fallback = BaselinePhysicalDetector(
            learning_rate=config.learning_rate,
            epochs=max(100, config.epochs // 2),
            l2=config.l2,
            seed=config.random_seed + 17,
        )
        self.backend = "tcn"

    def fit(self, batch: WindowedBatch) -> None:
        if batch.x.shape[0] < 8 and self.config.use_baseline_fallback:
            self.backend = "baseline"
            self.fallback.fit(batch.x, batch.y)
            return
        self.backend = "tcn"
        self.primary.fit(batch.x, batch.y)

    def predict(self, batch: WindowedBatch) -> BranchScores:
        probs = self.fallback.predict(batch.x) if self.backend == "baseline" else self.primary.predict(batch.x)
        details = {"physical_backend": self.backend, "mean_probability": float(probs.mean()) if len(probs) else 0.0}
        return BranchScores(probabilities=np.clip(probs, 0.0, 1.0), details=details)

