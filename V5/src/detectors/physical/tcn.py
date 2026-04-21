from __future__ import annotations

import numpy as np

from src.detectors.configs import PhysicalModelConfig
from src.detectors.data.preprocessing import NumpyLogisticModel
from src.detectors.physical.temporal_features import build_temporal_feature_matrix


class TemporalConvDetector:
    """Dependency-light temporal model with dilated-conv feature extraction + logistic head."""

    def __init__(self, config: PhysicalModelConfig) -> None:
        self.config = config
        self.feature_dim: int = 0
        self.model = NumpyLogisticModel(
            learning_rate=config.learning_rate,
            l2=config.l2,
            epochs=config.epochs,
            seed=config.random_seed,
        )
        self.constant_probability: float | None = None

    def fit(self, windows: np.ndarray, y: np.ndarray) -> None:
        feats = build_temporal_feature_matrix(windows, self.config.dilations)
        self.feature_dim = feats.shape[1]
        y = y.astype(int)
        if len(np.unique(y)) < 2:
            self.constant_probability = float(y.mean())
            return
        self.constant_probability = None
        self.model.fit(feats, y.astype(float))

    def predict(self, windows: np.ndarray) -> np.ndarray:
        feats = build_temporal_feature_matrix(windows, self.config.dilations)
        if self.constant_probability is not None:
            return np.full((len(feats),), self.constant_probability, dtype=float)
        return self.model.predict_proba(feats)

