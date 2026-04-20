from __future__ import annotations

import numpy as np

from src.detectors.data.preprocessing import NumpyLogisticModel


def _baseline_features(windows: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            windows.mean(axis=1),
            windows.std(axis=1),
            np.abs(np.diff(windows, axis=1, prepend=windows[:, :1, :])).mean(axis=1),
        ],
        axis=1,
    )


class BaselinePhysicalDetector:
    def __init__(self, learning_rate: float = 0.05, epochs: int = 200, l2: float = 1e-4, seed: int = 12345) -> None:
        self.model = NumpyLogisticModel(learning_rate=learning_rate, epochs=epochs, l2=l2, seed=seed)
        self.constant_probability: float | None = None

    def fit(self, windows: np.ndarray, y: np.ndarray) -> None:
        x = _baseline_features(windows)
        y = y.astype(int)
        if len(np.unique(y)) < 2:
            self.constant_probability = float(y.mean())
            return
        self.constant_probability = None
        self.model.fit(x, y.astype(float))

    def predict(self, windows: np.ndarray) -> np.ndarray:
        x = _baseline_features(windows)
        if self.constant_probability is not None:
            return np.full((len(x),), self.constant_probability, dtype=float)
        return self.model.predict_proba(x)

