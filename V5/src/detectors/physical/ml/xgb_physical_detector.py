from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


@dataclass(slots=True)
class XGBPhysicalDetector:
    """Lightweight fallback baseline.

    This is a simple logistic baseline over auxiliary features when advanced
    temporal modeling is unavailable.
    """

    learning_rate: float = 0.05
    l2: float = 1e-4
    epochs: int = 180
    random_seed: int = 12345
    weights: np.ndarray | None = None
    bias: float = 0.0
    constant_probability: float | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        labels = y.astype(int)
        if len(np.unique(labels)) < 2:
            self.constant_probability = float(labels.mean())
            self.weights = None
            self.bias = 0.0
            return
        self.constant_probability = None
        n, d = x.shape
        rng = np.random.default_rng(self.random_seed)
        self.weights = rng.normal(0.0, 0.01, size=d)
        self.bias = 0.0
        yv = labels.astype(float)
        for _ in range(self.epochs):
            probs = _sigmoid(x @ self.weights + self.bias)
            err = probs - yv
            grad_w = (x.T @ err) / max(n, 1) + self.l2 * self.weights
            grad_b = float(err.mean())
            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.constant_probability is not None:
            return np.full((len(x),), self.constant_probability, dtype=float)
        if self.weights is None:
            return np.full((len(x),), 0.5, dtype=float)
        return _sigmoid(x @ self.weights + self.bias)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: Path) -> "XGBPhysicalDetector":
        with path.open("rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, XGBPhysicalDetector):
            raise TypeError(f"Unexpected model at {path}: {type(obj)!r}")
        return obj

