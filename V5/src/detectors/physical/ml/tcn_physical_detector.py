from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


class _NumpyLogistic:
    def __init__(self, learning_rate: float, l2: float, epochs: int, seed: int) -> None:
        self.learning_rate = float(learning_rate)
        self.l2 = float(l2)
        self.epochs = int(epochs)
        self.seed = int(seed)
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        n, d = x.shape
        rng = np.random.default_rng(self.seed)
        self.weights = rng.normal(0.0, 0.01, size=d)
        self.bias = 0.0
        yv = y.astype(float)
        for _ in range(self.epochs):
            logits = x @ self.weights + self.bias
            probs = _sigmoid(logits)
            err = probs - yv
            grad_w = (x.T @ err) / max(n, 1) + self.l2 * self.weights
            grad_b = float(err.mean())
            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.weights is None:
            return np.full((len(x),), 0.5, dtype=float)
        return _sigmoid(x @ self.weights + self.bias)


def _dilated_temporal_features(windows: np.ndarray, dilations: tuple[int, ...]) -> np.ndarray:
    # Produces compact TCN-style statistics without requiring deep-learning dependencies.
    n, t, f = windows.shape
    feats = [windows.mean(axis=1), windows.std(axis=1)]
    for d in dilations:
        if d <= 0 or d >= t:
            continue
        diff = windows[:, d:, :] - windows[:, :-d, :]
        feats.append(diff.mean(axis=1))
        feats.append(np.abs(diff).mean(axis=1))
        feats.append(diff.std(axis=1))
    return np.concatenate(feats, axis=1).reshape(n, -1)


@dataclass(slots=True)
class TCNPhysicalDetector:
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    learning_rate: float = 0.05
    l2: float = 1e-4
    epochs: int = 220
    random_seed: int = 12345
    backend: str = "numpy_tcn"
    _model: _NumpyLogistic | None = None
    _constant_probability: float | None = None

    def fit(self, x_windows: np.ndarray, y: np.ndarray) -> None:
        x = _dilated_temporal_features(x_windows, self.dilations)
        labels = y.astype(int)
        if len(np.unique(labels)) < 2:
            self._constant_probability = float(labels.mean())
            self._model = None
            self.backend = "constant"
            return
        self._constant_probability = None
        model = _NumpyLogistic(
            learning_rate=self.learning_rate,
            l2=self.l2,
            epochs=self.epochs,
            seed=self.random_seed,
        )
        model.fit(x, labels)
        self._model = model
        self.backend = "numpy_tcn"

    def predict_proba(self, x_windows: np.ndarray) -> np.ndarray:
        if self._constant_probability is not None:
            return np.full((len(x_windows),), self._constant_probability, dtype=float)
        x = _dilated_temporal_features(x_windows, self.dilations)
        if self._model is None:
            return np.full((len(x),), 0.5, dtype=float)
        return self._model.predict_proba(x)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: Path) -> "TCNPhysicalDetector":
        with path.open("rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, TCNPhysicalDetector):
            raise TypeError(f"Unexpected model at {path}: {type(obj)!r}")
        return obj

