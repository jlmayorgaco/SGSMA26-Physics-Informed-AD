from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


class _NumpyLogistic:
    def __init__(self, learning_rate: float = 0.05, l2: float = 1e-4, epochs: int = 200, seed: int = 12345) -> None:
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


@dataclass(slots=True)
class CyberGBDTDetector:
    prefer_lightgbm: bool = True
    learning_rate: float = 0.05
    l2: float = 1e-4
    epochs: int = 220
    random_seed: int = 12345
    backend: str = "uninitialized"
    feature_names: list[str] | None = None
    _model: object | None = None
    _constant_probability: float | None = None

    def fit(self, x: np.ndarray, y: np.ndarray, feature_names: list[str] | None = None) -> None:
        self.feature_names = list(feature_names) if feature_names is not None else None
        labels = y.astype(int)
        if len(np.unique(labels)) < 2:
            self._constant_probability = float(labels.mean())
            self.backend = "constant"
            self._model = None
            return
        self._constant_probability = None
        if self.prefer_lightgbm:
            try:
                import lightgbm as lgb  # type: ignore

                params = {
                    "objective": "binary",
                    "learning_rate": self.learning_rate,
                    "num_leaves": 31,
                    "seed": self.random_seed,
                    "verbosity": -1,
                    "metric": "binary_logloss",
                }
                train_set = lgb.Dataset(x, label=labels)
                self._model = lgb.train(params=params, train_set=train_set, num_boost_round=80)
                self.backend = "lightgbm"
                return
            except Exception:
                pass
        model = _NumpyLogistic(
            learning_rate=self.learning_rate,
            l2=self.l2,
            epochs=self.epochs,
            seed=self.random_seed,
        )
        model.fit(x, labels)
        self._model = model
        self.backend = "numpy_logistic"

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self._constant_probability is not None:
            return np.full((len(x),), self._constant_probability, dtype=float)
        if self.backend == "lightgbm" and self._model is not None:
            out = self._model.predict(x)  # type: ignore[attr-defined]
            return np.asarray(out, dtype=float)
        if isinstance(self._model, _NumpyLogistic):
            return self._model.predict_proba(x)
        return np.full((len(x),), 0.5, dtype=float)

    def feature_importance(self) -> dict[str, float]:
        if self.feature_names is None:
            return {}
        if self.backend == "lightgbm" and self._model is not None:
            scores = np.asarray(self._model.feature_importance(importance_type="gain"), dtype=float)  # type: ignore[attr-defined]
            if scores.size == 0:
                return {}
            norm = scores / max(scores.sum(), 1e-9)
            return {name: float(val) for name, val in zip(self.feature_names, norm)}
        if isinstance(self._model, _NumpyLogistic) and self._model.weights is not None:
            w = np.abs(self._model.weights)
            norm = w / max(w.sum(), 1e-9)
            return {name: float(val) for name, val in zip(self.feature_names, norm)}
        return {}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: Path) -> "CyberGBDTDetector":
        with path.open("rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, CyberGBDTDetector):
            raise TypeError(f"Unexpected model at {path}: {type(obj)!r}")
        return obj

