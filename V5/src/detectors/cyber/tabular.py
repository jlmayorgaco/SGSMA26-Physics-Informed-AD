from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.detectors.configs import CyberModelConfig
from src.detectors.data.preprocessing import NumpyLogisticModel


def _summary_features(x: np.ndarray) -> np.ndarray:
    # x: [n, t, f]
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    min_v = x.min(axis=1)
    max_v = x.max(axis=1)
    delta = np.diff(x, axis=1, prepend=x[:, :1, :])
    mean_abs_delta = np.abs(delta).mean(axis=1)
    return np.concatenate([mean, std, min_v, max_v, mean_abs_delta], axis=1)


@dataclass(slots=True)
class TabularCyberDetector:
    config: CyberModelConfig
    feature_dim: int = 0
    backend: str = "numpy_logistic"
    _numpy_model: NumpyLogisticModel | None = None
    _lgb_model: object | None = None

    def _fit_lightgbm(self, x: np.ndarray, y: np.ndarray) -> bool:
        if not self.config.prefer_lightgbm:
            return False
        try:
            import lightgbm as lgb  # type: ignore
        except Exception:
            return False
        params = {
            "objective": "binary",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "metric": "binary_logloss",
            "seed": self.config.random_seed,
            "verbosity": -1,
        }
        dtrain = lgb.Dataset(x, label=y)
        self._lgb_model = lgb.train(params=params, train_set=dtrain, num_boost_round=50)
        self.backend = "lightgbm"
        return True

    def fit(self, windows: np.ndarray, y: np.ndarray) -> None:
        x = _summary_features(windows)
        self.feature_dim = x.shape[1]
        y = y.astype(int)
        # No-op model if one class only.
        if len(np.unique(y)) < 2:
            self.backend = "constant"
            self._numpy_model = None
            self._lgb_model = float(y.mean())  # type: ignore[assignment]
            return
        if self._fit_lightgbm(x, y):
            return
        self._numpy_model = NumpyLogisticModel(
            learning_rate=self.config.learning_rate,
            l2=self.config.l2,
            epochs=self.config.epochs,
            seed=self.config.random_seed,
        )
        self._numpy_model.fit(x, y.astype(float))
        self.backend = "numpy_logistic"

    def predict(self, windows: np.ndarray) -> np.ndarray:
        x = _summary_features(windows)
        if self.backend == "constant":
            v = float(self._lgb_model) if isinstance(self._lgb_model, float) else 0.5
            return np.full((len(x),), v, dtype=float)
        if self.backend == "lightgbm" and self._lgb_model is not None:
            out = self._lgb_model.predict(x)  # type: ignore[attr-defined]
            return np.asarray(out, dtype=float)
        if self._numpy_model is None:
            return np.full((len(x),), 0.5, dtype=float)
        return self._numpy_model.predict_proba(x)

