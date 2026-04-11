"""C2: LightGBM classifier over hand-crafted residual features."""

from __future__ import annotations

import numpy as np

from poc.classifiers.base import Classifier, hard_label_proba
from poc.features import ensure_tabular


class LightGBMClassifierPOC(Classifier):
    """LightGBM nine-class classifier for 30-50 residual features.

    Method: gradient-boosted decision trees with ``num_leaves=31``,
    ``n_estimators=200``, ``min_data_in_leaf=3`` and balanced class weights.
    Reference: Ke et al., LightGBM.  Parameter count approximation:
    ``n_trees * avg_leaves * 2`` for split thresholds plus leaf values.
    """

    def __init__(
        self,
        num_leaves: int = 31,
        n_estimators: int = 200,
        min_data_in_leaf: int = 3,
        learning_rate: float = 0.05,
        seed: int = 42,
    ) -> None:
        self.num_leaves = num_leaves
        self.n_estimators = n_estimators
        self.min_data_in_leaf = min_data_in_leaf
        self.learning_rate = learning_rate
        self.seed = seed

    def fit(self, features, labels) -> None:
        x = ensure_tabular(features)
        y = np.asarray(labels, dtype=int)
        self.classes_ = np.array(sorted(set(y.tolist())), dtype=int)
        if len(self.classes_) < 2 or len(y) < 8:
            self.model_ = _CentroidFallback().fit(x, y)
            self._fallback = True
            return
        try:
            from lightgbm import LGBMClassifier

            self.model_ = LGBMClassifier(
                objective="multiclass",
                num_class=9,
                num_leaves=self.num_leaves,
                n_estimators=self.n_estimators,
                min_data_in_leaf=self.min_data_in_leaf,
                learning_rate=self.learning_rate,
                class_weight="balanced",
                random_state=self.seed,
                n_jobs=-1,
                verbose=-1,
            )
            self.model_.fit(x, y)
            self._fallback = False
        except Exception:
            self.model_ = _CentroidFallback().fit(x, y)
            self._fallback = True

    def predict(self, features) -> np.ndarray:
        x = ensure_tabular(features)
        return np.asarray(self.model_.predict(x), dtype=int)

    def predict_proba(self, features) -> np.ndarray:
        x = ensure_tabular(features)
        if hasattr(self.model_, "predict_proba"):
            raw = np.asarray(self.model_.predict_proba(x), dtype=float)
            if raw.shape[1] == 9:
                return raw
            out = np.zeros((len(x), 9), dtype=float)
            classes = getattr(self.model_, "classes_", np.arange(raw.shape[1]))
            for j, cls in enumerate(classes):
                if 0 <= int(cls) < 9:
                    out[:, int(cls)] = raw[:, j]
            row_sum = np.maximum(out.sum(axis=1), 1e-12)
            return out / row_sum[:, None]
        return hard_label_proba(self.predict(x))

    def count_parameters(self) -> int:
        if getattr(self, "_fallback", False):
            return int(len(getattr(self.model_, "classes_", [])) * getattr(self.model_, "n_features_", 1))
        try:
            n_trees = self.model_.booster_.num_trees()
        except Exception:
            n_trees = self.n_estimators
        return int(n_trees * self.num_leaves * 2)


class _CentroidFallback:
    def fit(self, x: np.ndarray, y: np.ndarray):
        self.classes_ = np.array(sorted(set(y.tolist())), dtype=int)
        self.n_features_ = x.shape[1]
        self.centroids_ = np.vstack([np.nanmean(x[y == cls], axis=0) for cls in self.classes_])
        self.centroids_ = np.where(np.isfinite(self.centroids_), self.centroids_, 0.0)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        dist = ((x[:, None, :] - self.centroids_[None, :, :]) ** 2).mean(axis=2)
        return self.classes_[np.argmin(dist, axis=1)]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        dist = ((x[:, None, :] - self.centroids_[None, :, :]) ** 2).mean(axis=2)
        score = np.exp(-dist / (np.nanmedian(dist) + 1e-9))
        score = score / np.maximum(score.sum(axis=1, keepdims=True), 1e-12)
        out = np.zeros((len(x), 9), dtype=float)
        for j, cls in enumerate(self.classes_):
            out[:, int(cls)] = score[:, j]
        return out

