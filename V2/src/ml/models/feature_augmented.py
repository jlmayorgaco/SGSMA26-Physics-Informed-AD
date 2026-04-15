"""Sklearn-compatible classifier with an internal feature-engineering layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputClassifier


class FeatureAugmentedClassifier(BaseEstimator, ClassifierMixin):
    """Apply a domain feature layer, impute, then train a compact classifier."""

    def __init__(
        self,
        feature_layer: BaseEstimator,
        seed: int = 0,
        base_model: str = "extratrees",
        n_estimators: int = 180,
        max_depth: int = 14,
        min_samples_leaf: int = 2,
    ) -> None:
        self.feature_layer = feature_layer
        self.seed = int(seed)
        self.base_model = str(base_model)
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray) -> "FeatureAugmentedClassifier":
        self.feature_layer_ = self.feature_layer.fit(X, y)
        X_aug = self.feature_layer_.transform(X)
        self.imputer_ = SimpleImputer(strategy="median")
        X_imp = self.imputer_.fit_transform(X_aug)
        y_arr = np.asarray(y)
        self.is_multioutput_ = y_arr.ndim == 2 and y_arr.shape[1] > 1
        estimator = self._make_estimator()
        if self.is_multioutput_ and not self._supports_native_multioutput():
            estimator = MultiOutputClassifier(estimator, n_jobs=-1)
        self.estimator_ = estimator.fit(X_imp, y_arr)
        self.classes_ = getattr(self.estimator_, "classes_", None)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        X_imp = self.imputer_.transform(self.feature_layer_.transform(X))
        return self.estimator_.predict(X_imp)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray | list[np.ndarray]:
        X_imp = self.imputer_.transform(self.feature_layer_.transform(X))
        if hasattr(self.estimator_, "predict_proba"):
            return self.estimator_.predict_proba(X_imp)
        predictions = np.asarray(self.predict(X))
        if predictions.ndim != 1:
            return []
        classes = np.asarray(self.classes_)
        proba = np.zeros((len(predictions), len(classes)), dtype=float)
        for index, label in enumerate(predictions):
            matches = np.flatnonzero(classes == label)
            if matches.size:
                proba[index, matches[0]] = 1.0
        return proba

    def _make_estimator(self) -> RandomForestClassifier | ExtraTreesClassifier:
        if self.base_model == "randomforest":
            return RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=self.seed,
                n_jobs=-1,
            )
        return ExtraTreesClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_features="sqrt",
            class_weight="balanced",
            random_state=self.seed,
            n_jobs=-1,
        )

    def _supports_native_multioutput(self) -> bool:
        return self.base_model in {"extratrees", "randomforest"}
