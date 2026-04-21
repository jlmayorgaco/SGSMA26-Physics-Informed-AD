from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.detectors.configs import PreprocessingConfig


META_COLUMNS = {
    "TIMESTAMP",
    "EVENT",
    "DATA_PRESENT",
    "SCENARIO_ID",
    "TEMPLATE_NAME",
    "EVENT_COARSE",
    "DIFFICULTY_LEVEL",
    "SCENARIO_FAMILY",
    "SEED_FAMILY",
    "SPLIT",
    "ESTIMATOR_DIFFICULTY_SCORE",
    "DETECTOR_DIFFICULTY_SCORE",
    "CLASSIFIER_DIFFICULTY_SCORE",
    "LOCALIZER_DIFFICULTY_SCORE",
    "OVERALL_TRAINING_VALUE_SCORE",
}


def _numeric_feature_columns(frame: pd.DataFrame) -> list[str]:
    out: list[str] = []
    for col in frame.columns:
        if col in META_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(frame[col]):
            out.append(col)
    return sorted(out)


class SharedPreprocessor:
    def __init__(self, config: PreprocessingConfig) -> None:
        self.config = config
        self.feature_medians: dict[str, float] = {}
        self.feature_scales: dict[str, float] = {}
        self.feature_columns: list[str] = []

    def _clean(self, frame: pd.DataFrame, fit: bool) -> tuple[pd.DataFrame, list[str]]:
        out = frame.copy()
        feature_cols = _numeric_feature_columns(out)
        if "TIMESTAMP" in out.columns:
            out["TIMESTAMP"] = pd.to_numeric(out["TIMESTAMP"], errors="coerce")
        for col in feature_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        if self.config.fill_method == "ffill_bfill":
            out[feature_cols] = out[feature_cols].ffill().bfill()
        out[feature_cols] = out[feature_cols].fillna(0.0)
        if fit:
            self.feature_columns = feature_cols
            self.feature_medians = {c: float(out[c].median()) for c in feature_cols}
            self.feature_scales = {}
            for col in feature_cols:
                q_hi = float(out[col].quantile(self.config.clip_quantile))
                q_lo = float(out[col].quantile(1.0 - self.config.clip_quantile))
                scale = max(abs(q_hi - q_lo), 1e-6)
                self.feature_scales[col] = scale
        for col in feature_cols:
            median = self.feature_medians.get(col, 0.0)
            scale = self.feature_scales.get(col, 1.0)
            out[col] = out[col].clip(median - 5.0 * scale, median + 5.0 * scale)
            out[col] = (out[col] - median) / max(scale, 1e-6)
            if not np.isfinite(out[col]).all():
                out[col] = out[col].replace([np.inf, -np.inf], 0.0).fillna(0.0)
        if "EVENT" in out.columns:
            out["EVENT"] = pd.to_numeric(out["EVENT"], errors="coerce").fillna(0).astype(int)
        if "DATA_PRESENT" in out.columns:
            out["DATA_PRESENT"] = pd.to_numeric(out["DATA_PRESENT"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        return out, feature_cols

    def fit_transform(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        return self._clean(frame, fit=True)

    def transform(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        out, feature_cols = self._clean(frame, fit=False)
        if self.feature_columns:
            missing = [c for c in self.feature_columns if c not in out.columns]
            for col in missing:
                out[col] = 0.0
            feature_cols = list(self.feature_columns)
        return out, feature_cols


def safe_sigmoid(x: np.ndarray) -> np.ndarray:
    x_clip = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def binary_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    eps = 1e-9
    y_prob = np.clip(y_prob, eps, 1.0 - eps)
    return float(-(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob)).mean())


class NumpyLogisticModel:
    def __init__(self, learning_rate: float = 0.05, l2: float = 1e-4, epochs: int = 200, seed: int = 12345) -> None:
        self.learning_rate = learning_rate
        self.l2 = l2
        self.epochs = epochs
        self.seed = seed
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        n, d = x.shape
        rng = np.random.default_rng(self.seed)
        self.weights = rng.normal(0.0, 0.01, size=d)
        self.bias = 0.0
        y = y.astype(float)
        for _ in range(self.epochs):
            logits = x @ self.weights + self.bias
            probs = safe_sigmoid(logits)
            err = probs - y
            grad_w = (x.T @ err) / max(n, 1) + self.l2 * self.weights
            grad_b = float(err.mean())
            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.weights is None:
            return np.full(shape=(len(x),), fill_value=0.5, dtype=float)
        logits = x @ self.weights + self.bias
        return safe_sigmoid(logits)

