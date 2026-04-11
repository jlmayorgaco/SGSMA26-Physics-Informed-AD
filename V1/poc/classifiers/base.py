"""Base classifier API and shared probability helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Classifier(ABC):
    """Abstract nine-label classifier used in the E_i x C_j ablation."""

    @abstractmethod
    def fit(self, features, labels) -> None:
        """Fit classifier state from tabular features or residual windows."""

    @abstractmethod
    def predict(self, features) -> np.ndarray:
        """Return integer event labels."""

    @abstractmethod
    def predict_proba(self, features) -> np.ndarray:
        """Return class probabilities in columns ``0..8``."""

    @abstractmethod
    def count_parameters(self) -> int:
        """Return honest tunable/trainable parameter count."""


def hard_label_proba(labels: np.ndarray, n_classes: int = 9, confidence: float = 0.9) -> np.ndarray:
    """Convert hard labels to smoothed class probabilities."""

    labels = np.asarray(labels, dtype=int)
    proba = np.full((len(labels), n_classes), (1.0 - confidence) / max(n_classes - 1, 1))
    for i, label in enumerate(labels):
        if 0 <= label < n_classes:
            proba[i, label] = confidence
    return proba

