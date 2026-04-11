"""Shared chi-squared detector for all POC estimators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DetectionResult:
    """Debounced detector output."""

    alarm: np.ndarray
    onsets: np.ndarray
    offsets: np.ndarray
    threshold: float


class Chi2InnovationDetector:
    """Detector on ``EstimatorOutput.normalized_score``.

    Method: calibrate a robust empirical chi-squared threshold from known-normal
    scores, then debounce onset/offset decisions.  Equation:
    ``eta_t = mean_i ((nu_ti / sigma_i)^2)`` with threshold at a high normal
    quantile.  Parameter count: zero learned parameters.
    """

    def __init__(self, quantile: float = 0.995, k_on: int = 3, k_off: int = 15) -> None:
        self.quantile = quantile
        self.k_on = k_on
        self.k_off = k_off

    def fit(self, baseline_score: np.ndarray) -> None:
        score = np.asarray(baseline_score, dtype=float)
        finite = score[np.isfinite(score)]
        if len(finite) == 0:
            self.threshold_ = 1.0
        else:
            self.threshold_ = float(np.quantile(finite, self.quantile))
            if self.threshold_ <= 0:
                self.threshold_ = float(np.mean(finite) + 6.0 * np.std(finite) + 1e-6)

    def detect(self, score: np.ndarray) -> DetectionResult:
        if not hasattr(self, "threshold_"):
            raise RuntimeError("Chi2InnovationDetector.fit() must be called first.")
        raw = np.asarray(score, dtype=float) > self.threshold_
        alarm = _debounce(raw, self.k_on, self.k_off)
        return DetectionResult(
            alarm=alarm,
            onsets=_edges(alarm, rising=True),
            offsets=_edges(alarm, rising=False),
            threshold=self.threshold_,
        )


def _debounce(raw: np.ndarray, k_on: int, k_off: int) -> np.ndarray:
    alarm = np.zeros(len(raw), dtype=bool)
    active = False
    on_run = off_run = 0
    for i, flag in enumerate(raw.astype(bool)):
        if flag:
            on_run += 1
            off_run = 0
        else:
            off_run += 1
            on_run = 0
        if not active and on_run >= k_on:
            active = True
        elif active and off_run >= k_off:
            active = False
        alarm[i] = active
    return alarm


def _edges(alarm: np.ndarray, *, rising: bool) -> np.ndarray:
    prev = np.concatenate([[False], alarm[:-1]])
    if rising:
        return np.where(alarm & ~prev)[0]
    return np.where(~alarm & prev)[0]

