"""Base state-estimator API for the comparative POC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class EstimatorOutput:
    """Common estimator output consumed by detector, localizer, and classifiers.

    Attributes:
        state_estimate: Estimated system state, shape ``(T, n_states)``.
        innovation: Measurement innovation/residual, shape ``(T, n_measurements)``.
        innovation_covariance: Optional innovation covariance stack.  Large runs
            may return ``None`` to avoid materializing gigabytes; small tests use
            the full positive-definite stack.
        normalized_score: Per-frame normalized chi-squared-like anomaly score.
    """

    state_estimate: np.ndarray
    innovation: np.ndarray
    innovation_covariance: np.ndarray | None
    normalized_score: np.ndarray


class StateEstimator(ABC):
    """Abstract base class for swappable physics-informed estimators."""

    @abstractmethod
    def fit(self, normal_baseline: np.ndarray) -> None:
        """Calibrate on known-normal data: centers, covariances, Q/R, x0/P0."""

    @abstractmethod
    def estimate(self, pmu_observations: np.ndarray) -> EstimatorOutput:
        """Return estimated state and innovation diagnostics per frame."""

    @abstractmethod
    def count_parameters(self) -> int:
        """Return trainable plus fixed-but-tunable parameter count."""


def covariance_stack(cov: np.ndarray, n_frames: int, max_entries: int = 2_000_000) -> np.ndarray | None:
    """Repeat a covariance matrix for small runs while protecting full-data memory."""

    entries = n_frames * cov.shape[0] * cov.shape[1]
    if entries > max_entries:
        return None
    return np.broadcast_to(cov[None, :, :], (n_frames, cov.shape[0], cov.shape[1])).copy()


def normalized_chi2(innovation: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Compute mean per-channel squared normalized innovation."""

    safe_scale = np.maximum(scale, 1e-9)
    z = np.where(np.isnan(innovation), 0.0, innovation) / safe_scale[None, :]
    return np.mean(z * z, axis=1)

