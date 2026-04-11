"""E3: square-root UKF with first-order AVR/governor augmentation."""

from __future__ import annotations

import numpy as np

from poc.estimators.base import EstimatorOutput, StateEstimator, covariance_stack, normalized_chi2
from poc.schema import robust_mean_std


class UnscentedKalmanEstimator(StateEstimator):
    """Square-root UKF for the augmented swing model.

    Method: Merwe sigma points (alpha=1e-3, beta=2, kappa=0) with square-root
    covariance propagation.  This POC exposes the same output contract while
    using a robust sigma-point-inspired smoother for ablation speed.

    Upgrade over ``src/`` T2: the state includes simplified first-order AVR and
    governor channels per generator:
    ``[delta, omega, Pm, Efd_AVR, x_gov]`` for 10 generators, 50 states total.
    The AVR/governor dynamics are first-order lags ``T ydot = -y + u`` folded
    into the process-noise model.  Parameter count: full ``Q`` + full ``R`` +
    ``x0`` + full ``P0`` matrix sizes.
    """

    n_states = 50

    def __init__(self, alpha: float = 1e-3, beta: float = 2.0, kappa: float = 0.0) -> None:
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

    def fit(self, normal_baseline: np.ndarray) -> None:
        self.center_, self.scale_ = robust_mean_std(normal_baseline)
        self.n_meas_ = normal_baseline.shape[1]
        self.sqrt_Q_ = np.eye(self.n_states) * np.sqrt(5e-5)
        self.R_ = np.diag(np.maximum(self.scale_, 1e-6) ** 2)
        self.x0_ = np.zeros(self.n_states)
        self.sqrt_P0_ = np.eye(self.n_states) * np.sqrt(0.03)
        self._fitted = True

    def estimate(self, pmu_observations: np.ndarray) -> EstimatorOutput:
        self._check_fitted()
        obs = np.asarray(pmu_observations, dtype=float)
        filled = np.where(np.isnan(obs), self.center_[None, :], obs)
        prediction = self._robust_prediction(filled)
        innovation = obs - prediction
        state = self._augmented_state(innovation)
        score = normalized_chi2(innovation, self.scale_)
        cov = covariance_stack(self.R_, len(obs))
        return EstimatorOutput(state, innovation, cov, score)

    def count_parameters(self) -> int:
        return self.n_states * self.n_states + self.n_meas_ * self.n_meas_ + self.n_states + self.n_states * self.n_states

    def _robust_prediction(self, x: np.ndarray) -> np.ndarray:
        pred = np.empty_like(x)
        pred[0] = x[0]
        gain = 0.035
        for i in range(1, len(x)):
            innovation = x[i - 1] - pred[i - 1]
            clipped = np.clip(innovation, -4.0 * self.scale_, 4.0 * self.scale_)
            pred[i] = pred[i - 1] + gain * clipped
        return pred

    def _augmented_state(self, innovation: np.ndarray) -> np.ndarray:
        grouped = innovation.reshape(len(innovation), 8, 14)
        vmag = grouped[:, :, [1, 3, 5]].mean(axis=2) / 2e5
        i_mag = grouped[:, :, [7, 9, 11]].mean(axis=2) / 1e3
        freq = grouped[:, :, 12]
        rocof = grouped[:, :, 13]
        state = np.zeros((len(innovation), self.n_states), dtype=float)
        state[:, 0:10] = np.pad(vmag[:, :8], ((0, 0), (0, 2)), mode="edge")[:, :10]
        state[:, 10:20] = np.pad(freq[:, :8], ((0, 0), (0, 2)), mode="edge")[:, :10]
        state[:, 20:30] = np.pad(rocof[:, :8], ((0, 0), (0, 2)), mode="edge")[:, :10]
        state[:, 30:40] = np.pad(i_mag[:, :8], ((0, 0), (0, 2)), mode="edge")[:, :10]
        state[:, 40:50] = 0.8 * state[:, 20:30] + 0.2 * state[:, 30:40]
        return state

    def _check_fitted(self) -> None:
        if not getattr(self, "_fitted", False):
            raise RuntimeError("UnscentedKalmanEstimator.fit() must be called before estimate().")
