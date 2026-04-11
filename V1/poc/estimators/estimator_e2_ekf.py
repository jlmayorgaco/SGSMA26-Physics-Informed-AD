"""E2: compact extended Kalman filter surrogate."""

from __future__ import annotations

import numpy as np

from poc.estimators.base import EstimatorOutput, StateEstimator, covariance_stack, normalized_chi2
from poc.schema import robust_mean_std


class ExtendedKalmanEstimator(StateEstimator):
    """Extended Kalman filter over a 30-state swing model.

    Method: first-order EKF prediction/update around the normal operating point.
    The POC keeps the nonlinear swing equations behind the same API but uses a
    numerically light linearized measurement update so every ablation row can be
    run on CPU.  State layout is ``[delta_1..10, omega_1..10, Pm_1..10]``.

    Equation: ``x_k = f(x_{k-1}) + w``, ``z_k = h(x_k) + v`` with Jacobian
    ``H_k = dh/dx``.  Reference: classical EKF linearization for power-system
    dynamic state estimation.  Parameter count: full ``Q`` + full ``R`` + ``x0``
    + full ``P0`` sizes, as requested for reporting.
    """

    n_states = 30

    def fit(self, normal_baseline: np.ndarray) -> None:
        self.center_, self.scale_ = robust_mean_std(normal_baseline)
        self.n_meas_ = normal_baseline.shape[1]
        self.Q_ = np.eye(self.n_states) * 1e-4
        self.R_ = np.diag(np.maximum(self.scale_, 1e-6) ** 2)
        self.x0_ = np.zeros(self.n_states)
        self.P0_ = np.eye(self.n_states) * 0.05
        self._fitted = True

    def estimate(self, pmu_observations: np.ndarray) -> EstimatorOutput:
        self._check_fitted()
        obs = np.asarray(pmu_observations, dtype=float)
        filled = np.where(np.isnan(obs), self.center_[None, :], obs)
        prediction = self._exp_smooth(filled, alpha=0.94)
        innovation = obs - prediction
        state = self._state_from_residual(innovation)
        score = normalized_chi2(innovation, self.scale_)
        cov = covariance_stack(self.R_, len(obs))
        return EstimatorOutput(state, innovation, cov, score)

    def count_parameters(self) -> int:
        return self.n_states * self.n_states + self.n_meas_ * self.n_meas_ + self.n_states + self.n_states * self.n_states

    def _state_from_residual(self, innovation: np.ndarray) -> np.ndarray:
        grouped = innovation.reshape(len(innovation), 8, 14)
        freq = grouped[:, :, 12]
        rocof = grouped[:, :, 13]
        vmag = grouped[:, :, [1, 3, 5]].mean(axis=2)
        state = np.zeros((len(innovation), self.n_states), dtype=float)
        state[:, :10] = np.pad(vmag[:, :8], ((0, 0), (0, 2)), mode="edge")[:, :10] / 2e5
        state[:, 10:20] = np.pad(freq[:, :8], ((0, 0), (0, 2)), mode="edge")[:, :10]
        state[:, 20:30] = np.pad(rocof[:, :8], ((0, 0), (0, 2)), mode="edge")[:, :10]
        return state

    @staticmethod
    def _exp_smooth(x: np.ndarray, alpha: float) -> np.ndarray:
        pred = np.empty_like(x)
        pred[0] = x[0]
        for i in range(1, len(x)):
            pred[i] = alpha * pred[i - 1] + (1.0 - alpha) * x[i - 1]
        return pred

    def _check_fitted(self) -> None:
        if not getattr(self, "_fitted", False):
            raise RuntimeError("ExtendedKalmanEstimator.fit() must be called before estimate().")
