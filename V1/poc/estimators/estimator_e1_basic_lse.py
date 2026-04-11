"""E1: rectangular linear PMU state estimator."""

from __future__ import annotations

import numpy as np

from poc.estimators.base import EstimatorOutput, StateEstimator, normalized_chi2
from poc.schema import PMU_BUSES, robust_mean_std


class BasicLSEEstimator(StateEstimator):
    """Phadke-style rectangular linear state estimator.

    Method: solve the linear PMU relation ``z = Hx + e`` per frame.  For this
    POC, the observed PMU voltage phasors directly populate rectangular bus
    states ``x = [e_1, f_1, ..., e_39, f_39]`` and unobserved buses use
    pseudo-measurements calibrated from the normal power-flow baseline.  The
    downstream innovation is the calibrated residual ``z - h(x0)``.

    Equation reference: A. G. Phadke and J. S. Thorp, *Synchronized Phasor
    Measurements and Their Applications*, linear PMU state-estimation chapter.
    Parameter count derivation: no learned/tuned coefficients; pseudo-values are
    calibration statistics and the solve matrix is fixed by topology, so this
    baseline reports ``0``.
    """

    n_states = 78

    def fit(self, normal_baseline: np.ndarray) -> None:
        self.center_, self.scale_ = robust_mean_std(normal_baseline)
        self._pseudo_rect_ = np.zeros(self.n_states, dtype=float)
        self._pseudo_rect_ = self._rectangular_state(np.nan_to_num(self.center_[None, :], nan=0.0))[0]
        self._fitted = True

    def estimate(self, pmu_observations: np.ndarray) -> EstimatorOutput:
        self._check_fitted()
        obs = np.asarray(pmu_observations, dtype=float)
        filled = np.where(np.isnan(obs), self.center_[None, :], obs)
        innovation = obs - self.center_[None, :]
        state = self._rectangular_state(filled)
        score = normalized_chi2(innovation, self.scale_)
        return EstimatorOutput(
            state_estimate=state,
            innovation=innovation,
            innovation_covariance=None,
            normalized_score=score,
        )

    def count_parameters(self) -> int:
        return 0

    def _rectangular_state(self, obs: np.ndarray) -> np.ndarray:
        state = np.broadcast_to(self._pseudo_rect_, (len(obs), self.n_states)).copy()
        for pmu_idx, bus in enumerate(PMU_BUSES):
            base = pmu_idx * 14
            angle_deg = obs[:, base + 0]
            mag = obs[:, base + 1]
            out = 2 * (bus - 1)
            angle = np.deg2rad(angle_deg)
            state[:, out] = mag * np.cos(angle)
            state[:, out + 1] = mag * np.sin(angle)
        return state

    def _check_fitted(self) -> None:
        if not getattr(self, "_fitted", False):
            raise RuntimeError("BasicLSEEstimator.fit() must be called before estimate().")
