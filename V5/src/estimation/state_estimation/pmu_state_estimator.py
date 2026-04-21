"""Regularized quasi-static PMU-only network state estimator."""

from __future__ import annotations

import logging

import numpy as np

from src.estimation.state_estimation.measurement_model import FrameMeasurements, build_linear_frame
from src.estimation.state_estimation.models import (
    EstimationConfig,
    EstimationDiagnostics,
    EstimationResult,
    NetworkModel,
)
from src.estimation.state_estimation.priors import initial_state_from_pmu_voltage


LOGGER = logging.getLogger(__name__)


class PmuStateEstimator:
    """Estimate full network complex voltages from sparse PMU buses."""

    def __init__(self, network: NetworkModel, config: EstimationConfig) -> None:
        self.network = network
        self.config = config
        self._diagnostics: list[EstimationDiagnostics] = []

    @staticmethod
    def _safe_cond(mat: np.ndarray) -> float:
        if mat.size == 0:
            return 0.0
        try:
            return float(np.linalg.cond(mat))
        except np.linalg.LinAlgError:
            return float("inf")

    def estimate_frame(
        self,
        frame: FrameMeasurements,
        x_prev: np.ndarray | None,
        x_prior: np.ndarray,
        frame_index: int,
    ) -> tuple[np.ndarray, EstimationDiagnostics]:
        """Estimate one frame with linear regularized least squares in complex form."""
        lf = build_linear_frame(frame=frame, network=self.network, config=self.config)
        n = len(self.network.bus_order)
        x_ref = np.asarray(x_prev, dtype=complex) if x_prev is not None else np.asarray(x_prior, dtype=complex)

        if lf.A.shape[0] == 0:
            diag = EstimationDiagnostics(
                timestamp=lf.timestamp,
                frame_index=frame_index,
                measurement_count=0,
                voltage_measurement_count=0,
                current_measurement_count=0,
                residual_norm=0.0,
                matrix_condition_number=0.0,
                used_previous_state=x_prev is not None,
                n_valid_pmus=len(frame.valid_pmu_buses or []),
                n_pmus_used=0,
                pmu_buses_used=[],
                pmu_buses_excluded=sorted(frame.expected_pmu_buses or []),
                dropped_reasons=dict(frame.dropped_reasons or {}),
                solver_status="no_measurements",
                solver_message="No valid PMU measurements for this frame.",
            )
            return x_ref.copy(), diag

        w2 = np.diag(np.asarray(lf.w, dtype=float) ** 2)
        H = lf.A.conj().T @ w2 @ lf.A
        g = lf.A.conj().T @ w2 @ lf.b

        reg = (self.config.lambda_reg + self.config.mu_reg + self.config.condition_guard) * np.eye(n, dtype=complex)
        rhs = g + self.config.lambda_reg * x_ref + self.config.mu_reg * x_prior
        lhs = H + reg
        cond = self._safe_cond(lhs)

        solver_status = "ok"
        solver_message = ""
        try:
            x_hat = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            LOGGER.warning("Ill-conditioned solve at t=%.6f; falling back to lstsq.", lf.timestamp)
            x_hat = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
            solver_status = "lstsq_fallback"
            solver_message = "Direct solve failed, used lstsq fallback."

        resid = lf.A @ x_hat - lf.b
        resid_norm = float(np.sqrt(np.mean(np.abs(resid) ** 2))) if len(resid) > 0 else 0.0
        data_term = float(np.real(np.vdot(resid, w2 @ resid)))
        temporal_term = float(self.config.lambda_reg * np.real(np.vdot(x_hat - x_ref, x_hat - x_ref)))
        loadflow_term = float(self.config.mu_reg * np.real(np.vdot(x_hat - x_prior, x_hat - x_prior)))
        diag = EstimationDiagnostics(
            timestamp=lf.timestamp,
            frame_index=frame_index,
            measurement_count=int(lf.A.shape[0]),
            voltage_measurement_count=lf.n_voltage,
            current_measurement_count=lf.n_current,
            residual_norm=resid_norm,
            matrix_condition_number=cond,
            used_previous_state=x_prev is not None,
            n_valid_pmus=len(frame.valid_pmu_buses or []),
            n_pmus_used=len(lf.pmu_buses_used),
            pmu_buses_used=lf.pmu_buses_used,
            pmu_buses_excluded=lf.pmu_buses_excluded,
            dropped_reasons=lf.dropped_reasons,
            solver_status=solver_status,
            solver_message=solver_message,
            data_term_value=data_term,
            temporal_prior_term_value=temporal_term,
            loadflow_prior_term_value=loadflow_term,
            total_objective_value=data_term + temporal_term + loadflow_term,
        )
        return x_hat, diag

    def estimate_timeseries(self, frames: list[FrameMeasurements], x_prior: np.ndarray) -> EstimationResult:
        """Estimate a complete timestamp sequence."""
        if not frames:
            raise ValueError("No frames provided for estimation.")

        x_prev = initial_state_from_pmu_voltage(
            bus_order=self.network.bus_order,
            prior=np.asarray(x_prior, dtype=complex),
            pmu_frame=frames[0].voltage_by_bus,
        )
        states: list[np.ndarray] = []
        timestamps: list[float] = []
        used_counts: list[int] = []
        self._diagnostics = []

        for idx, frame in enumerate(frames):
            x_hat, diag = self.estimate_frame(frame=frame, x_prev=x_prev if idx > 0 else None, x_prior=x_prior, frame_index=idx)
            states.append(x_hat)
            timestamps.append(float(frame.timestamp))
            used_counts.append(int(frame.data_present_count))
            self._diagnostics.append(diag)
            x_prev = x_hat

        return EstimationResult(
            timestamps=np.asarray(timestamps, dtype=float),
            bus_order=self.network.bus_order,
            voltage_estimates_pu=np.vstack(states),
            diagnostics=self._diagnostics,
            data_present_used=np.asarray(used_counts, dtype=int),
        )

    def get_diagnostics(self) -> list[EstimationDiagnostics]:
        """Return last run diagnostics."""
        return list(self._diagnostics)
