"""Diagnostics for a physics-constrained graph/descriptor estimator.

This module implements the computable linear layer of the proposed method.  It
does not claim to be the final nonlinear ANDES DAE estimator.  Instead it
provides three pieces that can be checked independently:

* a tangent-space measurement decomposition for linearized constraints;
* a finite-horizon observability decomposition, which distinguishes an
  instantaneous blind direction from a dynamically observable one; and
* a steady-state Riccati diagnostic for a graph-coupled swing model.

The routines are deliberately generic so that the simplified graph constraint
used by the current benchmark can later be replaced by Jacobians exported from
the complete ANDES differential-algebraic model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy import linalg

from src.estimation.transient_estimators import graph_laplacian


def _rank_tolerance(singular_values: np.ndarray, shape: tuple[int, int], rtol: float | None) -> float:
    if singular_values.size == 0:
        return 0.0
    if rtol is not None:
        if rtol < 0:
            raise ValueError("rtol must be non-negative")
        return float(rtol * singular_values[0])
    return float(max(shape) * np.finfo(float).eps * singular_values[0])


def orthonormal_nullspace(matrix: np.ndarray, rtol: float | None = None) -> np.ndarray:
    """Return an orthonormal basis for the right nullspace of ``matrix``."""

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    rows, columns = values.shape
    if columns == 0:
        return np.empty((0, 0), dtype=float)
    if rows == 0:
        return np.eye(columns, dtype=float)
    _, singular, right = np.linalg.svd(values, full_matrices=True)
    tolerance = _rank_tolerance(singular, values.shape, rtol)
    rank = int(np.sum(singular > tolerance))
    return right[rank:].T.copy()


def _whiten_measurements(matrix: np.ndarray, covariance: np.ndarray | None) -> np.ndarray:
    if covariance is None:
        return np.asarray(matrix, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    if cov.shape != (matrix.shape[0], matrix.shape[0]):
        raise ValueError("measurement_covariance has incompatible dimensions")
    cov = 0.5 * (cov + cov.T)
    try:
        lower = linalg.cholesky(cov, lower=True, check_finite=True)
    except linalg.LinAlgError as exc:
        raise ValueError("measurement_covariance must be positive definite") from exc
    return linalg.solve_triangular(lower, matrix, lower=True, check_finite=True)


@dataclass(frozen=True)
class TangentObservabilityResult:
    """Instantaneous information decomposition on a constraint tangent space."""

    tangent_basis: np.ndarray
    effective_measurement: np.ndarray
    singular_values: np.ndarray
    observable_basis_physical: np.ndarray
    instantaneously_uninformed_basis_physical: np.ndarray
    information_matrix_tangent: np.ndarray
    constraint_rank: int
    tangent_dimension: int
    instantaneous_rank: int

    @property
    def ambient_dimension(self) -> int:
        return int(self.tangent_basis.shape[0])

    @property
    def instantaneously_uninformed_dimension(self) -> int:
        return int(self.tangent_dimension - self.instantaneous_rank)

    def summary(self) -> dict[str, Any]:
        nonzero = self.singular_values[: self.instantaneous_rank]
        condition = float(nonzero[0] / nonzero[-1]) if nonzero.size else float("inf")
        return {
            "ambient_dimension": self.ambient_dimension,
            "constraint_rank": int(self.constraint_rank),
            "tangent_dimension": int(self.tangent_dimension),
            "measurement_dimension": int(self.effective_measurement.shape[0]),
            "instantaneous_rank": int(self.instantaneous_rank),
            "instantaneously_uninformed_dimension": self.instantaneously_uninformed_dimension,
            "effective_measurement_condition_nonzero": condition,
            "information_symmetry_residual_norm": float(
                np.linalg.norm(self.information_matrix_tangent - self.information_matrix_tangent.T)
            ),
        }


def physical_tangent_observability(
    constraint_jacobian: np.ndarray,
    measurement_jacobian: np.ndarray,
    measurement_covariance: np.ndarray | None = None,
    rtol: float | None = None,
) -> TangentObservabilityResult:
    """Decompose PMU information after restricting updates to ``ker(G)``.

    ``constraint_jacobian`` is the Jacobian of ``g`` with respect to ``q`` and
    ``measurement_jacobian`` is the Jacobian of ``h`` with respect to ``q``.
    If ``N`` is
    an orthonormal basis for ``ker(G)``, the effective measurement is ``H N``.
    Its right singular vectors separate directions informed by the *current*
    measurement from directions that must be resolved by the prior or future
    dynamics.  The latter are not necessarily unobservable over time.
    """

    constraint = np.asarray(constraint_jacobian, dtype=float)
    measurement = np.asarray(measurement_jacobian, dtype=float)
    if constraint.ndim != 2 or measurement.ndim != 2:
        raise ValueError("Jacobians must be two-dimensional")
    if constraint.shape[1] != measurement.shape[1]:
        raise ValueError("constraint and measurement Jacobians must share the state dimension")

    tangent = orthonormal_nullspace(constraint, rtol=rtol)
    effective = measurement @ tangent
    whitened = _whiten_measurements(effective, measurement_covariance)
    _, singular, right = np.linalg.svd(whitened, full_matrices=True)
    tolerance = _rank_tolerance(singular, whitened.shape, rtol)
    rank = int(np.sum(singular > tolerance))
    observable = tangent @ right[:rank].T
    uninformed = tangent @ right[rank:].T
    information = whitened.T @ whitened
    constraint_rank = int(constraint.shape[1] - tangent.shape[1])
    return TangentObservabilityResult(
        tangent_basis=tangent,
        effective_measurement=effective,
        singular_values=singular,
        observable_basis_physical=observable,
        instantaneously_uninformed_basis_physical=uninformed,
        information_matrix_tangent=information,
        constraint_rank=constraint_rank,
        tangent_dimension=int(tangent.shape[1]),
        instantaneous_rank=rank,
    )


@dataclass(frozen=True)
class TemporalObservabilityResult:
    observability_matrix: np.ndarray
    singular_values: np.ndarray
    observable_basis: np.ndarray
    unobservable_basis: np.ndarray
    rank: int
    horizon_steps: int

    def summary(self) -> dict[str, Any]:
        nonzero = self.singular_values[: self.rank]
        return {
            "state_dimension": int(self.observability_matrix.shape[1]),
            "horizon_steps": int(self.horizon_steps),
            "temporal_rank": int(self.rank),
            "temporally_unobservable_dimension": int(self.unobservable_basis.shape[1]),
            "condition_number_nonzero": (
                float(nonzero[0] / nonzero[-1]) if nonzero.size else float("inf")
            ),
            "smallest_nonzero_singular_value": float(nonzero[-1]) if nonzero.size else 0.0,
        }


def temporal_observability(
    transition: np.ndarray,
    measurement_jacobian: np.ndarray,
    horizon_steps: int | None = None,
    rtol: float | None = None,
) -> TemporalObservabilityResult:
    """Return the finite-horizon observability decomposition of ``(A, H)``."""

    a = np.asarray(transition, dtype=float)
    h = np.asarray(measurement_jacobian, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("transition must be square")
    if h.ndim != 2 or h.shape[1] != a.shape[0]:
        raise ValueError("measurement_jacobian has incompatible dimensions")
    steps = int(horizon_steps or a.shape[0])
    if steps <= 0:
        raise ValueError("horizon_steps must be positive")
    blocks: list[np.ndarray] = []
    power = np.eye(a.shape[0], dtype=float)
    for _ in range(steps):
        blocks.append(h @ power)
        power = power @ a
    matrix = np.vstack(blocks)
    _, singular, right = np.linalg.svd(matrix, full_matrices=True)
    tolerance = _rank_tolerance(singular, matrix.shape, rtol)
    rank = int(np.sum(singular > tolerance))
    return TemporalObservabilityResult(
        observability_matrix=matrix,
        singular_values=singular,
        observable_basis=right[:rank].T.copy(),
        unobservable_basis=right[rank:].T.copy(),
        rank=rank,
        horizon_steps=steps,
    )


def graph_balance_linearization(
    ybus: np.ndarray,
    observed_indices: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Build a transparent graph-DAE surrogate for tangent diagnostics.

    The local state is ``q = [delta, frequency, injection]`` and the algebraic
    constraint is ``L delta - injection = 0``.  PMUs select angle and frequency.
    This surrogate is a diagnostic only; it must be replaced by the complete AC
    power-flow Jacobian before claiming a full nonlinear DAE implementation.
    """

    laplacian = graph_laplacian(np.asarray(ybus, dtype=complex), normalized=True)
    n_bus = laplacian.shape[0]
    observed = np.asarray(observed_indices, dtype=int)
    if observed.ndim != 1 or np.any(observed < 0) or np.any(observed >= n_bus):
        raise ValueError("observed_indices contains an invalid bus index")
    constraint = np.hstack([laplacian, np.zeros_like(laplacian), -np.eye(n_bus)])
    measurement = np.zeros((2 * len(observed), 3 * n_bus), dtype=float)
    measurement[np.arange(len(observed)), observed] = 1.0
    measurement[len(observed) + np.arange(len(observed)), n_bus + observed] = 1.0
    return constraint, measurement


@dataclass(frozen=True)
class ModalRiccatiResult:
    laplacian_eigenvalues: np.ndarray
    modes: np.ndarray
    transition: np.ndarray
    measurement_jacobian: np.ndarray
    predicted_covariance: np.ndarray
    posterior_covariance: np.ndarray
    modal_angle_information: np.ndarray
    modal_frequency_information: np.ndarray
    modal_angle_variance: np.ndarray
    modal_frequency_variance: np.ndarray
    modal_variance_reduction: np.ndarray
    bus_angle_variance: np.ndarray
    bus_frequency_variance: np.ndarray
    bus_variance_reduction: np.ndarray
    bus_risk: np.ndarray

    def summary(self) -> dict[str, Any]:
        return {
            "bus_count": int(len(self.laplacian_eigenvalues)),
            "state_dimension": int(self.transition.shape[0]),
            "measurement_dimension": int(self.measurement_jacobian.shape[0]),
            "posterior_trace": float(np.trace(self.posterior_covariance)),
            "predicted_trace": float(np.trace(self.predicted_covariance)),
            "mean_bus_variance_reduction": float(np.mean(self.bus_variance_reduction)),
            "minimum_bus_variance_reduction": float(np.min(self.bus_variance_reduction)),
            "maximum_bus_risk": float(np.max(self.bus_risk)),
        }


def _posterior_covariance(predicted: np.ndarray, measurement: np.ndarray, noise: np.ndarray) -> np.ndarray:
    innovation = measurement @ predicted @ measurement.T + noise
    gain_rhs = linalg.solve(innovation, measurement @ predicted, assume_a="pos")
    posterior = predicted - predicted @ measurement.T @ gain_rhs
    return 0.5 * (posterior + posterior.T)


def graph_modal_reconstructibility(
    ybus: np.ndarray,
    observed_indices: Sequence[int],
    dt: float,
    *,
    coupling: float = 4.0,
    damping: float = 1.0,
    inertia: np.ndarray | None = None,
    process_angle_variance: float = 2e-7,
    process_frequency_variance: float = 2e-5,
    measurement_angle_variance: float = 3e-5,
    measurement_frequency_variance: float = 4e-4,
) -> ModalRiccatiResult:
    """Compute graph-modal visibility and steady-state Kalman uncertainty.

    The model follows the current benchmark convention: state
    ``[angle(rad), frequency(Hz)]``, with ``angle_dot = 2*pi*frequency``.  A
    scalar damping assumes damping proportional to inertia.  Under this
    homogeneity assumption the generalized Laplacian modes decouple in the
    dynamics, although sparse PMU measurements can still couple them in the
    posterior covariance.
    """

    if dt <= 0 or coupling < 0 or damping < 0:
        raise ValueError("dt must be positive and physical coefficients non-negative")
    variances = (
        process_angle_variance,
        process_frequency_variance,
        measurement_angle_variance,
        measurement_frequency_variance,
    )
    if any(value <= 0 for value in variances):
        raise ValueError("all covariance variances must be positive")

    laplacian = graph_laplacian(np.asarray(ybus, dtype=complex), normalized=True)
    n_bus = laplacian.shape[0]
    observed = np.asarray(observed_indices, dtype=int)
    if observed.ndim != 1 or len(observed) == 0:
        raise ValueError("at least one observed index is required")
    if len(np.unique(observed)) != len(observed) or np.any(observed < 0) or np.any(observed >= n_bus):
        raise ValueError("observed_indices must be unique valid bus indices")
    mass_values = np.ones(n_bus, dtype=float) if inertia is None else np.asarray(inertia, dtype=float)
    if mass_values.shape != (n_bus,) or np.any(mass_values <= 0):
        raise ValueError("inertia must contain one positive value per bus")
    mass = np.diag(mass_values)
    eigenvalues, modes = linalg.eigh(laplacian, mass)
    eigenvalues = np.maximum(eigenvalues, 0.0)

    mass_inverse_laplacian = laplacian / mass_values[:, None]
    transition = np.eye(2 * n_bus, dtype=float)
    transition[:n_bus, n_bus:] = 2.0 * np.pi * dt * np.eye(n_bus)
    transition[n_bus:, :n_bus] = -dt * coupling * mass_inverse_laplacian
    transition[n_bus:, n_bus:] = (1.0 - dt * damping) * np.eye(n_bus)

    measurement = np.zeros((2 * len(observed), 2 * n_bus), dtype=float)
    measurement[np.arange(len(observed)), observed] = 1.0
    measurement[len(observed) + np.arange(len(observed)), n_bus + observed] = 1.0
    process = np.diag(
        np.r_[
            np.full(n_bus, process_angle_variance),
            np.full(n_bus, process_frequency_variance),
        ]
    )
    measurement_noise = np.diag(
        np.r_[
            np.full(len(observed), measurement_angle_variance),
            np.full(len(observed), measurement_frequency_variance),
        ]
    )

    try:
        predicted = linalg.solve_discrete_are(
            transition.T,
            measurement.T,
            process,
            measurement_noise,
            balanced=True,
        )
    except linalg.LinAlgError as exc:
        raise ValueError(
            "The Riccati equation has no stabilizing solution for this PMU placement/model"
        ) from exc
    predicted = 0.5 * (predicted + predicted.T)
    posterior = _posterior_covariance(predicted, measurement, measurement_noise)

    modal_transform = linalg.block_diag(modes.T @ mass, modes.T @ mass)
    modal_predicted = modal_transform @ predicted @ modal_transform.T
    modal_posterior = modal_transform @ posterior @ modal_transform.T
    modal_pred_diag = np.r_[np.diag(modal_predicted)[:n_bus], np.diag(modal_predicted)[n_bus:]]
    modal_post_diag = np.r_[np.diag(modal_posterior)[:n_bus], np.diag(modal_posterior)[n_bus:]]
    modal_reduction_pair = 1.0 - modal_post_diag / np.maximum(modal_pred_diag, 1e-18)
    modal_reduction = 0.5 * (modal_reduction_pair[:n_bus] + modal_reduction_pair[n_bus:])

    observed_mode_values = modes[observed]
    modal_angle_information = np.sum(observed_mode_values**2, axis=0) / measurement_angle_variance
    modal_frequency_information = (
        np.sum(observed_mode_values**2, axis=0) / measurement_frequency_variance
    )
    predicted_diag = np.diag(predicted)
    posterior_diag = np.diag(posterior)
    angle_reduction = 1.0 - posterior_diag[:n_bus] / np.maximum(predicted_diag[:n_bus], 1e-18)
    frequency_reduction = 1.0 - posterior_diag[n_bus:] / np.maximum(predicted_diag[n_bus:], 1e-18)
    bus_reduction = 0.5 * (angle_reduction + frequency_reduction)
    bus_risk = (
        posterior_diag[:n_bus] / measurement_angle_variance
        + posterior_diag[n_bus:] / measurement_frequency_variance
    )
    return ModalRiccatiResult(
        laplacian_eigenvalues=eigenvalues,
        modes=modes,
        transition=transition,
        measurement_jacobian=measurement,
        predicted_covariance=predicted,
        posterior_covariance=posterior,
        modal_angle_information=modal_angle_information,
        modal_frequency_information=modal_frequency_information,
        modal_angle_variance=np.diag(modal_posterior)[:n_bus].copy(),
        modal_frequency_variance=np.diag(modal_posterior)[n_bus:].copy(),
        modal_variance_reduction=modal_reduction,
        bus_angle_variance=posterior_diag[:n_bus].copy(),
        bus_frequency_variance=posterior_diag[n_bus:].copy(),
        bus_variance_reduction=bus_reduction,
        bus_risk=bus_risk,
    )
