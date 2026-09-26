"""Estimators used by the calibrated transient reconstruction benchmark.

The estimators share a strict input/output contract so that a static WLS-like
baseline, a network model, a swing model, a robust MAP smoother, a cooperative
method, and a learned model can be compared on exactly the same observations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import time
from typing import Any, Protocol, Sequence

import joblib
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr, splu
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.simulation.transient_scenarios import ScenarioDataset


def _wrap_rad(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return (arr + np.pi) % (2.0 * np.pi) - np.pi


def _fill_timewise(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    valid_mask = np.asarray(mask, dtype=bool) & np.isfinite(out)
    for col in range(out.shape[1]):
        good = np.flatnonzero(valid_mask[:, col])
        if len(good) == 0:
            out[:, col] = 0.0
        elif len(good) == 1:
            out[:, col] = out[good[0], col]
        else:
            out[:, col] = np.interp(np.arange(len(out)), good, out[good, col])
    return out


def graph_laplacian(ybus: np.ndarray, normalized: bool = True) -> np.ndarray:
    y = np.asarray(ybus, dtype=complex)
    weights = np.abs(y.copy())
    np.fill_diagonal(weights, 0.0)
    weights = 0.5 * (weights + weights.T)
    degree = np.sum(weights, axis=1)
    lap = np.diag(degree) - weights
    if not normalized:
        return lap
    scale = np.sqrt(np.maximum(degree, 1e-12))
    return lap / scale[:, None] / scale[None, :]


@dataclass
class EstimationProblem:
    scenario_id: str
    event_type: int
    event_start_s: float
    time_s: np.ndarray
    bus_ids: np.ndarray
    observed_buses: np.ndarray
    observed_indices: np.ndarray
    hidden_indices: np.ndarray
    ybus: np.ndarray
    base_voltage_pu: np.ndarray
    base_angle_rad: np.ndarray
    base_frequency_hz: np.ndarray
    observed_voltage_pu: np.ndarray
    observed_angle_rad: np.ndarray
    observed_frequency_hz: np.ndarray
    observed_mask: np.ndarray
    truth_voltage_pu: np.ndarray
    truth_angle_rad: np.ndarray
    truth_frequency_hz: np.ndarray
    event_label: np.ndarray

    @classmethod
    def from_dataset(cls, dataset: ScenarioDataset) -> "EstimationProblem":
        positions = {int(bus): idx for idx, bus in enumerate(dataset.bus_ids)}
        observed_indices = np.asarray([positions[int(bus)] for bus in dataset.pmu_buses], dtype=int)
        hidden_indices = np.asarray(
            [idx for idx in range(len(dataset.bus_ids)) if idx not in set(observed_indices)],
            dtype=int,
        )
        pre = dataset.time_s < dataset.spec.event_start_s
        if not np.any(pre):
            pre = np.arange(len(dataset.time_s)) < max(1, len(dataset.time_s) // 5)
        # A known pre-event operating point is standard in dynamic state
        # estimation and can be obtained from the converged power flow.
        base_v = np.median(dataset.voltage_pu[pre], axis=0)
        base_a = np.angle(np.mean(np.exp(1j * dataset.angle_rad[pre]), axis=0))
        base_f = np.median(dataset.frequency_hz[pre], axis=0)
        return cls(
            scenario_id=dataset.spec.scenario_id,
            event_type=dataset.spec.event_type,
            event_start_s=dataset.spec.event_start_s,
            time_s=dataset.time_s,
            bus_ids=dataset.bus_ids,
            observed_buses=dataset.pmu_buses,
            observed_indices=observed_indices,
            hidden_indices=hidden_indices,
            ybus=dataset.ybus,
            base_voltage_pu=base_v,
            base_angle_rad=base_a,
            base_frequency_hz=base_f,
            observed_voltage_pu=dataset.observed_voltage_pu,
            observed_angle_rad=dataset.observed_angle_rad,
            observed_frequency_hz=dataset.observed_frequency_hz,
            observed_mask=dataset.observed_mask,
            truth_voltage_pu=dataset.voltage_pu,
            truth_angle_rad=dataset.angle_rad,
            truth_frequency_hz=dataset.frequency_hz,
            event_label=dataset.event_label,
        )

    @property
    def dt(self) -> float:
        if len(self.time_s) < 2:
            return 1.0 / 30.0
        return float(np.median(np.diff(self.time_s)))

    def observed_deviations(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mask = self.observed_mask
        voltage = self.observed_voltage_pu - self.base_voltage_pu[self.observed_indices]
        angle_absolute = _fill_timewise(self.observed_angle_rad, mask)
        angle = np.unwrap(angle_absolute, axis=0) - self.base_angle_rad[self.observed_indices]
        frequency = self.observed_frequency_hz - self.base_frequency_hz[self.observed_indices]
        voltage[~mask] = np.nan
        angle[~mask] = np.nan
        frequency[~mask] = np.nan
        return voltage, angle, frequency


@dataclass
class EstimateResult:
    name: str
    voltage_pu: np.ndarray
    angle_rad: np.ndarray
    frequency_hz: np.ndarray
    runtime_s: float
    diagnostics: dict[str, Any] = field(default_factory=dict)
    voltage_std_pu: np.ndarray | None = None
    angle_std_rad: np.ndarray | None = None
    frequency_std_hz: np.ndarray | None = None

    def save(self, directory: str | Path) -> Path:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {
            "voltage_pu": self.voltage_pu,
            "angle_rad": self.angle_rad,
            "frequency_hz": self.frequency_hz,
        }
        if self.voltage_std_pu is not None:
            arrays["voltage_std_pu"] = self.voltage_std_pu
        if self.angle_std_rad is not None:
            arrays["angle_std_rad"] = self.angle_std_rad
        if self.frequency_std_hz is not None:
            arrays["frequency_std_hz"] = self.frequency_std_hz
        np.savez_compressed(root / f"{self.name}.npz", **arrays)
        return root / f"{self.name}.npz"


class StateEstimator(Protocol):
    name: str

    def estimate(self, problem: EstimationProblem) -> EstimateResult: ...


def _graph_wls_series(
    observations: np.ndarray,
    mask: np.ndarray,
    observed_indices: np.ndarray,
    laplacian: np.ndarray,
    measurement_weight: float,
    smoothness_weight: float,
    prior_weight: float,
    temporal_weight: float = 0.0,
) -> np.ndarray:
    t_count = observations.shape[0]
    n_bus = laplacian.shape[0]
    result = np.zeros((t_count, n_bus), dtype=float)
    spatial = smoothness_weight * laplacian + prior_weight * np.eye(n_bus)
    previous = np.zeros(n_bus, dtype=float)
    cache: dict[tuple[bool, ...], tuple[np.ndarray, np.ndarray]] = {}
    for ti in range(t_count):
        valid = np.asarray(mask[ti], dtype=bool) & np.isfinite(observations[ti])
        key = tuple(bool(x) for x in valid)
        if key not in cache:
            c = np.zeros((int(np.sum(valid)), n_bus), dtype=float)
            if np.any(valid):
                c[np.arange(int(np.sum(valid))), observed_indices[valid]] = 1.0
            h = spatial + measurement_weight * (c.T @ c) + temporal_weight * np.eye(n_bus)
            cache[key] = (h, c)
        h, c = cache[key]
        rhs = temporal_weight * previous
        if np.any(valid):
            rhs = rhs + measurement_weight * c.T @ observations[ti, valid]
        try:
            current = np.linalg.solve(h, rhs)
        except np.linalg.LinAlgError:
            current = np.linalg.lstsq(h, rhs, rcond=None)[0]
        result[ti] = current
        previous = current
    return result


class StaticGraphWLSEstimator:
    """Static graph-regularized WLS using only the current PMU snapshot."""

    name = "static_graph_wls"

    def __init__(self, measurement_weight: float = 500.0, smoothness_weight: float = 4.0, prior_weight: float = 0.2):
        self.measurement_weight = measurement_weight
        self.smoothness_weight = smoothness_weight
        self.prior_weight = prior_weight

    def estimate(self, problem: EstimationProblem) -> EstimateResult:
        started = time.perf_counter()
        v_obs, a_obs, f_obs = problem.observed_deviations()
        lap = graph_laplacian(problem.ybus)
        kwargs = dict(
            mask=problem.observed_mask,
            observed_indices=problem.observed_indices,
            laplacian=lap,
            measurement_weight=self.measurement_weight,
            smoothness_weight=self.smoothness_weight,
            prior_weight=self.prior_weight,
        )
        v_dev = _graph_wls_series(v_obs, **kwargs)
        a_dev = _graph_wls_series(a_obs, **kwargs)
        f_dev = _graph_wls_series(f_obs, **kwargs)
        return EstimateResult(
            name=self.name,
            voltage_pu=problem.base_voltage_pu + v_dev,
            angle_rad=problem.base_angle_rad + a_dev,
            frequency_hz=problem.base_frequency_hz + f_dev,
            runtime_s=time.perf_counter() - started,
            diagnostics={"formulation": "snapshot graph WLS"},
        )


class TemporalYBusEstimator:
    """Complex-voltage Ybus estimator with prior and temporal regularization."""

    name = "temporal_ybus"

    def __init__(
        self,
        measurement_weight: float = 1_000.0,
        network_weight: float = 2e-3,
        prior_weight: float = 0.2,
        temporal_weight: float = 2.0,
    ) -> None:
        self.measurement_weight = measurement_weight
        self.network_weight = network_weight
        self.prior_weight = prior_weight
        self.temporal_weight = temporal_weight

    def estimate(self, problem: EstimationProblem) -> EstimateResult:
        started = time.perf_counter()
        n_bus = len(problem.bus_ids)
        base = problem.base_voltage_pu * np.exp(1j * problem.base_angle_rad)
        observed = problem.observed_voltage_pu * np.exp(1j * problem.observed_angle_rad)
        observed_delta = observed - base[problem.observed_indices]
        estimated = np.zeros((len(problem.time_s), n_bus), dtype=complex)
        previous = np.zeros(n_bus, dtype=complex)
        network = self.network_weight * (problem.ybus.conj().T @ problem.ybus)
        cache: dict[tuple[bool, ...], tuple[np.ndarray, np.ndarray]] = {}
        for ti in range(len(problem.time_s)):
            valid = problem.observed_mask[ti] & np.isfinite(observed_delta[ti])
            key = tuple(bool(x) for x in valid)
            if key not in cache:
                c = np.zeros((int(np.sum(valid)), n_bus), dtype=complex)
                if np.any(valid):
                    c[np.arange(int(np.sum(valid))), problem.observed_indices[valid]] = 1.0
                h = (
                    network
                    + self.measurement_weight * (c.conj().T @ c)
                    + (self.prior_weight + self.temporal_weight) * np.eye(n_bus)
                )
                cache[key] = (h, c)
            h, c = cache[key]
            rhs = self.temporal_weight * previous
            if np.any(valid):
                rhs = rhs + self.measurement_weight * c.conj().T @ observed_delta[ti, valid]
            try:
                current = np.linalg.solve(h, rhs)
            except np.linalg.LinAlgError:
                current = np.linalg.lstsq(h, rhs, rcond=None)[0]
            estimated[ti] = base + current
            previous = current

        _, _, f_obs = problem.observed_deviations()
        f_dev = _graph_wls_series(
            f_obs,
            problem.observed_mask,
            problem.observed_indices,
            graph_laplacian(problem.ybus),
            measurement_weight=500.0,
            smoothness_weight=3.0,
            prior_weight=0.1,
            temporal_weight=2.0,
        )
        return EstimateResult(
            name=self.name,
            voltage_pu=np.abs(estimated),
            angle_rad=np.angle(estimated),
            frequency_hz=problem.base_frequency_hz + f_dev,
            runtime_s=time.perf_counter() - started,
            diagnostics={"formulation": "complex Ybus temporal Tikhonov"},
        )


def _swing_transition(problem: EstimationProblem, coupling: float, damping: float) -> np.ndarray:
    n_bus = len(problem.bus_ids)
    lap = graph_laplacian(problem.ybus)
    dt = problem.dt
    transition = np.eye(2 * n_bus, dtype=float)
    transition[:n_bus, n_bus:] = 2.0 * np.pi * dt * np.eye(n_bus)
    transition[n_bus:, :n_bus] = -dt * coupling * lap
    transition[n_bus:, n_bus:] = (1.0 - dt * damping) * np.eye(n_bus)
    return transition


def _voltage_dynamic_series(problem: EstimationProblem, diffusion: float = 1.0) -> np.ndarray:
    v_obs, _, _ = problem.observed_deviations()
    return _graph_wls_series(
        v_obs,
        problem.observed_mask,
        problem.observed_indices,
        graph_laplacian(problem.ybus),
        measurement_weight=500.0,
        smoothness_weight=3.0,
        prior_weight=0.1,
        temporal_weight=max(diffusion, 0.0),
    )


class SwingKalmanEstimator:
    """Kalman filter with a linearized, coupled swing transition model."""

    name = "swing_kalman"

    def __init__(
        self,
        coupling: float = 1.5,
        damping: float = 1.0,
        process_scale: float = 1.0,
        measurement_scale: float = 1.0,
    ) -> None:
        if process_scale <= 0 or measurement_scale <= 0:
            raise ValueError("noise scales must be positive")
        self.coupling = coupling
        self.damping = damping
        self.process_scale = process_scale
        self.measurement_scale = measurement_scale

    def estimate(self, problem: EstimationProblem) -> EstimateResult:
        started = time.perf_counter()
        _, angle_obs, freq_obs = problem.observed_deviations()
        n_bus = len(problem.bus_ids)
        transition = _swing_transition(problem, self.coupling, self.damping)
        state = np.zeros(2 * n_bus, dtype=float)
        covariance = np.eye(2 * n_bus) * 1e-3
        process = self.process_scale * np.diag(
            np.r_[np.full(n_bus, 2e-7), np.full(n_bus, 2e-5)]
        )
        states = np.zeros((len(problem.time_s), 2 * n_bus), dtype=float)
        identity = np.eye(2 * n_bus)
        gated = 0
        for ti in range(len(problem.time_s)):
            if ti:
                state = transition @ state
                covariance = transition @ covariance @ transition.T + process
            valid = problem.observed_mask[ti] & np.isfinite(angle_obs[ti]) & np.isfinite(freq_obs[ti])
            if np.any(valid):
                indices = problem.observed_indices[valid]
                h = np.zeros((2 * len(indices), 2 * n_bus), dtype=float)
                h[np.arange(len(indices)), indices] = 1.0
                h[len(indices) + np.arange(len(indices)), n_bus + indices] = 1.0
                measurement = np.r_[angle_obs[ti, valid], freq_obs[ti, valid]]
                residual = measurement - h @ state
                measurement_cov = self.measurement_scale * np.diag(
                    np.r_[np.full(len(indices), 3e-5), np.full(len(indices), 4e-4)]
                )
                innovation = h @ covariance @ h.T + measurement_cov
                inv_innovation = np.linalg.pinv(innovation)
                # A conservative innovation gate prevents one bad PMU from
                # destabilizing all hidden states while retaining diagnostics.
                standardized = np.abs(residual) / np.sqrt(np.maximum(np.diag(innovation), 1e-12))
                keep = standardized < 8.0
                if not np.all(keep):
                    gated += int(np.sum(~keep))
                    h = h[keep]
                    residual = residual[keep]
                    measurement_cov = measurement_cov[np.ix_(keep, keep)]
                    innovation = h @ covariance @ h.T + measurement_cov
                    inv_innovation = np.linalg.pinv(innovation)
                if len(residual):
                    gain = covariance @ h.T @ inv_innovation
                    state = state + gain @ residual
                    covariance = (identity - gain @ h) @ covariance @ (identity - gain @ h).T + gain @ measurement_cov @ gain.T
            states[ti] = state

        v_dev = _voltage_dynamic_series(problem)
        return EstimateResult(
            name=self.name,
            voltage_pu=problem.base_voltage_pu + v_dev,
            angle_rad=problem.base_angle_rad + states[:, :n_bus],
            frequency_hz=problem.base_frequency_hz + states[:, n_bus:],
            runtime_s=time.perf_counter() - started,
            diagnostics={"formulation": "linearized coupled swing KF", "gated_measurements": gated},
        )


def _nonlinear_swing_step(
    sigma_points: np.ndarray,
    base_angle: np.ndarray,
    weights: np.ndarray,
    dt: float,
    coupling: float,
    damping: float,
) -> np.ndarray:
    n_bus = len(base_angle)
    delta = sigma_points[:, :n_bus]
    freq = sigma_points[:, n_bus:]
    total_angle = delta + base_angle
    output = sigma_points.copy()
    output[:, :n_bus] = delta + 2.0 * np.pi * dt * freq
    for row in range(len(sigma_points)):
        difference = total_angle[row, :, None] - total_angle[row, None, :]
        base_difference = base_angle[:, None] - base_angle[None, :]
        electrical = np.sum(weights * (np.sin(difference) - np.sin(base_difference)), axis=1)
        output[row, n_bus:] = freq[row] + dt * (-damping * freq[row] - coupling * electrical)
    return output


class UnscentedSwingEstimator:
    """UKF using the nonlinear sinusoidal coupling of a reduced swing model."""

    name = "swing_ukf"

    def __init__(
        self,
        coupling: float = 0.8,
        damping: float = 1.0,
        alpha: float = 0.15,
        process_scale: float = 1.0,
        measurement_scale: float = 1.0,
    ) -> None:
        if process_scale <= 0 or measurement_scale <= 0:
            raise ValueError("noise scales must be positive")
        self.coupling = coupling
        self.damping = damping
        self.alpha = alpha
        self.process_scale = process_scale
        self.measurement_scale = measurement_scale

    def estimate(self, problem: EstimationProblem) -> EstimateResult:
        started = time.perf_counter()
        _, angle_obs, freq_obs = problem.observed_deviations()
        n_bus = len(problem.bus_ids)
        dimension = 2 * n_bus
        alpha = self.alpha
        beta = 2.0
        kappa = 0.0
        lam = alpha**2 * (dimension + kappa) - dimension
        scale = dimension + lam
        mean_weights = np.full(2 * dimension + 1, 1.0 / (2.0 * scale))
        covariance_weights = mean_weights.copy()
        mean_weights[0] = lam / scale
        covariance_weights[0] = mean_weights[0] + (1.0 - alpha**2 + beta)
        graph_weights = np.abs(problem.ybus.copy())
        np.fill_diagonal(graph_weights, 0.0)
        graph_weights = 0.5 * (graph_weights + graph_weights.T)
        graph_weights /= np.maximum(np.sum(graph_weights, axis=1, keepdims=True), 1e-12)

        state = np.zeros(dimension, dtype=float)
        covariance = np.eye(dimension) * 5e-4
        process = self.process_scale * np.diag(
            np.r_[np.full(n_bus, 2e-7), np.full(n_bus, 2e-5)]
        )
        states = np.zeros((len(problem.time_s), dimension), dtype=float)
        for ti in range(len(problem.time_s)):
            if ti:
                jitter = 1e-10
                for _ in range(5):
                    try:
                        root = np.linalg.cholesky(scale * (covariance + jitter * np.eye(dimension)))
                        break
                    except np.linalg.LinAlgError:
                        jitter *= 100.0
                else:
                    root = np.linalg.cholesky(scale * (np.eye(dimension) * 1e-3))
                sigma = np.vstack([state, state + root.T, state - root.T])
                propagated = _nonlinear_swing_step(
                    sigma,
                    problem.base_angle_rad,
                    graph_weights,
                    problem.dt,
                    self.coupling,
                    self.damping,
                )
                state = mean_weights @ propagated
                centered = propagated - state
                covariance = (centered.T * covariance_weights) @ centered + process
                covariance = 0.5 * (covariance + covariance.T)

            valid = problem.observed_mask[ti] & np.isfinite(angle_obs[ti]) & np.isfinite(freq_obs[ti])
            if np.any(valid):
                indices = problem.observed_indices[valid]
                h = np.zeros((2 * len(indices), dimension), dtype=float)
                h[np.arange(len(indices)), indices] = 1.0
                h[len(indices) + np.arange(len(indices)), n_bus + indices] = 1.0
                measurement = np.r_[angle_obs[ti, valid], freq_obs[ti, valid]]
                measurement_cov = self.measurement_scale * np.diag(
                    np.r_[np.full(len(indices), 3e-5), np.full(len(indices), 4e-4)]
                )
                innovation = h @ covariance @ h.T + measurement_cov
                gain = covariance @ h.T @ np.linalg.pinv(innovation)
                state = state + gain @ (measurement - h @ state)
                covariance = covariance - gain @ innovation @ gain.T
                covariance = 0.5 * (covariance + covariance.T) + 1e-12 * np.eye(dimension)
            states[ti] = state

        v_dev = _voltage_dynamic_series(problem)
        return EstimateResult(
            name=self.name,
            voltage_pu=problem.base_voltage_pu + v_dev,
            angle_rad=problem.base_angle_rad + states[:, :n_bus],
            frequency_hz=problem.base_frequency_hz + states[:, n_bus:],
            runtime_s=time.perf_counter() - started,
            diagnostics={"formulation": "nonlinear coupled swing UKF", "sigma_points": 2 * dimension + 1},
        )


def _robust_linear_smoother(
    observations: np.ndarray,
    observation_mask: np.ndarray,
    observed_indices: np.ndarray,
    transition: np.ndarray,
    initial: np.ndarray,
    process_sigma: np.ndarray,
    measurement_sigma: np.ndarray,
    robust_iterations: int = 3,
    huber_delta: float = 2.5,
    robust_loss: str = "huber",
    student_df: float = 4.0,
    uncertainty_probes: int = 0,
    random_seed: int = 2026,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray | None]:
    """Solve a fixed-window Gaussian MAP problem with Huber IRLS."""

    if robust_loss not in {"huber", "student_t"}:
        raise ValueError("robust_loss must be 'huber' or 'student_t'")
    if student_df <= 0:
        raise ValueError("student_df must be positive")

    t_count = observations.shape[0]
    state_dim = len(initial)
    valid_records: list[tuple[int, int, float, float]] = []
    for ti in range(t_count):
        for local_idx, global_idx in enumerate(observed_indices):
            if observation_mask[ti, local_idx] and np.isfinite(observations[ti, local_idx]):
                sigma = float(measurement_sigma[global_idx])
                valid_records.append((ti, int(global_idx), float(observations[ti, local_idx]), sigma))

    prior_rows = state_dim
    transition_rows = max(0, t_count - 1) * state_dim
    measurement_rows = len(valid_records)
    row_count = prior_rows + transition_rows + measurement_rows
    col_count = t_count * state_dim
    measurement_weights = np.ones(measurement_rows, dtype=float)
    solution = np.tile(initial, t_count)

    for iteration in range(max(1, robust_iterations)):
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        rhs = np.zeros(row_count, dtype=float)
        row = 0
        prior_sigma = np.maximum(process_sigma, 1e-5)
        for idx in range(state_dim):
            weight = 1.0 / prior_sigma[idx]
            rows.append(row)
            cols.append(idx)
            data.append(weight)
            rhs[row] = initial[idx] * weight
            row += 1
        for ti in range(1, t_count):
            previous_offset = (ti - 1) * state_dim
            current_offset = ti * state_dim
            for state_idx in range(state_dim):
                weight = 1.0 / max(float(process_sigma[state_idx]), 1e-8)
                nonzero = np.flatnonzero(np.abs(transition[state_idx]) > 1e-14)
                for source_idx in nonzero:
                    rows.append(row)
                    cols.append(previous_offset + int(source_idx))
                    data.append(-float(transition[state_idx, source_idx]) * weight)
                rows.append(row)
                cols.append(current_offset + state_idx)
                data.append(weight)
                row += 1
        measurement_row_start = row
        for rec_idx, (ti, state_idx, value, sigma) in enumerate(valid_records):
            weight = math.sqrt(measurement_weights[rec_idx]) / max(sigma, 1e-8)
            rows.append(row)
            cols.append(ti * state_dim + state_idx)
            data.append(weight)
            rhs[row] = value * weight
            row += 1
        design = sparse.coo_matrix((data, (rows, cols)), shape=(row_count, col_count)).tocsr()
        result = lsqr(design, rhs, atol=1e-7, btol=1e-7, iter_lim=800)
        solution = result[0]
        if measurement_rows and iteration + 1 < robust_iterations:
            residuals = np.zeros(measurement_rows, dtype=float)
            for rec_idx, (ti, state_idx, value, sigma) in enumerate(valid_records):
                residuals[rec_idx] = (solution[ti * state_dim + state_idx] - value) / max(sigma, 1e-8)
            magnitude = np.abs(residuals)
            if robust_loss == "huber":
                measurement_weights = np.where(
                    magnitude <= huber_delta,
                    1.0,
                    huber_delta / np.maximum(magnitude, 1e-12),
                )
            else:
                measurement_weights = (student_df + 1.0) / (student_df + residuals**2)

    posterior_std = None
    posterior_method = "not_requested"
    if uncertainty_probes > 0 and col_count:
        # The converged IRLS normal matrix is a Laplace approximation to the
        # posterior precision. Hutchinson probing estimates its inverse
        # diagonal without materializing a dense (T * state_dim)^2 matrix.
        precision = (design.T @ design).tocsc()
        scale = max(float(np.max(np.abs(precision.diagonal()))), 1.0)
        precision = precision + sparse.eye(col_count, format="csc") * (1e-10 * scale)
        try:
            factor = splu(precision)
            rng = np.random.default_rng(random_seed)
            variance = np.zeros(col_count, dtype=float)
            for _ in range(int(uncertainty_probes)):
                probe = rng.choice(np.array([-1.0, 1.0]), size=col_count)
                variance += probe * factor.solve(probe)
            variance /= float(uncertainty_probes)
            posterior_std = np.sqrt(np.maximum(variance, 1e-14)).reshape(t_count, state_dim)
            posterior_method = f"Laplace precision inverse diagonal, {uncertainty_probes} Hutchinson probes"
        except RuntimeError:
            posterior_method = "Laplace factorization failed"

    return solution.reshape(t_count, state_dim), {
        "lsqr_iterations": int(result[2]),
        "lsqr_residual_norm": float(result[3]),
        "downweighted_measurements": int(np.sum(measurement_weights < 0.999)),
        "measurement_rows": measurement_rows,
        "transition_rows": transition_rows,
        "robust_loss": robust_loss,
        "posterior_method": posterior_method,
    }, posterior_std


class RobustMHEMAPEstimator:
    """Robust fixed-window MHE/MAP with swing and voltage-diffusion priors."""

    name = "robust_mhe_map"

    def __init__(
        self,
        coupling: float = 1.5,
        damping: float = 1.0,
        robust_iterations: int = 3,
        process_scale: float = 1.0,
        measurement_scale: float = 1.0,
    ) -> None:
        if process_scale <= 0 or measurement_scale <= 0:
            raise ValueError("noise scales must be positive")
        self.coupling = coupling
        self.damping = damping
        self.robust_iterations = robust_iterations
        self.process_scale = process_scale
        self.measurement_scale = measurement_scale

    def estimate(self, problem: EstimationProblem) -> EstimateResult:
        started = time.perf_counter()
        v_obs, a_obs, f_obs = problem.observed_deviations()
        n_bus = len(problem.bus_ids)
        swing_observations = np.concatenate([a_obs, f_obs], axis=1)
        swing_mask = np.concatenate([problem.observed_mask, problem.observed_mask], axis=1)
        swing_indices = np.r_[problem.observed_indices, n_bus + problem.observed_indices]
        swing_states, swing_diag, _ = _robust_linear_smoother(
            observations=swing_observations,
            observation_mask=swing_mask,
            observed_indices=swing_indices,
            transition=_swing_transition(problem, self.coupling, self.damping),
            initial=np.zeros(2 * n_bus),
            process_sigma=self.process_scale * np.r_[np.full(n_bus, 8e-4), np.full(n_bus, 8e-3)],
            measurement_sigma=self.measurement_scale * np.r_[np.full(n_bus, 3e-3), np.full(n_bus, 1.5e-2)],
            robust_iterations=self.robust_iterations,
        )
        lap = graph_laplacian(problem.ybus)
        voltage_transition = np.eye(n_bus) - min(problem.dt * 0.8, 0.2) * lap
        voltage_states, voltage_diag, _ = _robust_linear_smoother(
            observations=v_obs,
            observation_mask=problem.observed_mask,
            observed_indices=problem.observed_indices,
            transition=voltage_transition,
            initial=np.zeros(n_bus),
            process_sigma=self.process_scale * np.full(n_bus, 1.5e-3),
            measurement_sigma=self.measurement_scale * np.full(n_bus, 4e-3),
            robust_iterations=self.robust_iterations,
        )
        return EstimateResult(
            name=self.name,
            voltage_pu=problem.base_voltage_pu + voltage_states,
            angle_rad=problem.base_angle_rad + swing_states[:, :n_bus],
            frequency_hz=problem.base_frequency_hz + swing_states[:, n_bus:],
            runtime_s=time.perf_counter() - started,
            diagnostics={"formulation": "Huber IRLS fixed-window MAP", "swing": swing_diag, "voltage": voltage_diag},
        )


class BayesianGraphMHEEstimator:
    """Physics-first Bayesian graph smoother for sparse-PMU reconstruction.

    The latent vector at every sample contains bus-voltage, bus-angle, and
    frequency deviations. A reduced network swing transition couples angle and
    frequency through the admittance Laplacian; voltage follows a graph
    diffusion prior. Student-t IRLS provides a robust likelihood and the final
    sparse precision matrix supplies a Laplace posterior approximation.

    This is intentionally a reduced bus-level model. It is a reproducible step
    between the existing linear MHE and a full generator/network DAE estimator;
    it must not be described as the full ANDES DAE inverse model.
    """

    name = "bayesian_graph_mhe"

    def __init__(
        self,
        coupling: float = 1.5,
        damping: float = 1.0,
        voltage_diffusion: float = 0.8,
        robust_iterations: int = 4,
        student_df: float = 4.0,
        uncertainty_probes: int = 6,
        process_scale: float = 1.0,
        measurement_scale: float = 1.0,
    ) -> None:
        if coupling < 0 or damping < 0 or voltage_diffusion < 0:
            raise ValueError("physics weights must be non-negative")
        if process_scale <= 0 or measurement_scale <= 0:
            raise ValueError("noise scales must be positive")
        self.coupling = coupling
        self.damping = damping
        self.voltage_diffusion = voltage_diffusion
        self.robust_iterations = robust_iterations
        self.student_df = student_df
        self.uncertainty_probes = uncertainty_probes
        self.process_scale = process_scale
        self.measurement_scale = measurement_scale

    def _transition(self, problem: EstimationProblem) -> np.ndarray:
        n_bus = len(problem.bus_ids)
        dt = problem.dt
        lap = graph_laplacian(problem.ybus)
        transition = np.zeros((3 * n_bus, 3 * n_bus), dtype=float)
        transition[:n_bus, :n_bus] = np.eye(n_bus) - min(dt * self.voltage_diffusion, 0.2) * lap
        transition[n_bus : 2 * n_bus, n_bus : 2 * n_bus] = np.eye(n_bus)
        transition[n_bus : 2 * n_bus, 2 * n_bus :] = 2.0 * np.pi * dt * np.eye(n_bus)
        transition[2 * n_bus :, n_bus : 2 * n_bus] = -dt * self.coupling * lap
        transition[2 * n_bus :, 2 * n_bus :] = (1.0 - dt * self.damping) * np.eye(n_bus)
        return transition

    def estimate(self, problem: EstimationProblem) -> EstimateResult:
        started = time.perf_counter()
        v_obs, a_obs, f_obs = problem.observed_deviations()
        n_bus = len(problem.bus_ids)
        observations = np.concatenate([v_obs, a_obs, f_obs], axis=1)
        observation_mask = np.concatenate([problem.observed_mask] * 3, axis=1)
        observed_indices = np.r_[
            problem.observed_indices,
            n_bus + problem.observed_indices,
            2 * n_bus + problem.observed_indices,
        ]
        states, diagnostics, posterior_std = _robust_linear_smoother(
            observations=observations,
            observation_mask=observation_mask,
            observed_indices=observed_indices,
            transition=self._transition(problem),
            initial=np.zeros(3 * n_bus),
            process_sigma=self.process_scale * np.r_[
                np.full(n_bus, 1.5e-3),
                np.full(n_bus, 8e-4),
                np.full(n_bus, 8e-3),
            ],
            measurement_sigma=self.measurement_scale * np.r_[
                np.full(n_bus, 4e-3),
                np.full(n_bus, 3e-3),
                np.full(n_bus, 1.5e-2),
            ],
            robust_iterations=self.robust_iterations,
            robust_loss="student_t",
            student_df=self.student_df,
            uncertainty_probes=self.uncertainty_probes,
            random_seed=2026 + problem.event_type,
        )
        kwargs: dict[str, np.ndarray | None] = {
            "voltage_std_pu": None,
            "angle_std_rad": None,
            "frequency_std_hz": None,
        }
        if posterior_std is not None:
            kwargs = {
                "voltage_std_pu": posterior_std[:, :n_bus],
                "angle_std_rad": posterior_std[:, n_bus : 2 * n_bus],
                "frequency_std_hz": posterior_std[:, 2 * n_bus :],
            }
        return EstimateResult(
            name=self.name,
            voltage_pu=problem.base_voltage_pu + states[:, :n_bus],
            angle_rad=problem.base_angle_rad + states[:, n_bus : 2 * n_bus],
            frequency_hz=problem.base_frequency_hz + states[:, 2 * n_bus :],
            runtime_s=time.perf_counter() - started,
            diagnostics={
                "formulation": "Student-t Bayesian reduced swing/graph fixed-window MAP",
                "state_dimension_per_sample": 3 * n_bus,
                "coupling": self.coupling,
                "damping": self.damping,
                "voltage_diffusion": self.voltage_diffusion,
                "student_df": self.student_df,
                "process_scale": self.process_scale,
                "measurement_scale": self.measurement_scale,
                **diagnostics,
            },
            **kwargs,
        )


class CooperativeConsensusEstimator:
    """Distributed consensus-plus-innovation reconstruction.

    Each bus uses a swing prediction, exchanges only its state with electrical
    neighbors, and assimilates a local PMU innovation when one is available.
    The implementation is synchronous to make the distributed algorithm fully
    reproducible on one process.
    """

    name = "cooperative_consensus"

    def __init__(self, consensus_iterations: int = 35, spatial_weight: float = 2.0, measurement_weight: float = 80.0):
        self.consensus_iterations = consensus_iterations
        self.spatial_weight = spatial_weight
        self.measurement_weight = measurement_weight

    def _consensus(
        self,
        predicted: np.ndarray,
        observation: np.ndarray,
        valid: np.ndarray,
        observed_indices: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        current = predicted.copy()
        degree = np.sum(weights, axis=1)
        local_measurement = np.zeros_like(current)
        local_weight = np.zeros_like(current)
        if np.any(valid):
            local_measurement[observed_indices[valid]] = observation[valid]
            local_weight[observed_indices[valid]] = self.measurement_weight
        for _ in range(self.consensus_iterations):
            neighbor_sum = weights @ current
            numerator = predicted + self.spatial_weight * neighbor_sum + local_weight * local_measurement
            denominator = 1.0 + self.spatial_weight * degree + local_weight
            current = numerator / np.maximum(denominator, 1e-12)
        return current

    def estimate(self, problem: EstimationProblem) -> EstimateResult:
        started = time.perf_counter()
        v_obs, a_obs, f_obs = problem.observed_deviations()
        n_bus = len(problem.bus_ids)
        weights = np.abs(problem.ybus.copy())
        np.fill_diagonal(weights, 0.0)
        weights = 0.5 * (weights + weights.T)
        weights /= max(float(np.max(np.sum(weights, axis=1))), 1e-12)
        transition = _swing_transition(problem, coupling=1.5, damping=1.0)
        state = np.zeros(2 * n_bus, dtype=float)
        voltage = np.zeros(n_bus, dtype=float)
        states = np.zeros((len(problem.time_s), 2 * n_bus), dtype=float)
        voltages = np.zeros((len(problem.time_s), n_bus), dtype=float)
        for ti in range(len(problem.time_s)):
            predicted = transition @ state if ti else state
            valid = problem.observed_mask[ti]
            predicted[:n_bus] = self._consensus(
                predicted[:n_bus], a_obs[ti], valid & np.isfinite(a_obs[ti]), problem.observed_indices, weights
            )
            predicted[n_bus:] = self._consensus(
                predicted[n_bus:], f_obs[ti], valid & np.isfinite(f_obs[ti]), problem.observed_indices, weights
            )
            voltage = self._consensus(
                voltage, v_obs[ti], valid & np.isfinite(v_obs[ti]), problem.observed_indices, weights
            )
            state = predicted
            states[ti] = state
            voltages[ti] = voltage
        return EstimateResult(
            name=self.name,
            voltage_pu=problem.base_voltage_pu + voltages,
            angle_rad=problem.base_angle_rad + states[:, :n_bus],
            frequency_hz=problem.base_frequency_hz + states[:, n_bus:],
            runtime_s=time.perf_counter() - started,
            diagnostics={
                "formulation": "distributed consensus plus local innovation",
                "consensus_iterations": self.consensus_iterations,
            },
        )


def _learned_features(problem: EstimationProblem) -> np.ndarray:
    v, a, f = problem.observed_deviations()
    mask = problem.observed_mask.astype(float)
    filled = [_fill_timewise(values, problem.observed_mask) for values in (v, a, f)]
    derivatives = [np.gradient(values, problem.dt, axis=0) for values in filled]
    return np.column_stack([*filled, *derivatives, mask])


def _learned_targets(problem: EstimationProblem) -> np.ndarray:
    v = problem.truth_voltage_pu - problem.base_voltage_pu
    a = np.unwrap(problem.truth_angle_rad, axis=0) - problem.base_angle_rad
    f = problem.truth_frequency_hz - problem.base_frequency_hz
    return np.column_stack([v, a, f])


class DataDrivenRidgeEstimator:
    """Multi-output dynamic ridge model trained only on complete simulations."""

    name = "data_driven_ridge"

    def __init__(self, alpha: float = 5.0) -> None:
        self.alpha = alpha
        self.model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        self.bus_ids: np.ndarray | None = None
        self.observed_buses: np.ndarray | None = None
        self._fitted = False

    def fit(self, problems: Sequence[EstimationProblem]) -> "DataDrivenRidgeEstimator":
        if not problems:
            raise ValueError("At least one training scenario is required")
        first = problems[0]
        for problem in problems[1:]:
            if not np.array_equal(problem.bus_ids, first.bus_ids) or not np.array_equal(problem.observed_buses, first.observed_buses):
                raise ValueError("All learned scenarios must use the same network and PMU placement")
        features = np.vstack([_learned_features(problem) for problem in problems])
        targets = np.vstack([_learned_targets(problem) for problem in problems])
        self.model.fit(features, targets)
        self.bus_ids = first.bus_ids.copy()
        self.observed_buses = first.observed_buses.copy()
        self._fitted = True
        return self

    def estimate(self, problem: EstimationProblem) -> EstimateResult:
        if not self._fitted:
            raise RuntimeError("DataDrivenRidgeEstimator.fit must be called before estimate")
        started = time.perf_counter()
        predicted = self.model.predict(_learned_features(problem))
        n_bus = len(problem.bus_ids)
        return EstimateResult(
            name=self.name,
            voltage_pu=problem.base_voltage_pu + predicted[:, :n_bus],
            angle_rad=problem.base_angle_rad + predicted[:, n_bus : 2 * n_bus],
            frequency_hz=problem.base_frequency_hz + predicted[:, 2 * n_bus :],
            runtime_s=time.perf_counter() - started,
            diagnostics={"formulation": "standardized multi-output ridge", "alpha": self.alpha},
        )

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "DataDrivenRidgeEstimator":
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"Expected {cls.__name__}, found {type(model).__name__}")
        return model


class HybridPhysicsDataEstimator:
    """Convex fusion of robust physics MAP and a learned reconstruction."""

    name = "hybrid_physics_data"

    def __init__(
        self,
        learned: DataDrivenRidgeEstimator,
        physics: StateEstimator | None = None,
        physics_weight: float = 0.65,
    ) -> None:
        if not 0.0 <= physics_weight <= 1.0:
            raise ValueError("physics_weight must be in [0, 1]")
        self.learned = learned
        self.physics = physics or RobustMHEMAPEstimator()
        self.physics_weight = physics_weight

    def estimate(self, problem: EstimationProblem) -> EstimateResult:
        started = time.perf_counter()
        physical = self.physics.estimate(problem)
        learned = self.learned.estimate(problem)
        return self.fuse(physical, learned, runtime_s=time.perf_counter() - started)

    def fuse(
        self,
        physical: EstimateResult,
        learned: EstimateResult,
        runtime_s: float | None = None,
    ) -> EstimateResult:
        """Fuse already-computed results without rerunning the MAP estimator."""

        weight = self.physics_weight
        return EstimateResult(
            name=self.name,
            voltage_pu=weight * physical.voltage_pu + (1.0 - weight) * learned.voltage_pu,
            angle_rad=weight * physical.angle_rad + (1.0 - weight) * learned.angle_rad,
            frequency_hz=weight * physical.frequency_hz + (1.0 - weight) * learned.frequency_hz,
            runtime_s=float(runtime_s if runtime_s is not None else physical.runtime_s + learned.runtime_s),
            diagnostics={
                "formulation": "convex robust-MAP/data-driven fusion",
                "physics_weight": weight,
                "physics_runtime_s": physical.runtime_s,
                "learned_runtime_s": learned.runtime_s,
            },
        )


def swing_observability_diagnostics(
    problem: EstimationProblem,
    coupling: float = 1.5,
    damping: float = 1.0,
    horizon_steps: int | None = None,
) -> dict[str, Any]:
    """Numerically assess local observability of the linearized swing model."""

    n_bus = len(problem.bus_ids)
    state_dim = 2 * n_bus
    transition = _swing_transition(problem, coupling, damping)
    h = np.zeros((2 * len(problem.observed_indices), state_dim), dtype=float)
    h[np.arange(len(problem.observed_indices)), problem.observed_indices] = 1.0
    h[len(problem.observed_indices) + np.arange(len(problem.observed_indices)), n_bus + problem.observed_indices] = 1.0
    blocks = []
    power = np.eye(state_dim)
    steps = int(horizon_steps or state_dim)
    for _ in range(steps):
        blocks.append(h @ power)
        power = power @ transition
    matrix = np.vstack(blocks)
    singular = np.linalg.svd(matrix, compute_uv=False)
    tolerance = max(matrix.shape) * np.finfo(float).eps * max(float(singular[0]), 1.0)
    rank = int(np.sum(singular > tolerance))
    nonzero = singular[singular > tolerance]
    condition = float(nonzero[0] / nonzero[-1]) if len(nonzero) else float("inf")
    return {
        "state_dimension": state_dim,
        "measurement_dimension": int(h.shape[0]),
        "horizon_steps": steps,
        "observability_rank": rank,
        "fully_observable_linearized": rank == state_dim,
        "condition_number_nonzero": condition,
        "smallest_nonzero_singular_value": float(nonzero[-1]) if len(nonzero) else 0.0,
    }


def evaluate_estimate(problem: EstimationProblem, estimate: EstimateResult) -> dict[str, Any]:
    """Return hidden-bus metrics for the full trajectory and event interval."""

    hidden = problem.hidden_indices
    if len(hidden) == 0:
        raise ValueError("The benchmark needs at least one hidden bus")
    event_mask = problem.event_label != 0
    if not np.any(event_mask):
        event_mask = problem.time_s >= problem.event_start_s

    def metrics_for_rows(rows: np.ndarray) -> dict[str, float]:
        truth_v = problem.truth_voltage_pu[np.ix_(rows, hidden)]
        pred_v = estimate.voltage_pu[np.ix_(rows, hidden)]
        truth_a = problem.truth_angle_rad[np.ix_(rows, hidden)]
        pred_a = estimate.angle_rad[np.ix_(rows, hidden)]
        truth_f = problem.truth_frequency_hz[np.ix_(rows, hidden)]
        pred_f = estimate.frequency_hz[np.ix_(rows, hidden)]
        v_error = pred_v - truth_v
        a_error = _wrap_rad(pred_a - truth_a)
        f_error = pred_f - truth_f
        metrics = {
            "voltage_rmse_pu": float(np.sqrt(np.mean(v_error**2))),
            "voltage_mae_pu": float(np.mean(np.abs(v_error))),
            "angle_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(a_error**2)))),
            "angle_mae_deg": float(np.rad2deg(np.mean(np.abs(a_error)))),
            "frequency_rmse_mhz": float(1_000.0 * np.sqrt(np.mean(f_error**2))),
            "frequency_mae_mhz": float(1_000.0 * np.mean(np.abs(f_error))),
        }
        uncertainty_blocks = (
            ("voltage", v_error, estimate.voltage_std_pu),
            ("angle", a_error, estimate.angle_std_rad),
            ("frequency", f_error, estimate.frequency_std_hz),
        )
        for label, error, std in uncertainty_blocks:
            if std is None:
                continue
            selected_std = np.maximum(std[np.ix_(rows, hidden)], 1e-9)
            metrics[f"{label}_coverage_90"] = float(
                np.mean(np.abs(error) <= 1.6448536269514722 * selected_std)
            )
            metrics[f"{label}_gaussian_nll"] = float(
                np.mean(np.log(selected_std) + 0.5 * (error / selected_std) ** 2)
            )
        return metrics

    full_rows = np.arange(len(problem.time_s))
    event_rows = np.flatnonzero(event_mask)
    return {
        "scenario_id": problem.scenario_id,
        "event_type": problem.event_type,
        "estimator": estimate.name,
        "hidden_bus_count": int(len(hidden)),
        "runtime_s": float(estimate.runtime_s),
        **{f"full_{key}": value for key, value in metrics_for_rows(full_rows).items()},
        **{f"event_{key}": value for key, value in metrics_for_rows(event_rows).items()},
    }
