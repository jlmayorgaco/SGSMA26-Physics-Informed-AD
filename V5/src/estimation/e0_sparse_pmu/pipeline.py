"""E0 sparse-PMU state/signal estimation on the PowerDynamics IEEE-39 case.

The estimator phase receives only the eight files in ``input_sparse``.  The
39 reference files are copied to ``ground_truth`` and are opened only by the
post-estimation evaluator.  This separation is deliberate: it prevents a
comparison artifact from silently becoming an estimator input.

The E0-R reconstruction model is a regularized complex-voltage estimator with
latent, power-factor-preserving load multipliers.  Nominal load admittances
are priors, not hard constraints: for each static-load bus the equation is
``S(V)=V*conj(YV)=(1 + alpha)S0`` and ``alpha`` is estimated from the
observed voltage phasors plus the network model.  A sparse/temporal prior
keeps most load multipliers at zero while allowing a short event.  Current is
derived as ``Y V`` and is reported as out-of-model validation until its PMU
semantics are canonicalized.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import re
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, TwoSlopeNorm
from scipy.optimize import lsq_linear


ALL_BUSES = tuple(f"BUS{i}" for i in range(1, 40))
OBSERVED_BUSES = ("BUS2", "BUS5", "BUS6", "BUS10", "BUS19", "BUS22", "BUS29", "BUS39")
SIGNALS = ("VA_MAG", "VA_ANG", "IA_MAG", "IA_ANG", "Freq", "ROCOF")
WINDOWS = {
    "nominal": (0.0, 5.0, False),
    "event": (5.0, 10.0, False),
    "recovery": (10.0, 15.0, True),
}
SCENARIO_ID = "SIM_PD39_SPARSE_PMU_E0"
BASE_MVA = 100.0
NOMINAL_FREQUENCY_HZ = 60.0
SAMPLE_RATE_HZ = 30.0
LOAD_ALPHA_MIN = -0.5
LOAD_ALPHA_MAX = 0.5
CORE_SIGNALS = ("VA_MAG", "VA_ANG", "Freq", "ROCOF")
CURRENT_VALIDATION_SIGNALS = ("IA_MAG", "IA_ANG")


@dataclass(frozen=True)
class ReferenceData:
    buses: tuple[str, ...]
    timestamps: np.ndarray
    derivative_times: np.ndarray
    frames: dict[str, pd.DataFrame]
    voltage_pu: np.ndarray
    current_pu: np.ndarray
    voltage_v: np.ndarray
    current_a: np.ndarray
    frequency_hz: np.ndarray
    rocof_hz_s: np.ndarray
    event: np.ndarray
    base_kv: np.ndarray


@dataclass(frozen=True)
class EstimateData:
    timestamps: np.ndarray
    voltage_pu: np.ndarray
    current_pu: np.ndarray
    voltage_v: np.ndarray
    current_a: np.ndarray
    frequency_hz: np.ndarray
    rocof_hz_s: np.ndarray
    alpha_estimates: np.ndarray
    sigma_voltage_v: np.ndarray
    sigma_voltage_angle_deg: np.ndarray
    sigma_current_a: np.ndarray
    sigma_current_angle_deg: np.ndarray
    sigma_frequency_hz: np.ndarray
    sigma_rocof_hz_s: np.ndarray
    diagnostics: pd.DataFrame


def wrap_angle_deg(values: np.ndarray | Sequence[float] | float) -> np.ndarray:
    """Return circular angle values in [-180, 180)."""

    return (np.asarray(values, dtype=float) + 180.0) % 360.0 - 180.0


def circular_error_deg(estimate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Compute the shortest signed angular difference in degrees."""

    return wrap_angle_deg(np.asarray(estimate) - np.asarray(reference))


def _bus_number(bus: str) -> int:
    match = re.search(r"(\d+)", str(bus))
    if match is None:
        raise ValueError(f"Invalid bus identifier: {bus}")
    return int(match.group(1))


def _bus_sort(values: Iterable[str]) -> list[str]:
    return sorted(values, key=_bus_number)


def _raw_file_for_bus(directory: Path, bus: str) -> Path:
    number = _bus_number(bus)
    candidates = sorted(directory.glob(f"Bus{number}_*.csv"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one CSV for {bus} in {directory}, found {candidates}")
    return candidates[0]


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _positive_sequence_from_phase_a(frame: pd.DataFrame, bus: str, base_kv: float, base_mva: float) -> tuple[np.ndarray, np.ndarray]:
    """Use the balanced phase-A representation exported by the Julia model."""

    number = _bus_number(bus)
    voltage_base = base_kv * 1000.0 / math.sqrt(3.0)
    current_base = base_mva * 1e6 / (math.sqrt(3.0) * base_kv * 1000.0)
    v_mag = frame[f"BUS{number}_VA_MAG"].to_numpy(dtype=float)
    v_ang = np.deg2rad(frame[f"BUS{number}_VA_ANG"].to_numpy(dtype=float))
    i_mag = frame[f"BUS{number}_IA_MAG"].to_numpy(dtype=float)
    i_ang = np.deg2rad(frame[f"BUS{number}_IA_ANG"].to_numpy(dtype=float))
    return (
        (v_mag / voltage_base) * np.exp(1j * v_ang),
        (i_mag / current_base) * np.exp(1j * i_ang),
    )


def _load_reference(
    raw_dir: Path,
    audit: Mapping[str, Any],
    buses: Sequence[str] = ALL_BUSES,
    include_event: bool = True,
) -> ReferenceData:
    selected_buses = tuple(buses)
    frames: dict[str, pd.DataFrame] = {}
    for bus in selected_buses:
        frame = pd.read_csv(_raw_file_for_bus(raw_dir, bus))
        if not include_event:
            frame = frame.drop(columns=["Event"], errors="ignore")
        frames[bus] = frame

    timestamps = frames[selected_buses[0]]["TIMESTAMP"].to_numpy(dtype=float)
    if any(not np.array_equal(frame["TIMESTAMP"].to_numpy(dtype=float), timestamps) for frame in frames.values()):
        raise ValueError("Reference timestamps are not identical across buses")

    buses_by_number = {int(row["bus"]): row for row in audit["buses"]}
    base_kv = np.asarray([float(buses_by_number[_bus_number(bus)]["base_kv"]) for bus in selected_buses], dtype=float)
    voltage_pu = np.zeros((len(timestamps), len(selected_buses)), dtype=complex)
    current_pu = np.zeros_like(voltage_pu)
    voltage_v = np.zeros_like(voltage_pu.real)
    current_a = np.zeros_like(voltage_v)
    frequency = np.zeros_like(voltage_v)
    rocof = np.zeros_like(voltage_v)
    event = frames["BUS1"]["Event"].to_numpy(dtype=int) if include_event and "BUS1" in frames else np.zeros(len(timestamps), dtype=int)

    for index, bus in enumerate(selected_buses):
        frame = frames[bus]
        v_pu, i_pu = _positive_sequence_from_phase_a(frame, bus, base_kv[index], BASE_MVA)
        voltage_pu[:, index] = v_pu
        current_pu[:, index] = i_pu
        voltage_v[:, index] = frame[f"{bus}_VA_MAG"].to_numpy(dtype=float)
        current_a[:, index] = frame[f"{bus}_IA_MAG"].to_numpy(dtype=float)
        frequency[:, index] = frame[f"{bus}_Freq"].to_numpy(dtype=float)
        rocof[:, index] = frame[f"{bus}_ROCOF"].to_numpy(dtype=float)

    derivative_times = np.arange(len(timestamps), dtype=float) / SAMPLE_RATE_HZ
    return ReferenceData(
        buses=selected_buses,
        timestamps=timestamps,
        derivative_times=derivative_times,
        frames=frames,
        voltage_pu=voltage_pu,
        current_pu=current_pu,
        voltage_v=voltage_v,
        current_a=current_a,
        frequency_hz=frequency,
        rocof_hz_s=rocof,
        event=event,
        base_kv=base_kv,
    )


def build_ybus_from_audit(audit: Mapping[str, Any], bus_count: int = 39) -> np.ndarray:
    """Build the IEEE39 Ybus using PowerDynamics' PiLine convention.

    PowerDynamics defines ``V1 = r_src * Vsrc`` and ``V2 = r_dst * Vdst``;
    therefore a transformer uses ``r_src^2`` on the source diagonal and
    ``r_src*r_dst`` in the off-diagonal.  This differs from the more common
    inverse-tap textbook convention and is important for exact consistency.
    """

    ybus = np.zeros((bus_count, bus_count), dtype=complex)
    for branch in audit["branches"]:
        src = int(branch["src_bus"]) - 1
        dst = int(branch["dst_bus"]) - 1
        z = complex(float(branch["R"]), float(branch["X"]))
        if abs(z) == 0.0:
            raise ValueError(f"Zero series impedance at branch {branch}")
        y_series = 1.0 / z
        r_src = float(branch.get("r_src", 1.0) or 1.0)
        r_dst = float(branch.get("r_dst", 1.0) or 1.0)
        y_src = complex(float(branch.get("G_src", 0.0)), float(branch.get("B_src", 0.0)))
        y_dst = complex(float(branch.get("G_dst", 0.0)), float(branch.get("B_dst", 0.0)))
        ybus[src, src] += (y_series + y_src) * r_src * r_src
        ybus[dst, dst] += (y_series + y_dst) * r_dst * r_dst
        ybus[src, dst] -= y_series * r_src * r_dst
        ybus[dst, src] -= y_series * r_src * r_dst
    return ybus


def _load_admittances(audit: Mapping[str, Any], bus_count: int = 39) -> np.ndarray:
    """Approximate nominal constant-impedance load admittances from model data."""

    result = np.zeros(bus_count, dtype=complex)
    for row in audit.get("loads", []):
        bus = int(row["bus"]) - 1
        s_nominal = complex(float(row["Pset"]), float(row["Qset"]))
        # The model uses a 1 pu voltage normalization for the ZIP load
        # parameter. The estimator does not inspect any hidden CSV here.
        result[bus] = np.conj(s_nominal)
    return result


def _physics_rows(audit: Mapping[str, Any], ybus: np.ndarray, observed_indices: set[int]) -> np.ndarray:
    rows: list[np.ndarray] = []
    buses = {int(row["bus"]): row for row in audit["buses"]}
    loads = _load_admittances(audit, ybus.shape[0])
    for number in range(1, ybus.shape[0] + 1):
        index = number - 1
        row = buses[number]
        if index in observed_indices or bool(row.get("has_gen", False)):
            continue
        physics_row = ybus[index, :].copy()
        if bool(row.get("has_load", False)):
            physics_row[index] -= loads[index]
        rows.append(physics_row)
    return np.vstack(rows) if rows else np.zeros((0, ybus.shape[0]), dtype=complex)


def _complex_to_real(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.block([[A.real, -A.imag], [A.imag, A.real]]),
        np.concatenate([b.real, b.imag]),
    )


def _phase_derivatives(angles_rad: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unwrapped = np.unwrap(angles_rad, axis=0)
    frequency = NOMINAL_FREQUENCY_HZ + np.gradient(unwrapped, times, axis=0) / (2.0 * np.pi)
    rocof = np.gradient(frequency, times, axis=0)
    return frequency, rocof


def _observed_change_gate(voltage_pu: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """Find a candidate transient interval from sparse voltage only.

    This is an estimator-side change detector, not the hidden ``Event``
    column.  It is used only to relax alpha sparsity during the interval in
    which the observed PMUs show a coherent step.
    """

    magnitude = np.abs(voltage_pu)
    edge_score = np.mean(
        np.abs(np.diff(magnitude, axis=0)) / np.maximum(magnitude[:-1], 1e-9),
        axis=1,
    )
    order = np.argsort(edge_score)[::-1]
    selected: list[int] = []
    minimum_gap = max(2, int(SAMPLE_RATE_HZ * 1.0))
    for candidate in order:
        if all(abs(int(candidate) - item) >= minimum_gap for item in selected):
            selected.append(int(candidate))
        if len(selected) == 2:
            break
    if len(selected) < 2:
        return np.zeros(len(timestamps), dtype=bool)
    selected.sort()
    start = selected[0] + 1
    end = selected[1] + 1
    gate = np.zeros(len(timestamps), dtype=bool)
    gate[start : min(end + 1, len(gate))] = True
    return gate


def _covariance_derived(
    x: np.ndarray,
    covariance: np.ndarray,
    ybus: np.ndarray,
    base_kv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(x)
    sigma_v_mag = np.zeros(n, dtype=float)
    sigma_v_ang_deg = np.zeros(n, dtype=float)
    sigma_i_mag = np.zeros(n, dtype=float)
    sigma_i_ang_deg = np.zeros(n, dtype=float)
    for i in range(n):
        vr, vi = float(x[i].real), float(x[i].imag)
        magnitude = max(math.hypot(vr, vi), 1e-12)
        cov_v = covariance[np.ix_([i, n + i], [i, n + i])]
        grad_mag = np.asarray([vr / magnitude, vi / magnitude])
        grad_ang = np.asarray([-vi / magnitude**2, vr / magnitude**2])
        sigma_v_mag[i] = math.sqrt(max(float(grad_mag @ cov_v @ grad_mag), 0.0))
        sigma_v_ang_deg[i] = math.degrees(math.sqrt(max(float(grad_ang @ cov_v @ grad_ang), 0.0)))

        row = ybus[i, :]
        j_real = np.r_[row.real, -row.imag]
        j_imag = np.r_[row.imag, row.real]
        cov_i = np.asarray(
            [j_real @ covariance @ j_real, j_real @ covariance @ j_imag,
             j_imag @ covariance @ j_real, j_imag @ covariance @ j_imag]
        ).reshape(2, 2)
        current = row @ x
        cr, ci = float(current.real), float(current.imag)
        current_mag = max(abs(current), 1e-12)
        grad_i_mag = np.asarray([cr / current_mag, ci / current_mag])
        grad_i_ang = np.asarray([-ci / current_mag**2, cr / current_mag**2])
        sigma_i_mag[i] = math.sqrt(max(float(grad_i_mag @ cov_i @ grad_i_mag), 0.0))
        sigma_i_ang_deg[i] = math.degrees(math.sqrt(max(float(grad_i_ang @ cov_i @ grad_i_ang), 0.0)))

    voltage_base = base_kv * 1000.0 / math.sqrt(3.0)
    current_base = BASE_MVA * 1e6 / (math.sqrt(3.0) * base_kv * 1000.0)
    return sigma_v_mag * voltage_base, sigma_v_ang_deg, sigma_i_mag * current_base, sigma_i_ang_deg


def _estimate_timeseries(
    timestamps: np.ndarray,
    voltage_pu: np.ndarray,
    current_pu: np.ndarray,
    ybus: np.ndarray,
    physics_rows: np.ndarray,
    observed_indices: Sequence[int],
    base_kv: np.ndarray,
    voltage_weight: float = 1.0,
    current_weight: float = 0.25,
    physics_weight: float = 0.1,
    prior_weight: float = 0.01,
    temporal_weight: float = 0.1,
) -> EstimateData:
    n_times = voltage_pu.shape[0]
    n_buses = len(base_kv)
    observed = np.asarray(observed_indices, dtype=int)
    selector = np.zeros((len(observed), n_buses), dtype=complex)
    selector[np.arange(len(observed)), observed] = 1.0
    prior = np.ones(n_buses, dtype=complex)

    # Estimate a model-only static prior from the first sparse frame. It is
    # based on sparse measurements plus network equations, never on hidden GT.
    first_A = np.vstack(
        [
            voltage_weight * selector,
            current_weight * ybus[observed, :],
            physics_weight * physics_rows,
            math.sqrt(prior_weight) * np.eye(n_buses),
        ]
    )
    first_b = np.concatenate(
        [
            voltage_weight * voltage_pu[0, :],
            current_weight * current_pu[0, :],
            np.zeros(len(physics_rows), dtype=complex),
            math.sqrt(prior_weight) * prior,
        ]
    )
    prior = np.linalg.lstsq(first_A, first_b, rcond=None)[0]
    previous = prior.copy()

    states = np.zeros((n_times, n_buses), dtype=complex)
    sigma_v = np.zeros((n_times, n_buses), dtype=float)
    sigma_va = np.zeros_like(sigma_v)
    sigma_i = np.zeros_like(sigma_v)
    sigma_ia = np.zeros_like(sigma_v)
    diagnostics: list[dict[str, Any]] = []

    for time_index in range(n_times):
        A = np.vstack(
            [
                voltage_weight * selector,
                current_weight * ybus[observed, :],
                physics_weight * physics_rows,
                math.sqrt(prior_weight) * np.eye(n_buses),
                math.sqrt(temporal_weight) * np.eye(n_buses),
            ]
        )
        b = np.concatenate(
            [
                voltage_weight * voltage_pu[time_index, :],
                current_weight * current_pu[time_index, :],
                np.zeros(len(physics_rows), dtype=complex),
                math.sqrt(prior_weight) * prior,
                math.sqrt(temporal_weight) * previous,
            ]
        )
        state = np.linalg.lstsq(A, b, rcond=None)[0]
        residual = A @ state - b
        real_A, real_b = _complex_to_real(A, b)
        normal = real_A.T @ real_A
        try:
            covariance = np.linalg.pinv(normal, rcond=1e-10)
        except np.linalg.LinAlgError:
            covariance = np.eye(2 * n_buses) * 1e-6
        degrees_of_freedom = max(2 * len(b) - 2 * n_buses, 1)
        sigma2 = float(np.sum(np.abs(residual) ** 2) / degrees_of_freedom)
        covariance *= max(sigma2, 1e-12)
        sv, sva, si, sia = _covariance_derived(state, covariance, ybus, base_kv)
        states[time_index] = state
        sigma_v[time_index] = sv
        sigma_va[time_index] = sva
        sigma_i[time_index] = si
        sigma_ia[time_index] = sia
        diagnostics.append(
            {
                "TIMESTAMP": float(timestamps[time_index]),
                "FRAME_INDEX": time_index,
                "N_OBSERVED_PMUS": len(observed),
                "MEASUREMENT_ROWS": int(len(b)),
                "RESIDUAL_NORM": float(np.sqrt(np.mean(np.abs(residual) ** 2))),
                "MATRIX_CONDITION": float(np.linalg.cond(normal)),
                "POSTERIOR_SIGMA2": sigma2,
                "ESTIMATOR_SEES_GROUND_TRUTH": False,
                "ESTIMATOR_SEES_EVENT_LABEL": False,
            }
        )
        previous = state

    frequency, rocof = _phase_derivatives(np.angle(states), np.arange(n_times) / SAMPLE_RATE_HZ)
    return EstimateData(
        timestamps=timestamps,
        voltage_pu=states,
        current_pu=(states @ ybus.T),
        voltage_v=np.abs(states) * (base_kv[None, :] * 1000.0 / math.sqrt(3.0)),
        current_a=np.abs(states @ ybus.T) * (BASE_MVA * 1e6 / (math.sqrt(3.0) * base_kv[None, :] * 1000.0)),
        frequency_hz=frequency,
        rocof_hz_s=rocof,
        alpha_estimates=np.zeros((n_times, n_buses), dtype=float),
        sigma_voltage_v=sigma_v,
        sigma_voltage_angle_deg=sigma_va,
        sigma_current_a=sigma_i,
        sigma_current_angle_deg=sigma_ia,
        sigma_frequency_hz=np.zeros_like(sigma_v),
        sigma_rocof_hz_s=np.zeros_like(sigma_v),
        diagnostics=pd.DataFrame(diagnostics),
    )


def _load_candidate_indices(audit: Mapping[str, Any], n_buses: int) -> tuple[np.ndarray, np.ndarray]:
    """Return static-load buses and their nominal admittances.

    Generator injections are deliberately excluded from the latent-load
    parameterization.  A bus with both a generator and a load is also left
    free because its net injection is not identifiable from a load-only
    multiplier without adding a generator model.
    """

    buses = {int(row["bus"]): row for row in audit["buses"]}
    admittances = _load_admittances(audit, n_buses)
    candidates = [
        number - 1
        for number in range(1, n_buses + 1)
        if bool(buses[number].get("has_load", False)) and not bool(buses[number].get("has_gen", False))
    ]
    return np.asarray(candidates, dtype=int), admittances


def _latent_alpha_real_system(
    ybus: np.ndarray,
    load_indices: np.ndarray,
    zero_injection_indices: np.ndarray,
    generator_indices: np.ndarray,
    load_admittances: np.ndarray,
    observed_indices: np.ndarray,
    observed_voltage: np.ndarray,
    phase_increment_targets: np.ndarray | None,
    x_reference: np.ndarray,
    alpha_reference: np.ndarray,
    previous_state: np.ndarray,
    previous_alpha: np.ndarray,
    state_anchor: np.ndarray,
    *,
    physics_weight: float,
    voltage_weight: float,
    state_prior_weight: float,
    temporal_state_weight: float,
    alpha_sparsity_weight: float,
    alpha_temporal_weight: float,
    generator_temporal_weight: float,
    dynamic_phase_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assemble one Gauss--Newton linearization with real alpha variables.

    The nonlinear term ``alpha * y_load * V`` is linearized around the
    previous state.  Voltage observations are the only PMU measurements used
    in the core solve; measured current remains an OOD validation signal.
    """

    n_buses = ybus.shape[0]
    n_alpha = len(load_indices)
    selector = np.zeros((len(observed_indices), n_buses), dtype=complex)
    selector[np.arange(len(observed_indices)), observed_indices] = 1.0

    complex_x_rows: list[np.ndarray] = []
    complex_alpha_rows: list[np.ndarray] = []
    complex_rhs: list[complex] = []
    direct_x_rows: list[np.ndarray] = []
    direct_alpha_rows: list[np.ndarray] = []
    direct_rhs: list[complex] = []

    def add_complex_row(row_x: np.ndarray, row_alpha: np.ndarray, rhs: complex, weight: float) -> None:
        complex_x_rows.append(np.asarray(row_x, dtype=complex) * weight)
        complex_alpha_rows.append(np.asarray(row_alpha, dtype=complex) * weight)
        complex_rhs.append(complex(rhs) * weight)

    for row_index, bus_index in enumerate(observed_indices):
        alpha_row = np.zeros(n_alpha, dtype=complex)
        add_complex_row(selector[row_index], alpha_row, observed_voltage[row_index], voltage_weight)

    if phase_increment_targets is not None:
        for row_index, bus_index in enumerate(observed_indices):
            reference_voltage = previous_state[bus_index]
            if abs(reference_voltage) < 1e-9:
                continue
            reciprocal = 1.0 / reference_voltage
            angle_row = np.zeros(2 * n_buses, dtype=complex)
            # i*Im((x - x_prev)/x_prev) is a complex row whose imaginary
            # component is the measured phase increment.  The target uses
            # both sparse PMU frequency and ROCOF.
            angle_row[bus_index] = 1j * reciprocal.imag
            angle_row[n_buses + bus_index] = 1j * reciprocal.real
            direct_x_rows.append(angle_row * dynamic_phase_weight)
            direct_alpha_rows.append(np.zeros(n_alpha, dtype=complex))
            direct_rhs.append(1j * float(phase_increment_targets[row_index]) * dynamic_phase_weight)

    load_set = set(int(index) for index in load_indices)
    load_position = {int(index): position for position, index in enumerate(load_indices)}
    for bus_index in range(n_buses):
        alpha_row = np.zeros(n_alpha, dtype=complex)
        if bus_index in load_set:
            position = load_position[bus_index]
            y_load = load_admittances[bus_index]
            # PowerDynamics' ZIP export uses the load P/Q setpoint as a
            # constant-power complex injection at this stage.  In current
            # form this is YV = (1+alpha)*conj(S0)/conj(V).  Linearize the
            # equivalent power equation S(V)=V*conj(YV), which also makes
            # the requested P/Q-preserving alpha explicit.
            row_y = ybus[bus_index]
            v_ref = x_reference[bus_index]
            i_ref = row_y @ x_reference
            s_ref = v_ref * np.conj(i_ref)
            s_nominal = np.conj(y_load)
            jacobian = np.zeros(2 * n_buses, dtype=complex)
            jacobian[:n_buses] = v_ref * np.conj(row_y)
            jacobian[n_buses:] = -1j * v_ref * np.conj(row_y)
            jacobian[bus_index] += np.conj(i_ref)
            jacobian[n_buses + bus_index] += 1j * np.conj(i_ref)
            direct_alpha = np.zeros(n_alpha, dtype=complex)
            direct_alpha[position] = -s_nominal
            state_reference_real = np.r_[x_reference.real, x_reference.imag]
            rhs = -s_ref + jacobian @ state_reference_real + s_nominal
            row_scale = max(float(np.linalg.norm(jacobian)), float(abs(s_nominal)), 1.0)
            direct_x_rows.append(jacobian / row_scale * physics_weight)
            direct_alpha_rows.append(direct_alpha / row_scale * physics_weight)
            direct_rhs.append(rhs / row_scale * physics_weight)
            continue
        elif bus_index in set(int(index) for index in zero_injection_indices):
            # Non-generator, non-load buses are zero-injection junctions.
            row_x = ybus[bus_index].copy()
            rhs = 0.0j
        else:
            # Generator injections and generator+load net injections are
            # intentionally latent and receive no pseudo-measurement row.
            continue
        row_scale = max(float(np.linalg.norm(row_x)), float(np.linalg.norm(alpha_row)), 1.0)
        add_complex_row(row_x / row_scale, alpha_row / row_scale, rhs / row_scale, physics_weight)

    # Generator injections are not known pseudo-measurements.  A weak
    # temporal prior on their *apparent power* prevents an unobserved load
    # change from being explained for free by an arbitrary generator jump.
    # The prior is deliberately soft because generator electromechanical
    # dynamics are not part of this E0 model.
    previous_generator_power = np.asarray(
        [previous_state[index] * np.conj(ybus[index] @ previous_state) for index in generator_indices],
        dtype=complex,
    )
    for generator_position, bus_index in enumerate(generator_indices):
        v_ref = x_reference[bus_index]
        row_y = ybus[bus_index]
        i_ref = row_y @ x_reference
        s_ref = v_ref * np.conj(i_ref)
        jacobian = np.zeros(2 * n_buses, dtype=complex)
        jacobian[:n_buses] = v_ref * np.conj(row_y)
        jacobian[n_buses:] = -1j * v_ref * np.conj(row_y)
        jacobian[bus_index] += np.conj(i_ref)
        jacobian[n_buses + bus_index] += 1j * np.conj(i_ref)
        rhs = previous_generator_power[generator_position] - s_ref + jacobian @ np.r_[x_reference.real, x_reference.imag]
        row_scale = max(float(np.linalg.norm(jacobian)), 1.0)
        direct_x_rows.append(jacobian / row_scale * math.sqrt(generator_temporal_weight))
        direct_alpha_rows.append(np.zeros(n_alpha, dtype=complex))
        direct_rhs.append(rhs / row_scale * math.sqrt(generator_temporal_weight))

    identity_x = np.eye(n_buses, dtype=complex)
    for row_index in range(n_buses):
        add_complex_row(
            identity_x[row_index],
            np.zeros(n_alpha, dtype=complex),
            state_anchor[row_index],
            math.sqrt(state_prior_weight),
        )
        add_complex_row(
            identity_x[row_index],
            np.zeros(n_alpha, dtype=complex),
            previous_state[row_index],
            math.sqrt(temporal_state_weight),
        )

    complex_x = np.vstack(complex_x_rows)
    complex_alpha = np.vstack(complex_alpha_rows)
    complex_b = np.asarray(complex_rhs, dtype=complex)
    real_x = np.block([[complex_x.real, -complex_x.imag], [complex_x.imag, complex_x.real]])
    real_x = np.hstack([real_x, np.zeros((real_x.shape[0], n_alpha), dtype=float)])
    real_alpha = np.vstack([complex_alpha.real, complex_alpha.imag])
    real_b = np.concatenate([complex_b.real, complex_b.imag])
    if direct_x_rows:
        direct_x = np.vstack(direct_x_rows)
        direct_alpha = np.vstack(direct_alpha_rows)
        direct_b = np.asarray(direct_rhs, dtype=complex)
        direct_real_x = np.vstack([direct_x.real, direct_x.imag])
        direct_real_alpha = np.vstack([direct_alpha.real, direct_alpha.imag])
        direct_real_b = np.concatenate([direct_b.real, direct_b.imag])
        direct_matrix = np.hstack([direct_real_x, direct_real_alpha])
        real_x = np.vstack([real_x, direct_matrix])
        real_b = np.concatenate([real_b, direct_real_b])

    # IRLS is a convex surrogate for the alpha L1/group-sparse penalty.  The
    # small epsilon prevents a zero alpha from becoming an infinite hard row.
    alpha_epsilon = 0.02
    sparsity_rows = np.zeros((n_alpha, 2 * n_buses + n_alpha), dtype=float)
    sparsity_rows[:, 2 * n_buses:] = np.diag(
        np.sqrt(alpha_sparsity_weight / (np.abs(alpha_reference) + alpha_epsilon))
    )
    temporal_rows = np.zeros_like(sparsity_rows)
    temporal_rows[:, 2 * n_buses:] = np.eye(n_alpha) * math.sqrt(alpha_temporal_weight)
    real_a = np.vstack([real_x, sparsity_rows, temporal_rows])
    real_rhs = np.concatenate(
        [real_b, np.zeros(n_alpha, dtype=float), math.sqrt(alpha_temporal_weight) * previous_alpha]
    )

    lower = np.r_[np.full(2 * n_buses, -np.inf), np.full(n_alpha, LOAD_ALPHA_MIN)]
    upper = np.r_[np.full(2 * n_buses, np.inf), np.full(n_alpha, LOAD_ALPHA_MAX)]
    return real_a, real_rhs, lower, upper, complex_x


def _estimate_timeseries_latent_alpha(
    timestamps: np.ndarray,
    voltage_pu: np.ndarray,
    ybus: np.ndarray,
    audit: Mapping[str, Any],
    observed_indices: Sequence[int],
    base_kv: np.ndarray,
    observed_frequency_hz: np.ndarray | None = None,
    observed_rocof_hz_s: np.ndarray | None = None,
    *,
    voltage_weight: float = 100.0,
    physics_weight: float = 100.0,
    state_prior_weight: float = 0.0001,
    temporal_state_weight: float = 0.005,
    alpha_sparsity_weight: float = 0.0005,
    alpha_temporal_weight: float = 0.001,
    generator_temporal_weight: float = 0.1,
    dynamic_phase_weight: float = 10.0,
    linearization_steps: int = 3,
) -> EstimateData:
    """Estimate all voltages and latent load multipliers from sparse V PMUs.

    This is the E0-R model.  It knows the *class* of event (load multiplier)
    through its parameterization, but it does not receive the event bus,
    event magnitude, event timestamps, hidden PMUs, or hidden event labels.
    """

    n_times = voltage_pu.shape[0]
    n_buses = ybus.shape[0]
    observed = np.asarray(observed_indices, dtype=int)
    load_indices, load_admittances = _load_candidate_indices(audit, n_buses)
    observed_change_gate = _observed_change_gate(voltage_pu, timestamps)
    buses = {int(row["bus"]): row for row in audit["buses"]}
    generator_indices = np.asarray(
        [number - 1 for number in range(1, n_buses + 1) if bool(buses[number].get("has_gen", False))],
        dtype=int,
    )
    zero_injection_indices = np.asarray(
        [
            number - 1
            for number in range(1, n_buses + 1)
            if not bool(buses[number].get("has_load", False))
            and not bool(buses[number].get("has_gen", False))
        ],
        dtype=int,
    )
    n_alpha = len(load_indices)
    state_anchor = np.ones(n_buses, dtype=complex)
    previous_state = state_anchor.copy()
    previous_alpha = np.zeros(n_alpha, dtype=float)

    states = np.zeros((n_times, n_buses), dtype=complex)
    alphas = np.zeros((n_times, n_buses), dtype=float)
    sigma_v = np.zeros((n_times, n_buses), dtype=float)
    sigma_va = np.zeros_like(sigma_v)
    sigma_i = np.zeros_like(sigma_v)
    sigma_ia = np.zeros_like(sigma_v)
    diagnostics: list[dict[str, Any]] = []

    for time_index in range(n_times):
        x_reference = previous_state.copy()
        alpha_reference = previous_alpha.copy()
        state = previous_state.copy()
        alpha = previous_alpha.copy()
        covariance = np.eye(2 * n_buses + n_alpha, dtype=float)
        real_a = np.eye(2 * n_buses + n_alpha, dtype=float)
        real_b = np.zeros(2 * n_buses + n_alpha, dtype=float)
        for iteration in range(linearization_steps):
            phase_targets = None
            if time_index > 0 and observed_frequency_hz is not None and observed_rocof_hz_s is not None:
                dt = 1.0 / SAMPLE_RATE_HZ
                phase_targets = 2.0 * np.pi * (
                    (observed_frequency_hz[time_index] - NOMINAL_FREQUENCY_HZ) * dt
                    + 0.5 * observed_rocof_hz_s[time_index] * dt * dt
                )
            real_a, real_b, lower, upper, _ = _latent_alpha_real_system(
                ybus=ybus,
                load_indices=load_indices,
                zero_injection_indices=zero_injection_indices,
                generator_indices=generator_indices,
                load_admittances=load_admittances,
                observed_indices=observed,
                observed_voltage=voltage_pu[time_index],
                phase_increment_targets=phase_targets,
                x_reference=x_reference,
                alpha_reference=alpha_reference,
                previous_state=previous_state,
                previous_alpha=previous_alpha,
                state_anchor=state_anchor,
                physics_weight=physics_weight,
                voltage_weight=voltage_weight,
                state_prior_weight=state_prior_weight,
                temporal_state_weight=temporal_state_weight,
                alpha_sparsity_weight=alpha_sparsity_weight
                * (2.0 if time_index == 0 else 1.0)
                * (0.05 if observed_change_gate[time_index] else 1.0),
                alpha_temporal_weight=alpha_temporal_weight * (0.1 if observed_change_gate[time_index] else 1.0),
                generator_temporal_weight=generator_temporal_weight,
                dynamic_phase_weight=dynamic_phase_weight,
            )
            solved = lsq_linear(real_a, real_b, bounds=(lower, upper), lsmr_tol="auto", verbose=0)
            vector = solved.x
            state = vector[:n_buses] + 1j * vector[n_buses : 2 * n_buses]
            alpha = vector[2 * n_buses:]
            # Damping avoids a single poorly conditioned frame moving the
            # linearization point too far before the next Gauss--Newton step.
            if iteration + 1 < linearization_steps:
                x_reference = 0.5 * x_reference + 0.5 * state
                alpha_reference = 0.5 * alpha_reference + 0.5 * alpha

        residual = real_a @ solved.x - real_b
        normal = real_a.T @ real_a
        try:
            covariance = np.linalg.pinv(normal, rcond=1e-10)
        except np.linalg.LinAlgError:
            covariance = np.eye(real_a.shape[1], dtype=float) * 1e-6
        degrees_of_freedom = max(len(real_b) - len(solved.x), 1)
        sigma2 = float(np.sum(residual**2) / degrees_of_freedom)
        covariance *= max(sigma2, 1e-12)
        state_covariance = covariance[: 2 * n_buses, : 2 * n_buses]
        sv, sva, si, sia = _covariance_derived(state, state_covariance, ybus, base_kv)
        states[time_index] = state
        alphas[time_index, load_indices] = alpha
        sigma_v[time_index] = sv
        sigma_va[time_index] = sva
        sigma_i[time_index] = si
        sigma_ia[time_index] = sia
        diagnostics.append(
            {
                "TIMESTAMP": float(timestamps[time_index]),
                "FRAME_INDEX": time_index,
                "N_OBSERVED_PMUS": len(observed),
                "MEASUREMENT_ROWS": int(len(real_b)),
                "RESIDUAL_NORM": float(np.sqrt(np.mean(residual**2))),
                "MATRIX_CONDITION": float(np.linalg.cond(normal)),
                "POSTERIOR_SIGMA2": sigma2,
                "N_ACTIVE_LOAD_ALPHAS": int(np.sum(np.abs(alpha) >= 0.02)),
                "MAX_ABS_LOAD_ALPHA": float(np.max(np.abs(alpha))) if len(alpha) else 0.0,
                "OBSERVED_CHANGE_GATE": bool(observed_change_gate[time_index]),
                "SOLVER_SUCCESS": bool(solved.success),
                "ESTIMATOR_SEES_GROUND_TRUTH": False,
                "ESTIMATOR_SEES_EVENT_LABEL": False,
            }
        )
        previous_state = state
        previous_alpha = alpha
        if time_index == 0:
            # The first sparse frame is the model-only operating-point
            # anchor.  This is an online baseline choice, not a ground-truth
            # or event-label lookup.
            state_anchor = state.copy()

    current = states @ ybus.T
    frequency, rocof = _phase_derivatives(np.angle(states), np.arange(n_times) / SAMPLE_RATE_HZ)
    return EstimateData(
        timestamps=timestamps,
        voltage_pu=states,
        current_pu=current,
        voltage_v=np.abs(states) * (base_kv[None, :] * 1000.0 / math.sqrt(3.0)),
        current_a=np.abs(current) * (BASE_MVA * 1e6 / (math.sqrt(3.0) * base_kv[None, :] * 1000.0)),
        frequency_hz=frequency,
        rocof_hz_s=rocof,
        alpha_estimates=alphas,
        sigma_voltage_v=sigma_v,
        sigma_voltage_angle_deg=sigma_va,
        sigma_current_a=sigma_i,
        sigma_current_angle_deg=sigma_ia,
        sigma_frequency_hz=np.zeros_like(sigma_v),
        sigma_rocof_hz_s=np.zeros_like(sigma_v),
        diagnostics=pd.DataFrame(diagnostics),
    )


def _fill_dynamic_uncertainty(estimate: EstimateData) -> EstimateData:
    sigma_angle_rad = np.deg2rad(estimate.sigma_voltage_angle_deg)
    dt = 1.0 / SAMPLE_RATE_HZ
    sigma_freq = np.zeros_like(sigma_angle_rad)
    sigma_freq[0] = np.sqrt(sigma_angle_rad[0] ** 2 + sigma_angle_rad[1] ** 2) / (2.0 * np.pi * dt)
    sigma_freq[-1] = np.sqrt(sigma_angle_rad[-1] ** 2 + sigma_angle_rad[-2] ** 2) / (2.0 * np.pi * dt)
    sigma_freq[1:-1] = np.sqrt(sigma_angle_rad[:-2] ** 2 + sigma_angle_rad[2:] ** 2) / (4.0 * np.pi * dt)
    sigma_rocof = np.zeros_like(sigma_freq)
    sigma_rocof[0] = np.sqrt(sigma_freq[0] ** 2 + sigma_freq[1] ** 2) / dt
    sigma_rocof[-1] = np.sqrt(sigma_freq[-1] ** 2 + sigma_freq[-2] ** 2) / dt
    sigma_rocof[1:-1] = np.sqrt(sigma_freq[:-2] ** 2 + sigma_freq[2:] ** 2) / (2.0 * dt)
    return replace(estimate, sigma_frequency_hz=sigma_freq, sigma_rocof_hz_s=sigma_rocof)


def _raw_like_frame(bus: str, timestamps: np.ndarray, voltage_pu: np.ndarray, current_pu: np.ndarray, frequency: np.ndarray, rocof: np.ndarray, base_kv: float) -> pd.DataFrame:
    number = _bus_number(bus)
    voltage_base = base_kv * 1000.0 / math.sqrt(3.0)
    current_base = BASE_MVA * 1e6 / (math.sqrt(3.0) * base_kv * 1000.0)
    v = voltage_pu * voltage_base
    i = current_pu * current_base
    va = np.rad2deg(np.angle(voltage_pu))
    ia = np.rad2deg(np.angle(current_pu))
    columns = {
        "TIMESTAMP": timestamps,
        f"BUS{number}_VA_ANG": wrap_angle_deg(va),
        f"BUS{number}_VA_MAG": np.abs(v),
        f"BUS{number}_VB_ANG": wrap_angle_deg(va - 120.0),
        f"BUS{number}_VB_MAG": np.abs(v),
        f"BUS{number}_VC_ANG": wrap_angle_deg(va + 120.0),
        f"BUS{number}_VC_MAG": np.abs(v),
        f"BUS{number}_IA_ANG": wrap_angle_deg(ia),
        f"BUS{number}_IA_MAG": np.abs(i),
        f"BUS{number}_IB_ANG": wrap_angle_deg(ia - 120.0),
        f"BUS{number}_IB_MAG": np.abs(i),
        f"BUS{number}_IC_ANG": wrap_angle_deg(ia + 120.0),
        f"BUS{number}_IC_MAG": np.abs(i),
        f"BUS{number}_Freq": frequency,
        f"BUS{number}_ROCOF": rocof,
        "DATA_PRESENT": np.ones(len(timestamps), dtype=int),
    }
    return pd.DataFrame(columns)


def _uncertainty_frame(bus: str, timestamps: np.ndarray, estimate: EstimateData, index: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TIMESTAMP": timestamps,
            "SIGMA_VA_MAG": estimate.sigma_voltage_v[:, index],
            "SIGMA_VA_ANG": estimate.sigma_voltage_angle_deg[:, index],
            "SIGMA_IA_MAG": estimate.sigma_current_a[:, index],
            "SIGMA_IA_ANG": estimate.sigma_current_angle_deg[:, index],
            "SIGMA_Freq": estimate.sigma_frequency_hz[:, index],
            "SIGMA_ROCOF": estimate.sigma_rocof_hz_s[:, index],
            "BUS": bus,
        }
    )


def _copy_ground_truth_and_sparse(reference_dir: Path, ground_truth_dir: Path, sparse_dir: Path) -> None:
    ground_truth_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    for bus in ALL_BUSES:
        source = _raw_file_for_bus(reference_dir, bus)
        shutil.copy2(source, ground_truth_dir / source.name)
    for bus in OBSERVED_BUSES:
        source = _raw_file_for_bus(reference_dir, bus)
        sparse_frame = pd.read_csv(source).drop(columns=["Event"], errors="ignore")
        sparse_frame.to_csv(sparse_dir / source.name, index=False)


def _comparison_frame(bus: str, reference: ReferenceData, estimate: EstimateData, index: int) -> pd.DataFrame:
    number = _bus_number(bus)
    ref = reference.frames[bus]
    est = _raw_like_frame(
        bus,
        reference.timestamps,
        estimate.voltage_pu[:, index],
        estimate.current_pu[:, index],
        estimate.frequency_hz[:, index],
        estimate.rocof_hz_s[:, index],
        reference.base_kv[index],
    )
    result: dict[str, Any] = {"TIMESTAMP": reference.timestamps, "REF_Event": reference.event}
    pairs = {
        "VA_MAG": (ref[f"BUS{number}_VA_MAG"].to_numpy(float), est[f"BUS{number}_VA_MAG"].to_numpy(float), "linear"),
        "VA_ANG": (ref[f"BUS{number}_VA_ANG"].to_numpy(float), est[f"BUS{number}_VA_ANG"].to_numpy(float), "angle"),
        "IA_MAG": (ref[f"BUS{number}_IA_MAG"].to_numpy(float), est[f"BUS{number}_IA_MAG"].to_numpy(float), "linear"),
        "IA_ANG": (ref[f"BUS{number}_IA_ANG"].to_numpy(float), est[f"BUS{number}_IA_ANG"].to_numpy(float), "angle"),
        "Freq": (ref[f"BUS{number}_Freq"].to_numpy(float), est[f"BUS{number}_Freq"].to_numpy(float), "linear"),
        "ROCOF": (ref[f"BUS{number}_ROCOF"].to_numpy(float), est[f"BUS{number}_ROCOF"].to_numpy(float), "linear"),
    }
    for signal, (ref_values, est_values, mode) in pairs.items():
        result[f"REF_{signal}"] = ref_values
        result[f"EST_{signal}"] = est_values
        result[f"ERR_{signal}"] = circular_error_deg(est_values, ref_values) if mode == "angle" else est_values - ref_values
    return pd.DataFrame(result)


def _metric(values: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    error = np.asarray(values, dtype=float) - np.asarray(reference, dtype=float)
    return {
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAE": float(np.mean(np.abs(error))),
        "Bias": float(np.mean(error)),
        "MaxAE": float(np.max(np.abs(error))),
    }


def _nrmse(rmse: float, reference: np.ndarray, physical_floor: float = 1.0) -> float:
    scale = float(np.max(reference) - np.min(reference))
    if scale < physical_floor:
        scale = max(float(np.max(np.abs(reference))), physical_floor)
    return float(rmse / scale)


def _shortest_path_distances(audit: Mapping[str, Any], observed: Sequence[str]) -> np.ndarray:
    n = 39
    distances = np.full((n, n), np.inf, dtype=float)
    np.fill_diagonal(distances, 0.0)
    for branch in audit["branches"]:
        i = int(branch["src_bus"]) - 1
        j = int(branch["dst_bus"]) - 1
        z = abs(complex(float(branch["R"]), float(branch["X"])))
        distances[i, j] = min(distances[i, j], z)
        distances[j, i] = min(distances[j, i], z)
    for k in range(n):
        distances = np.minimum(distances, distances[:, [k]] + distances[[k], :])
    observed_indices = [_bus_number(bus) - 1 for bus in observed]
    return np.min(distances[:, observed_indices], axis=1)


def _event_inference(estimate: EstimateData, timestamps: np.ndarray) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Perform E0-I localization from the inferred latent load multipliers."""

    alpha = np.asarray(estimate.alpha_estimates, dtype=float)
    baseline_count = max(10, min(len(timestamps) // 5, len(timestamps) - 1))
    baseline = np.median(alpha[:baseline_count], axis=0)
    centered = alpha - baseline[None, :]
    strength = np.max(np.abs(centered), axis=1)
    edge_score = np.nanmax(np.abs(np.diff(centered, axis=0)), axis=1)
    gate = estimate.diagnostics.get("OBSERVED_CHANGE_GATE") if not estimate.diagnostics.empty else None
    gated_indices = np.flatnonzero(np.asarray(gate, dtype=bool)) if gate is not None else np.asarray([], dtype=int)
    if len(gated_indices):
        onset_index = int(gated_indices[0])
        end_index = int(gated_indices[-1])
    else:
        order = np.argsort(edge_score)[::-1]
        chosen: list[int] = []
        minimum_gap = max(2, int(SAMPLE_RATE_HZ * 1.0))
        for candidate in order:
            if candidate <= 0 or candidate >= len(timestamps) - 2:
                continue
            if all(abs(int(candidate) - item) >= minimum_gap for item in chosen):
                chosen.append(int(candidate))
            if len(chosen) == 2:
                break
        chosen.sort()
        onset_index = chosen[0] if chosen else int(np.argmax(edge_score))
        end_index = chosen[1] if len(chosen) > 1 else min(onset_index + int(5 * SAMPLE_RATE_HZ), len(timestamps) - 1)
    onset = float(timestamps[onset_index])
    end = float(timestamps[end_index])
    middle = (timestamps >= onset + 0.5) & (timestamps < end - 0.5)
    pre = timestamps < onset
    post = timestamps >= end
    scores: list[dict[str, Any]] = []
    for index in range(alpha.shape[1]):
        alpha_pre = float(np.median(centered[pre, index])) if np.any(pre) else float(centered[0, index])
        alpha_mid = float(np.median(centered[middle, index])) if np.any(middle) else alpha_pre
        alpha_post = float(np.median(centered[post, index])) if np.any(post) else alpha_pre
        delta = alpha_mid - alpha_pre
        scores.append(
            {
                "BUS": f"BUS{index + 1}",
                "ALPHA_PRE": alpha_pre,
                "ALPHA_EVENT": alpha_mid,
                "ALPHA_POST": alpha_post,
                "ALPHA_CHANGE": delta,
                "MAX_ABS_ALPHA": float(np.max(np.abs(centered[:, index]))),
                "LOAD_CHANGE_SCORE": abs(delta),
            }
        )
    score_df = pd.DataFrame(scores).sort_values("LOAD_CHANGE_SCORE", ascending=False).reset_index(drop=True)
    winner = score_df.iloc[0].to_dict()
    alpha_event = float(winner["ALPHA_EVENT"])
    multiplier_plausible = LOAD_ALPHA_MIN <= alpha_event <= LOAD_ALPHA_MAX
    alpha_detected = abs(alpha_event) >= 0.02
    inference = {
        "estimator_input": "sparse positive-sequence voltage PMUs plus model audit",
        "reconstruction_gate": "E0-R: oracle event type=load only",
        "inference_gate": "E0-I: bus, magnitude, and interval inferred from latent alpha",
        "ground_truth_used": False,
        "event_label_used": False,
        "estimated_onset_s": onset,
        "estimated_end_s": end,
        "candidate_bus": winner["BUS"],
        "classification": "load_change" if alpha_detected and multiplier_plausible else "change_detected_needs_review",
        "quality_flag": "plausible_multiplier" if alpha_detected and multiplier_plausible else "insufficient_local_alpha_or_model_review",
        "estimated_load_multiplier": 1.0 + alpha_event,
        "event_strength_max_abs_alpha": float(np.max(strength)),
        "top_candidates": score_df.head(5).to_dict(orient="records"),
    }
    return score_df, inference


def _write_per_bus_plot(path: Path, bus: str, comparison: pd.DataFrame) -> None:
    signals = [("VA_MAG", "Voltage magnitude [V]", False), ("VA_ANG", "Voltage angle [deg]", True), ("IA_MAG", "Current magnitude [A]", False), ("IA_ANG", "Current angle [deg]", True), ("Freq", "Frequency [Hz]", False), ("ROCOF", "ROCOF [Hz/s]", False)]
    fig, axes = plt.subplots(len(signals), 2, figsize=(15, 18), sharex=True)
    time = comparison["TIMESTAMP"].to_numpy(float)
    for row, (signal, label, _) in enumerate(signals):
        overlay, residual = axes[row]
        overlay.plot(time, comparison[f"REF_{signal}"], label="Ground truth", color="#005f73", linewidth=1.5)
        overlay.plot(time, comparison[f"EST_{signal}"], label="Estimated", color="#ca6702", linestyle="--", linewidth=1.2)
        residual.plot(time, comparison[f"ERR_{signal}"], color="#9b2226", linewidth=1.1)
        overlay.set_ylabel(label)
        residual.set_ylabel("error")
        overlay.grid(alpha=0.2)
        residual.grid(alpha=0.2)
        for ax in (overlay, residual):
            ax.axvline(5.0, color="red", linestyle="--", linewidth=0.8)
            ax.axvline(10.0, color="blue", linestyle="--", linewidth=0.8)
        if row == 0:
            overlay.legend(loc="best")
            residual.set_title("Residual")
    axes[-1, 0].set_xlabel("Time [s]")
    axes[-1, 1].set_xlabel("Time [s]")
    fig.suptitle(f"{bus} — Ground truth vs estimated virtual PMU", fontsize=16)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_summary_plots(plot_dir: Path, metrics: pd.DataFrame, distances: pd.DataFrame, event_scores: pd.DataFrame) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = metrics[metrics["WINDOW"] == "all"]
    heat = all_metrics.pivot(index="BUS", columns="SIGNAL", values="NRMSE").reindex(index=list(ALL_BUSES), columns=list(SIGNALS))
    heat_values = np.maximum(heat.to_numpy(float), 1e-6)
    vmax = max(float(np.nanpercentile(heat_values, 98.0)), 1e-5)
    fig, ax = plt.subplots(figsize=(12, 12))
    image = ax.imshow(heat_values, aspect="auto", cmap="magma", norm=LogNorm(vmin=1e-6, vmax=vmax))
    ax.set_xticks(np.arange(len(heat.columns)), heat.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(heat.index)), heat.index)
    ax.set_title("E0 NRMSE heatmap — 39 buses x six signals")
    fig.colorbar(image, ax=ax, label="NRMSE (log scale)")
    fig.tight_layout()
    fig.savefig(plot_dir / "nrmse_heatmap_39x6.png", dpi=150)
    plt.close(fig)

    grouped = (
        all_metrics[all_metrics["SIGNAL"].isin(SIGNALS)]
        .groupby(["OBSERVABILITY", "SIGNAL"], as_index=False)["NRMSE"]
        .mean()
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    pivot = grouped.pivot(index="SIGNAL", columns="OBSERVABILITY", values="NRMSE").reindex(SIGNALS)
    pivot.plot(kind="bar", ax=ax, color={"observed": "#0a9396", "unobserved": "#bb3e03"})
    ax.set_title("Signal-wise NRMSE: observed vs unobserved")
    ax.set_ylabel("Mean NRMSE (dimensionless)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(plot_dir / "observed_vs_unobserved_rmse.png", dpi=150)
    plt.close(fig)

    distance_frame = distances.merge(all_metrics[all_metrics["SIGNAL"] == "VA_MAG"][["BUS", "NRMSE"]], on="BUS", how="left")
    fig, ax = plt.subplots(figsize=(7, 4))
    for observed, group in distance_frame.groupby("OBSERVED"):
        ax.scatter(group["DISTANCE_TO_NEAREST_PMU"], group["NRMSE"], label="observed" if observed else "unobserved", alpha=0.85)
    ax.set_xlabel("Electrical distance to nearest observed PMU")
    ax.set_ylabel("Voltage magnitude NRMSE")
    ax.set_title("Reconstruction error vs electrical distance")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "error_vs_electrical_distance.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    top = event_scores.head(15).iloc[::-1]
    ax.barh(top["BUS"], top["LOAD_CHANGE_SCORE"], color="#ee9b00")
    ax.set_title("Event localization score from estimated signals")
    ax.set_xlabel("|P ratio - 1| + |Q ratio - 1|")
    fig.tight_layout()
    fig.savefig(plot_dir / "event_localization_scores.png", dpi=150)
    plt.close(fig)


def _write_alpha_artifacts(
    metrics_dir: Path,
    plot_dir: Path,
    estimate: EstimateData,
    audit: Mapping[str, Any],
) -> None:
    """Persist the latent-load result used by E0-I and its diagnostic plot."""

    load_buses = {
        f"BUS{int(row['bus'])}"
        for row in audit.get("buses", [])
        if bool(row.get("has_load", False)) and not bool(row.get("has_gen", False))
    }
    rows = []
    for time_index, timestamp in enumerate(estimate.timestamps):
        for bus_index, bus in enumerate(ALL_BUSES):
            rows.append(
                {
                    "TIMESTAMP": float(timestamp),
                    "BUS": bus,
                    "ALPHA_EST": float(estimate.alpha_estimates[time_index, bus_index]),
                    "LOAD_CANDIDATE": bus in load_buses,
                }
            )
    pd.DataFrame(rows).to_csv(metrics_dir / "alpha_estimates.csv", index=False)

    matrix = estimate.alpha_estimates.T
    limit = max(0.05, float(np.nanpercentile(np.abs(matrix), 99.0)))
    fig, ax = plt.subplots(figsize=(14, 10))
    image = ax.imshow(
        matrix,
        aspect="auto",
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        extent=[float(estimate.timestamps[0]), float(estimate.timestamps[-1]), len(ALL_BUSES) + 0.5, 0.5],
    )
    ax.set_yticks(np.arange(1, len(ALL_BUSES) + 1), ALL_BUSES)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Bus")
    ax.set_title("E0 latent load multiplier alpha — model-only inference")
    ax.axvline(5.0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(10.0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    fig.colorbar(image, ax=ax, label="alpha (P,Q multiplier = 1 + alpha)")
    fig.tight_layout()
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / "alpha_heatmap_bus_time.png", dpi=150)
    plt.close(fig)


def _evaluate(reference: ReferenceData, estimate: EstimateData, comparison_dir: Path, metrics_dir: Path, plot_dir: Path, audit: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    comparison_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    observed_set = set(OBSERVED_BUSES)
    for index, bus in enumerate(ALL_BUSES):
        comparison = _comparison_frame(bus, reference, estimate, index)
        comparison.to_csv(comparison_dir / f"{bus}_Comparison.csv", index=False)
        _write_per_bus_plot(plot_dir / "per_bus" / f"{bus}_ref_vs_est.png", bus, comparison)
        for window, (start, end, include_end) in {"all": (0.0, 15.0, True), **WINDOWS}.items():
            mask = (reference.timestamps >= start) & ((reference.timestamps <= end) if include_end else (reference.timestamps < end))
            for signal in SIGNALS:
                ref_values = comparison.loc[mask, f"REF_{signal}"].to_numpy(float)
                est_values = comparison.loc[mask, f"EST_{signal}"].to_numpy(float)
                error = circular_error_deg(est_values, ref_values) if signal.endswith("ANG") else est_values - ref_values
                stats = _metric(ref_values + error, ref_values)
                # _metric above is linear; replace angle statistics with circular error.
                if signal.endswith("ANG"):
                    stats = {
                        "RMSE": float(np.sqrt(np.mean(error**2))),
                        "MAE": float(np.mean(np.abs(error))),
                        "Bias": float(np.mean(error)),
                        "MaxAE": float(np.max(np.abs(error))),
                    }
                rows.append(
                    {
                        "BUS": bus,
                        "OBSERVABILITY": "observed" if bus in observed_set else "unobserved",
                        "SIGNAL": signal,
                        "WINDOW": window,
                        **stats,
                        "NRMSE": _nrmse(
                            stats["RMSE"],
                            ref_values,
                            physical_floor={
                                "VA_MAG": 1.0,
                                "VA_ANG": 1.0,
                                "IA_MAG": 1.0,
                                "IA_ANG": 1.0,
                                "Freq": 0.01,
                                "ROCOF": 0.01,
                            }[signal],
                        ),
                    }
                )
        # Phasor errors are reported separately because they combine magnitude and angle.
        for window, (start, end, include_end) in {"all": (0.0, 15.0, True), **WINDOWS}.items():
            mask = (reference.timestamps >= start) & ((reference.timestamps <= end) if include_end else (reference.timestamps < end))
            for phasor_name, ref_phasor, est_phasor in [
                ("V_PHASOR_REL", reference.voltage_pu[:, index], estimate.voltage_pu[:, index]),
                ("I_PHASOR_REL", reference.current_pu[:, index], estimate.current_pu[:, index]),
            ]:
                relative = np.abs(est_phasor[mask] - ref_phasor[mask]) / np.maximum(np.abs(ref_phasor[mask]), 1e-9)
                rows.append(
                    {
                        "BUS": bus,
                        "OBSERVABILITY": "observed" if bus in observed_set else "unobserved",
                        "SIGNAL": phasor_name,
                        "WINDOW": window,
                        "RMSE": float(np.sqrt(np.mean(relative**2))),
                        "MAE": float(np.mean(relative)),
                        "Bias": float(np.mean(relative)),
                        "MaxAE": float(np.max(relative)),
                        "NRMSE": float(np.sqrt(np.mean(relative**2))),
                    }
                )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(metrics_dir / "signal_metrics.csv", index=False)

    distance_values = _shortest_path_distances(audit, OBSERVED_BUSES)
    distances = pd.DataFrame(
        {
            "BUS": list(ALL_BUSES),
            "OBSERVED": [bus in observed_set for bus in ALL_BUSES],
            "DISTANCE_TO_NEAREST_PMU": distance_values,
        }
    )
    distances.to_csv(metrics_dir / "electrical_distance_to_nearest_pmu.csv", index=False)

    coverage_rows: list[dict[str, Any]] = []
    uncertainty_by_signal = {
        "VA_MAG": estimate.sigma_voltage_v,
        "VA_ANG": estimate.sigma_voltage_angle_deg,
        "IA_MAG": estimate.sigma_current_a,
        "IA_ANG": estimate.sigma_current_angle_deg,
        "Freq": estimate.sigma_frequency_hz,
        "ROCOF": estimate.sigma_rocof_hz_s,
    }
    values_by_signal = {
        "VA_MAG": estimate.voltage_v,
        "VA_ANG": np.rad2deg(np.angle(estimate.voltage_pu)),
        "IA_MAG": estimate.current_a,
        "IA_ANG": np.rad2deg(np.angle(estimate.current_pu)),
        "Freq": estimate.frequency_hz,
        "ROCOF": estimate.rocof_hz_s,
    }
    refs_by_signal = {
        "VA_MAG": reference.voltage_v,
        "VA_ANG": np.rad2deg(np.angle(reference.voltage_pu)),
        "IA_MAG": reference.current_a,
        "IA_ANG": np.rad2deg(np.angle(reference.current_pu)),
        "Freq": reference.frequency_hz,
        "ROCOF": reference.rocof_hz_s,
    }
    for index, bus in enumerate(ALL_BUSES):
        for signal in SIGNALS:
            error = circular_error_deg(values_by_signal[signal][:, index], refs_by_signal[signal][:, index]) if signal.endswith("ANG") else values_by_signal[signal][:, index] - refs_by_signal[signal][:, index]
            sigma = uncertainty_by_signal[signal][:, index]
            covered = np.abs(error) <= 1.96 * np.maximum(sigma, 1e-12)
            coverage_rows.append(
                {
                    "BUS": bus,
                    "OBSERVABILITY": "observed" if bus in observed_set else "unobserved",
                    "SIGNAL": signal,
                    "COVERAGE_95": float(np.mean(covered)),
                    "MEAN_SIGMA": float(np.mean(sigma)),
                    "P95_ABS_ERROR": float(np.quantile(np.abs(error), 0.95)),
                }
            )
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(metrics_dir / "coverage_95.csv", index=False)

    event_scores, event_inference = _event_inference(estimate, reference.timestamps)
    event_scores.to_csv(metrics_dir / "event_localization_scores.csv", index=False)
    (metrics_dir / "event_inference.json").write_text(json.dumps(_jsonable(event_inference), indent=2), encoding="utf-8")
    _write_alpha_artifacts(metrics_dir, plot_dir, estimate, audit)

    all_metrics = metrics[metrics["WINDOW"] == "all"]
    observed_metrics = all_metrics[all_metrics["OBSERVABILITY"] == "observed"]
    unobserved_metrics = all_metrics[all_metrics["OBSERVABILITY"] == "unobserved"]
    signal_summary = (
        all_metrics[all_metrics["SIGNAL"].isin(SIGNALS)]
        .groupby(["OBSERVABILITY", "SIGNAL"], as_index=False)[["RMSE", "NRMSE"]]
        .mean()
    )
    summary = {
        "scenario_id": SCENARIO_ID,
        "estimator": "latent_sparse_load_alpha_voltage_state_estimator",
        "reconstruction_gate": "E0-R",
        "inference_gate": "E0-I",
        "core_signals": list(CORE_SIGNALS),
        "current_validation_signals": list(CURRENT_VALIDATION_SIGNALS),
        "bus_count": len(ALL_BUSES),
        "observed_bus_count": len(OBSERVED_BUSES),
        "unobserved_bus_count": len(ALL_BUSES) - len(OBSERVED_BUSES),
        "observed_buses": list(OBSERVED_BUSES),
        "unobserved_buses": [bus for bus in ALL_BUSES if bus not in observed_set],
        "ground_truth_used_during_estimation": False,
        "event_label_used_during_estimation": False,
        "mean_rmse_by_signal_observability": signal_summary.to_dict(orient="records"),
        "mean_voltage_nrmse_unobserved": float(unobserved_metrics[unobserved_metrics["SIGNAL"] == "VA_MAG"]["NRMSE"].mean()),
        "mean_coverage_95_observed": float(coverage[coverage["OBSERVABILITY"] == "observed"]["COVERAGE_95"].mean()),
        "mean_coverage_95_unobserved": float(coverage[coverage["OBSERVABILITY"] == "unobserved"]["COVERAGE_95"].mean()),
        "event_inference": event_inference,
        "latent_alpha": {
            "parameterization": "P(t)=(1+alpha(t))*P0; Q(t)=(1+alpha(t))*Q0",
            "candidate_count": int(np.sum(np.any(np.abs(estimate.alpha_estimates) > 0.0, axis=0))),
            "max_abs_alpha": float(np.max(np.abs(estimate.alpha_estimates))),
            "ground_truth_used_for_inference": False,
        },
    }
    (metrics_dir / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    _write_summary_plots(plot_dir, metrics, distances, event_scores)
    return metrics, coverage, summary


def _write_report(output_dir: Path, summary: Mapping[str, Any], estimate: EstimateData) -> None:
    report_dir = output_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    inference = summary["event_inference"]
    lines = [
        f"# {SCENARIO_ID}",
        "",
        "Sparse-PMU E0 validation on the PowerDynamics IEEE-39 Bus-7 load pulse.",
        "",
        "## Leakage contract",
        "",
        "- Estimator input: only eight files in `input_sparse/`.",
        "- Hidden during inference: the 31 non-PMU buses and every `REF_*` value.",
        "- `ground_truth/` is opened only after estimation, by the evaluator.",
        "- Event labels are not supplied to the estimator.",
        "",
        "## Estimation model",
        "",
        "E0-R uses only positive-sequence voltage PMUs and the model audit. Static-load buses have latent power-factor-preserving multipliers, `P=(1+alpha)P0` and `Q=(1+alpha)Q0`; nominal values are soft sparse priors. Generator-bus injections remain unknown. Current is derived from `YV` only for out-of-model validation.",
        "",
        "The reconstruction gate receives the oracle event class `load` but not its bus, magnitude, or interval. The inference gate then estimates those quantities from the latent alpha trajectory.",
        "",
        "## Results",
        "",
        f"- Observed buses: {', '.join(summary['observed_buses'])}",
        f"- Unobserved buses: {summary['unobserved_bus_count']}",
        f"- Mean unobserved voltage NRMSE: `{summary['mean_voltage_nrmse_unobserved']:.6g}`",
        f"- Mean 95% coverage, observed: `{summary['mean_coverage_95_observed']:.6g}`",
        f"- Mean 95% coverage, unobserved: `{summary['mean_coverage_95_unobserved']:.6g}`",
        "- Metrics remain signal-wise: RMSE/MAE/Bias/MaxAE use native units and NRMSE is dimensionless; no cross-unit score is reported.",
        "",
        "### Signal-wise mean error",
        "",
        "| Observability | Signal | RMSE | NRMSE |",
        "|---|---|---:|---:|",
    ]
    for row in summary["mean_rmse_by_signal_observability"]:
        lines.append(f"| {row['OBSERVABILITY']} | {row['SIGNAL']} | {row['RMSE']:.6g} | {row['NRMSE']:.6g} |")
    lines.extend(
        [
            "",
            "## Event inference from estimated signals",
            "",
            f"- Candidate: `{inference['candidate_bus']}`",
            f"- Classification: `{inference['classification']}`",
            f"- Estimated interval: `{inference['estimated_onset_s']:.3f}`–`{inference['estimated_end_s']:.3f}` s",
            f"- Estimated load multiplier: `{inference['estimated_load_multiplier']:.6g}`",
            f"- Maximum absolute latent alpha: `{inference['event_strength_max_abs_alpha']:.6g}`",
            "",
            "The complete per-bus comparisons, windowed metrics, uncertainty coverage, alpha trajectories/heatmap, event scores, electrical distances, and figures are stored alongside this report.",
        ]
    )
    (report_dir / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_e0_sparse_pmu_experiment(reference_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Run the complete leakage-safe E0 experiment."""

    reference_dir = Path(reference_dir).resolve()
    output_dir = Path(output_dir).resolve()
    audit_path = reference_dir.parent / "01_model_audit" / "model_audit.json"
    if not audit_path.exists():
        raise FileNotFoundError(f"Missing PowerDynamics model audit: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    ground_truth_dir = output_dir / "ground_truth"
    sparse_dir = output_dir / "input_sparse"
    estimated_dir = output_dir / "estimated"
    uncertainty_dir = output_dir / "uncertainty"
    _copy_ground_truth_and_sparse(reference_dir, ground_truth_dir, sparse_dir)

    # The following call receives input_sparse only. Ground truth is not in
    # its signature or data path; evaluation happens in a later phase.
    sparse = _load_reference(sparse_dir, audit, buses=OBSERVED_BUSES, include_event=False)
    ybus = build_ybus_from_audit(audit)
    base_kv_full = np.asarray([float(row["base_kv"]) for row in sorted(audit["buses"], key=lambda row: int(row["bus"]))])
    estimated = _estimate_timeseries_latent_alpha(
        timestamps=sparse.timestamps,
        voltage_pu=sparse.voltage_pu,
        ybus=ybus,
        audit=audit,
        observed_indices=[_bus_number(bus) - 1 for bus in OBSERVED_BUSES],
        base_kv=base_kv_full,
        observed_frequency_hz=sparse.frequency_hz,
        observed_rocof_hz_s=sparse.rocof_hz_s,
    )
    estimated = _fill_dynamic_uncertainty(estimated)

    estimated_dir.mkdir(parents=True, exist_ok=True)
    uncertainty_dir.mkdir(parents=True, exist_ok=True)
    for index, bus in enumerate(ALL_BUSES):
        _raw_like_frame(
            bus,
            sparse.timestamps,
            estimated.voltage_pu[:, index],
            estimated.current_pu[:, index],
            estimated.frequency_hz[:, index],
            estimated.rocof_hz_s[:, index],
            base_kv_full[index],
        ).to_csv(estimated_dir / f"{bus}_Estimated.csv", index=False)
        _uncertainty_frame(bus, sparse.timestamps, estimated, index).to_csv(uncertainty_dir / f"{bus}_Uncertainty.csv", index=False)
    estimated.diagnostics.to_csv(output_dir / "estimator_diagnostics.csv", index=False)

    # Only now, after estimation, is reference data opened for comparison.
    reference = _load_reference(ground_truth_dir, audit)
    metrics, coverage, summary = _evaluate(
        reference=reference,
        estimate=estimated,
        comparison_dir=output_dir / "comparison",
        metrics_dir=output_dir / "metrics",
        plot_dir=output_dir / "plots",
        audit=audit,
    )
    _write_report(output_dir, summary, estimated)
    manifest = {
        "scenario_id": SCENARIO_ID,
        "reference_dir": str(reference_dir),
        "output_dir": str(output_dir),
        "observed_buses": list(OBSERVED_BUSES),
        "unobserved_bus_count": 31,
        "directories": {
            "ground_truth": "ground_truth (evaluation only)",
            "input_sparse": "input_sparse (estimator input)",
            "estimated": "estimated",
            "comparison": "comparison (evaluation only)",
        },
        "estimator_contract": {
            "ground_truth_used_during_estimation": False,
            "event_label_used_during_estimation": False,
            "ref_columns_visible_to_estimator": False,
            "reconstruction_gate": "E0-R: oracle event type=load only",
            "inference_gate": "E0-I: latent alpha bus/magnitude/interval inference",
            "model": "positive-sequence voltage Gauss-Newton WLS + PiLine equations + sparse temporal latent load alpha",
            "dynamic_measurements": "observed sparse PMU frequency and ROCOF constrain observed phase increments",
            "current_semantics": "Ybus@V derived validation only; not used in the core solve",
            "hyperparameters": {
                "voltage_weight": 100.0,
                "physics_weight": 100.0,
                "state_prior_weight": 0.0001,
                "temporal_state_weight": 0.005,
                "alpha_sparsity_weight": 0.0005,
                "alpha_temporal_weight": 0.001,
                "generator_temporal_weight": 0.1,
                "dynamic_phase_weight": 10.0,
                "alpha_bounds": [LOAD_ALPHA_MIN, LOAD_ALPHA_MAX],
                "linearization_steps": 3,
            },
        },
        "timestamps": int(len(sparse.timestamps)),
        "summary": summary,
    }
    (output_dir / "manifest.json").write_text(json.dumps(_jsonable(manifest), indent=2), encoding="utf-8")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the leakage-safe E0 sparse-PMU IEEE39 experiment.")
    parser.add_argument("--reference-dir", type=Path, default=Path("output/SIM_PD39_LOAD_BUS7_PLUS10_RETURN_V1/02_raw_csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/SIM_PD39_SPARSE_PMU_E0"))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary = run_e0_sparse_pmu_experiment(args.reference_dir, args.output_dir)
    print(json.dumps(_jsonable(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
