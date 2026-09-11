"""PMU measurement operators with declared current-terminal and causal timing semantics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

import numpy as np

from pmu_hybrid.constants import CONTROLLED_SYNTHETIC_TERMINAL_MAP, NOMINAL_FREQUENCY_HZ, PMU_BUSES
from pmu_hybrid.physics.network import Branch, branch_terminal_currents


@dataclass(frozen=True)
class PMUTerminal:
    """A frozen, oriented branch terminal used by one controlled PMU channel."""

    pmu_bus: int
    branch_id: str
    terminal: str
    other_bus: int
    polarity: int = 1
    status: str = CONTROLLED_SYNTHETIC_TERMINAL_MAP
    direction: str = "current_leaving_named_pmu_bus_into_selected_branch"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MeasurementBatch:
    """Clean positive-sequence observations plus an explicit availability mask."""

    values: np.ndarray
    data_present: np.ndarray
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        mask = np.asarray(self.data_present, dtype=bool)
        if values.ndim != 3 or values.shape[-1] != 7 or mask.shape != values.shape[:2]:
            raise ValueError("values must have shape (frames, pmus, 7) and mask (frames, pmus)")


def voltage_operator(bus_ids: Iterable[int], voltage: Iterable[complex], pmu_buses: Iterable[int] = PMU_BUSES) -> np.ndarray:
    """Apply the PMU voltage selector y_p = e_p^T V in Cartesian form."""
    ids = tuple(int(bus) for bus in bus_ids)
    vector = np.asarray(tuple(voltage), dtype=complex)
    if vector.shape != (len(ids),):
        raise ValueError("voltage vector does not match bus_ids")
    positions = {bus: index for index, bus in enumerate(ids)}
    missing = [int(bus) for bus in pmu_buses if int(bus) not in positions]
    if missing:
        raise ValueError(f"PMU buses absent from state: {missing}")
    return np.asarray([vector[positions[int(bus)]] for bus in pmu_buses], dtype=complex)


def ideal_state_frequency(angle_rad: Iterable[float], time_s: Iterable[float], *, nominal_frequency_hz: float = NOMINAL_FREQUENCY_HZ) -> tuple[np.ndarray, np.ndarray]:
    """Ideal state-derivative mode, kept distinct from a causal PMU filter.

    The electrical angle derivative is converted to Hz only after explicitly
    declaring the approximation f = f0 + theta_dot/(2*pi). ``np.gradient`` is
    intentionally confined to this offline ideal reference mode.
    """
    angles = np.unwrap(np.asarray(tuple(angle_rad), dtype=float))
    times = np.asarray(tuple(time_s), dtype=float)
    if angles.ndim != 1 or times.shape != angles.shape or len(times) < 2:
        raise ValueError("angle and time must be one-dimensional arrays with at least two samples")
    if np.any(np.diff(times) <= 0):
        raise ValueError("time must be strictly increasing")
    frequency = nominal_frequency_hz + np.gradient(angles, times) / (2.0 * np.pi)
    rocof = np.gradient(frequency, times)
    return frequency, rocof


def build_measurement_batch(voltage: np.ndarray, current: np.ndarray, frequency_hz: np.ndarray, rocof_hz_s: np.ndarray, data_present: np.ndarray, *, metadata: Mapping[str, Any] | None = None) -> MeasurementBatch:
    """Pack clean observations while preserving missingness as a mask."""
    voltage = np.asarray(voltage, dtype=complex)
    current = np.asarray(current, dtype=complex)
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    rocof_hz_s = np.asarray(rocof_hz_s, dtype=float)
    mask = np.asarray(data_present, dtype=bool)
    if voltage.ndim != 2 or any(array.shape != voltage.shape for array in (current, frequency_hz, rocof_hz_s, mask)):
        raise ValueError("all observation arrays must have shape (frames, pmus)")
    values = np.stack((voltage.real, voltage.imag, current.real, current.imag, frequency_hz, rocof_hz_s, mask.astype(float)), axis=-1)
    return MeasurementBatch(values=values, data_present=mask, metadata={"operator": "clean_positive_sequence", **dict(metadata or {})})


def select_controlled_terminal_map(branches: Iterable[Branch], pmu_buses: Iterable[int] = PMU_BUSES) -> dict[int, PMUTerminal]:
    """Choose a deterministic terminal map without inspecting PMU recordings.

    No branch choice is inferred from competition current channels. The smallest
    unordered endpoint pair incident to each PMU is used only for controlled
    synthetic trajectories, and the orientation is always current leaving the
    named PMU bus.
    """
    source = tuple(branches)
    mapping: dict[int, PMUTerminal] = {}
    for bus in tuple(int(item) for item in pmu_buses):
        candidates: list[tuple[tuple[int, int, str], Branch, str, int]] = []
        for branch in source:
            if not branch.in_service or bus not in (branch.from_bus, branch.to_bus):
                continue
            terminal = "from" if branch.from_bus == bus else "to"
            other = branch.to_bus if terminal == "from" else branch.from_bus
            candidates.append(((min(bus, other), max(bus, other), branch.identifier), branch, terminal, other))
        if not candidates:
            raise ValueError(f"PMU bus {bus} has no in-service incident branch")
        _, branch, terminal, other = min(candidates, key=lambda item: item[0])
        mapping[bus] = PMUTerminal(
            pmu_bus=bus, branch_id=branch.identifier, terminal=terminal, other_bus=int(other),
        )
    return mapping


def terminal_current(mapping: PMUTerminal, branch: Branch, v_from: complex, v_to: complex) -> complex:
    """Evaluate one frozen terminal current and reject inconsistent maps."""
    if mapping.branch_id != branch.identifier:
        raise ValueError("PMU mapping and branch identifier disagree")
    if mapping.terminal == "from" and branch.from_bus != mapping.pmu_bus:
        raise ValueError("PMU map says from-terminal but PMU is not branch.from_bus")
    if mapping.terminal == "to" and branch.to_bus != mapping.pmu_bus:
        raise ValueError("PMU map says to-terminal but PMU is not branch.to_bus")
    i_from, i_to = branch_terminal_currents(branch, v_from, v_to)
    raw = i_from if mapping.terminal == "from" else i_to
    return complex(int(mapping.polarity) * raw)


def _causal_unwrap(phase_rad: np.ndarray) -> np.ndarray:
    """Unwrap sequentially so each result depends on no future sample."""
    phase = np.asarray(phase_rad, dtype=float)
    result = np.full_like(phase, np.nan)
    previous: float | None = None
    for index, value in enumerate(phase):
        if not np.isfinite(value):
            previous = None
            continue
        if previous is None:
            result[index] = value
        else:
            delta = np.angle(np.exp(1j * (value - previous)))
            result[index] = result[index - 1] + delta
        previous = value
    return result


def causal_frequency_rocof(phase_rad: np.ndarray, sample_rate_hz: float, *, window_frames: int = 5, nominal_frequency_hz: float = NOMINAL_FREQUENCY_HZ) -> tuple[np.ndarray, np.ndarray]:
    """Estimate frequency and ROCOF from a trailing phase-slope window only."""
    if sample_rate_hz <= 0 or window_frames < 2:
        raise ValueError("sample_rate_hz must be positive and window_frames >= 2")
    phase = _causal_unwrap(phase_rad)
    frequency = np.full_like(phase, np.nan)
    timestep = 1.0 / sample_rate_hz
    for end in range(len(phase)):
        begin = max(0, end - window_frames + 1)
        segment = phase[begin:end + 1]
        if len(segment) < 2 or not np.all(np.isfinite(segment)):
            continue
        times = np.arange(begin, end + 1, dtype=float) * timestep
        slope = np.polyfit(times - times[0], segment, 1)[0]
        frequency[end] = nominal_frequency_hz + slope / (2.0 * np.pi)
    rocof = np.full_like(frequency, np.nan)
    valid = np.isfinite(frequency[1:]) & np.isfinite(frequency[:-1])
    indices = np.flatnonzero(valid) + 1
    rocof[indices] = (frequency[indices] - frequency[indices - 1]) / timestep
    return frequency, rocof


def observation_row(voltage: complex, current: complex, frequency_hz: float, rocof_hz_s: float, data_present: bool) -> dict[str, float | bool]:
    """Return the estimator's positive-sequence phasor-space measurement row."""
    return {
        "V_RE": float(np.real(voltage)),
        "V_IM": float(np.imag(voltage)),
        "I_TERMINAL_RE": float(np.real(current)),
        "I_TERMINAL_IM": float(np.imag(current)),
        "F_HZ": float(frequency_hz),
        "ROCOF_HZ_S": float(rocof_hz_s),
        "DATA_PRESENT": bool(data_present),
    }
