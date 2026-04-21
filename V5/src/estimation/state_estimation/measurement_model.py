"""Measurement model builder for M6 regularized PMU state estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.estimation.state_estimation.models import EstimationConfig, NetworkModel


@dataclass(slots=True)
class FrameMeasurements:
    """Per-timestamp PMU measurements already converted to per-unit."""

    timestamp: float
    voltage_by_bus: dict[str, complex]
    current_by_bus: dict[str, complex]
    data_present_count: int
    expected_pmu_buses: list[str] | None = None
    valid_pmu_buses: list[str] | None = None
    dropped_reasons: dict[str, str] | None = None


@dataclass(slots=True)
class LinearModelFrame:
    """Linear complex measurement system A x ~= b with per-row weights."""

    timestamp: float
    A: np.ndarray
    b: np.ndarray
    w: np.ndarray
    n_voltage: int
    n_current: int
    pmu_buses_used: list[str]
    pmu_buses_excluded: list[str]
    dropped_reasons: dict[str, str]


def build_linear_frame(
    frame: FrameMeasurements,
    network: NetworkModel,
    config: EstimationConfig,
) -> LinearModelFrame:
    """Build linear measurement model for a single timestamp."""
    n_bus = len(network.bus_order)
    pos = {bus: i for i, bus in enumerate(network.bus_order)}

    rows: list[np.ndarray] = []
    rhs: list[complex] = []
    weights: list[float] = []
    n_voltage = 0
    n_current = 0
    used: set[str] = set()

    for bus, v in frame.voltage_by_bus.items():
        if bus not in pos or not np.isfinite(np.real(v)) or not np.isfinite(np.imag(v)):
            continue
        row = np.zeros(n_bus, dtype=complex)
        row[pos[bus]] = 1.0 + 0j
        rows.append(row)
        rhs.append(complex(v))
        weights.append(float(config.voltage_weight))
        n_voltage += 1
        used.add(bus)

    for bus, i_meas in frame.current_by_bus.items():
        if bus not in pos or not np.isfinite(np.real(i_meas)) or not np.isfinite(np.imag(i_meas)):
            continue
        rows.append(np.asarray(network.ybus[pos[bus], :], dtype=complex))
        rhs.append(complex(i_meas))
        weights.append(float(config.current_weight))
        n_current += 1
        used.add(bus)

    if not rows:
        A = np.zeros((0, n_bus), dtype=complex)
        b = np.zeros(0, dtype=complex)
        w = np.zeros(0, dtype=float)
    else:
        A = np.vstack(rows)
        b = np.asarray(rhs, dtype=complex)
        w = np.asarray(weights, dtype=float)

    return LinearModelFrame(
        timestamp=float(frame.timestamp),
        A=A,
        b=b,
        w=w,
        n_voltage=n_voltage,
        n_current=n_current,
        pmu_buses_used=sorted(used),
        pmu_buses_excluded=sorted(set(frame.expected_pmu_buses or []) - set(used)),
        dropped_reasons=dict(frame.dropped_reasons or {}),
    )
