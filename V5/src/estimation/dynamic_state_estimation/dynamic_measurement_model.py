"""Dynamic measurement model for hybrid state estimators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.estimation.dynamic_state_estimation.hybrid_state_definition import HybridStateLayout
from src.estimation.state_estimation.measurement_model import FrameMeasurements
from src.estimation.state_estimation.models import NetworkModel


@dataclass(slots=True)
class DynamicMeasurementFrame:
    """Measurement frame in real-valued stacked form."""

    timestamp: float
    z: np.ndarray
    h: np.ndarray
    H: np.ndarray
    R: np.ndarray
    n_voltage: int
    n_current: int
    pmu_buses_used: list[str]
    pmu_buses_excluded: list[str]
    dropped_reasons: dict[str, str]


def voltage_vector_from_state(x: np.ndarray, layout: HybridStateLayout) -> np.ndarray:
    """Extract complex bus voltage vector from hybrid state."""
    xr = np.asarray(x, dtype=float)
    n = len(layout.bus_order)
    vr = xr[layout.vr_offset : layout.vr_offset + n]
    vi = xr[layout.vi_offset : layout.vi_offset + n]
    return vr + 1j * vi


def build_measurement_matrices(
    frame: FrameMeasurements,
    network: NetworkModel,
    layout: HybridStateLayout,
    x_pred: np.ndarray,
    voltage_sigma: float = 0.01,
    current_sigma: float = 0.02,
) -> DynamicMeasurementFrame:
    """Build z, h(x_pred), H, R for EKF/UKF update."""
    n_bus = len(layout.bus_order)
    n_state = layout.total_dim
    y = np.asarray(network.ybus, dtype=complex)
    pos = layout.bus_to_index
    x = np.asarray(x_pred, dtype=float)
    v = voltage_vector_from_state(x, layout)
    i_est = y @ v

    z_rows: list[float] = []
    h_rows: list[float] = []
    hmat_rows: list[np.ndarray] = []
    r_diag: list[float] = []
    used: set[str] = set()
    n_voltage = 0
    n_current = 0

    for bus, meas_v in frame.voltage_by_bus.items():
        if bus not in pos:
            continue
        bi = pos[bus]
        row_r = np.zeros(n_state, dtype=float)
        row_i = np.zeros(n_state, dtype=float)
        row_r[layout.vr_offset + bi] = 1.0
        row_i[layout.vi_offset + bi] = 1.0
        hmat_rows.extend([row_r, row_i])
        z_rows.extend([float(np.real(meas_v)), float(np.imag(meas_v))])
        h_rows.extend([float(np.real(v[bi])), float(np.imag(v[bi]))])
        r_diag.extend([voltage_sigma**2, voltage_sigma**2])
        n_voltage += 2
        used.add(bus)

    g = np.real(y)
    b = np.imag(y)
    for bus, meas_i in frame.current_by_bus.items():
        if bus not in pos:
            continue
        bi = pos[bus]
        row_ir = np.zeros(n_state, dtype=float)
        row_ii = np.zeros(n_state, dtype=float)
        row_ir[layout.vr_offset : layout.vr_offset + n_bus] = g[bi, :]
        row_ir[layout.vi_offset : layout.vi_offset + n_bus] = -b[bi, :]
        row_ii[layout.vr_offset : layout.vr_offset + n_bus] = b[bi, :]
        row_ii[layout.vi_offset : layout.vi_offset + n_bus] = g[bi, :]
        hmat_rows.extend([row_ir, row_ii])
        z_rows.extend([float(np.real(meas_i)), float(np.imag(meas_i))])
        h_rows.extend([float(np.real(i_est[bi])), float(np.imag(i_est[bi]))])
        r_diag.extend([current_sigma**2, current_sigma**2])
        n_current += 2
        used.add(bus)

    if not hmat_rows:
        hmat = np.zeros((0, n_state), dtype=float)
    else:
        hmat = np.vstack(hmat_rows)

    return DynamicMeasurementFrame(
        timestamp=float(frame.timestamp),
        z=np.asarray(z_rows, dtype=float),
        h=np.asarray(h_rows, dtype=float),
        H=hmat,
        R=np.diag(np.asarray(r_diag, dtype=float)) if r_diag else np.zeros((0, 0), dtype=float),
        n_voltage=n_voltage,
        n_current=n_current,
        pmu_buses_used=sorted(used),
        pmu_buses_excluded=sorted(set(frame.expected_pmu_buses or []) - set(used)),
        dropped_reasons=dict(frame.dropped_reasons or {}),
    )


def measurement_function(
    x: np.ndarray,
    frame: FrameMeasurements,
    network: NetworkModel,
    layout: HybridStateLayout,
) -> np.ndarray:
    """Compute measurement model h(x) in the same ordering as build_measurement_matrices."""
    v = voltage_vector_from_state(np.asarray(x, dtype=float), layout)
    i_est = np.asarray(network.ybus, dtype=complex) @ v
    pos = layout.bus_to_index
    out: list[float] = []
    for bus in frame.voltage_by_bus:
        if bus not in pos:
            continue
        bi = pos[bus]
        out.extend([float(np.real(v[bi])), float(np.imag(v[bi]))])
    for bus in frame.current_by_bus:
        if bus not in pos:
            continue
        bi = pos[bus]
        out.extend([float(np.real(i_est[bi])), float(np.imag(i_est[bi]))])
    return np.asarray(out, dtype=float)

