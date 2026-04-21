"""Reduced swing-equation dynamics for hybrid M8 estimators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.estimation.dynamic_state_estimation.hybrid_state_definition import HybridStateLayout


@dataclass(slots=True)
class SwingParams:
    """Simplified swing-parameter set used by reduced-order model."""

    inertia_m: float = 4.0
    damping_d: float = 1.2
    synchronizing_k: float = 3.0
    voltage_rotation_gain: float = 0.25


def step_swing(delta: np.ndarray, omega: np.ndarray, dt: float, params: SwingParams) -> tuple[np.ndarray, np.ndarray]:
    """One explicit-Euler step for reduced swing equations."""
    d = np.asarray(delta, dtype=float)
    w = np.asarray(omega, dtype=float)
    ddot = w
    wdot = -(params.damping_d / max(params.inertia_m, 1e-6)) * w - (params.synchronizing_k / max(params.inertia_m, 1e-6)) * d
    d_next = d + dt * ddot
    w_next = w + dt * wdot
    return d_next, w_next


def predict_hybrid_state(x: np.ndarray, layout: HybridStateLayout, dt: float, params: SwingParams) -> np.ndarray:
    """Predict next hybrid state with reduced swing coupling."""
    x0 = np.asarray(x, dtype=float)
    x1 = x0.copy()
    if layout.dynamic_dim == 0:
        return x1

    n_bus = len(layout.bus_order)
    n_gen = len(layout.generator_buses)
    delta = x0[layout.delta_offset : layout.delta_offset + n_gen]
    omega = x0[layout.omega_offset : layout.omega_offset + n_gen]
    d_next, w_next = step_swing(delta=delta, omega=omega, dt=dt, params=params)
    x1[layout.delta_offset : layout.delta_offset + n_gen] = d_next
    x1[layout.omega_offset : layout.omega_offset + n_gen] = w_next

    # Couple generator speed to generator-bus voltage rotation.
    for bus in layout.generator_buses:
        bi = layout.bus_to_index[bus]
        gi = layout.gen_to_index[bus]
        vr = x1[layout.vr_offset + bi]
        vi = x1[layout.vi_offset + bi]
        z = complex(vr, vi)
        theta = params.voltage_rotation_gain * d_next[gi]
        z1 = z * np.exp(1j * theta)
        x1[layout.vr_offset + bi] = float(np.real(z1))
        x1[layout.vi_offset + bi] = float(np.imag(z1))

    # Keep non-generator dynamic terms untouched by bus count.
    if len(x1) != 2 * n_bus + 2 * n_gen:
        raise ValueError("Hybrid state shape mismatch during prediction.")
    return x1

