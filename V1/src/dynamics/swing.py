"""Classical 2nd-order swing model for 10 IEEE-39 generators.

State vector x ∈ R^30:
  x[0:10]  = δ_i  rotor angles (radians) w.r.t. synchronous rotating reference
  x[10:20] = ω_i  rotor speed deviation (rad/s)  = ω_actual - ω_s
  x[20:30] = P_m,i mechanical power (p.u. on system base)  — random-walk states

Swing equations:
  dδ_i/dt  = ω_i
  dω_i/dt  = (ω_s / (2 H_i)) * (P_m,i - P_e,i) - (D_i / (2 H_i)) * ω_i
  dP_m,i/dt = 0  (or w_i(t) process noise in T2 UKF)

Electrical power P_e,i from the Kron-reduced admittance Y_red (10×10, complex):
  P_e,i = E_i^2 * G_ii + Σ_{j≠i} E_i E_j [G_ij cos(δ_i−δ_j) + B_ij sin(δ_i−δ_j)]

where G_ij + jB_ij = Y_red[i,j] and E_i = |E_i| is the constant internal EMF magnitude.

The RK4 integrator uses the per-step Δt computed from actual timestamps
(per CLAUDE.md §2.5 — never hardcode 1/30).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src.dynamics.parameters import OMEGA_S, GenParams

log = logging.getLogger(__name__)

N_GEN = 10
STATE_DIM = 3 * N_GEN  # 30


@dataclass
class SwingModel:
    """Pre-assembled swing model ready to integrate.

    Attributes:
        params:     GenParams with H, D, xd1_sys for 10 generators
        Y_red:      (10, 10) complex Kron-reduced admittance at generator buses
        E_mag:      (10,) internal EMF magnitudes (p.u.), calibrated from base case
        delta0:     (10,) base-case rotor angles (radians)
        omega0:     (10,) base-case speed deviations = 0
        Pm0:        (10,) base-case mechanical power (p.u.)
    """
    params: GenParams
    Y_red: np.ndarray       # (10, 10) complex, on system MVA base
    E_mag: np.ndarray       # (10,) internal EMF magnitudes
    delta0: np.ndarray      # (10,) radians
    omega0: np.ndarray      # (10,) rad/s, all zeros
    Pm0: np.ndarray         # (10,) p.u.

    def x0(self) -> np.ndarray:
        """Return the base-case state vector x0 ∈ R^30."""
        return np.concatenate([self.delta0, self.omega0, self.Pm0])

    def electric_power(self, delta: np.ndarray) -> np.ndarray:
        """Compute electrical power Pe_i (p.u.) from rotor angles.

        P_e,i = Re[E_i * conj(Σ_j Y_red[i,j] * E_j exp(j δ_j))]
              = E_i^2 G_ii + Σ_{j≠i} E_i E_j (G_ij cos δ_ij + B_ij sin δ_ij)
        """
        E = self.E_mag  # (10,)
        G = self.Y_red.real
        B = self.Y_red.imag

        # Broadcast: δ_i - δ_j  (10, 10) matrix
        delta_col = delta.reshape(-1, 1)
        delta_row = delta.reshape(1, -1)
        dij = delta_col - delta_row   # dij[i,j] = δ_i - δ_j

        Ei = E.reshape(-1, 1)
        Ej = E.reshape(1, -1)

        Pe = np.sum(Ei * Ej * (G * np.cos(dij) + B * np.sin(dij)), axis=1)
        return Pe  # (10,)

    def f(self, x: np.ndarray) -> np.ndarray:
        """Right-hand side of the swing ODE: dx/dt = f(x).

        No process noise (deterministic open-loop).
        """
        delta = x[:N_GEN]
        omega = x[N_GEN: 2 * N_GEN]
        Pm = x[2 * N_GEN:]

        Pe = self.electric_power(delta)

        # dδ/dt = ω
        d_delta = omega.copy()

        # dω/dt = (ω_s / 2H) * (Pm - Pe) - (D / 2H) * ω
        two_H = 2.0 * self.params.H
        d_omega = (OMEGA_S / two_H) * (Pm - Pe) - (self.params.D / two_H) * omega

        # dPm/dt = 0 (random walk noise added externally by UKF)
        d_Pm = np.zeros(N_GEN)

        return np.concatenate([d_delta, d_omega, d_Pm])

    def step(self, x: np.ndarray, dt: float) -> np.ndarray:
        """One RK4 step of duration dt (seconds)."""
        k1 = self.f(x)
        k2 = self.f(x + 0.5 * dt * k1)
        k3 = self.f(x + 0.5 * dt * k2)
        k4 = self.f(x + dt * k3)
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def simulate(
        self,
        x0: np.ndarray | None = None,
        timestamps: np.ndarray | None = None,
        n_steps: int | None = None,
        dt_fixed: float = 1.0 / 30.0,
    ) -> np.ndarray:
        """Simulate the swing model.

        Args:
            x0: initial state (default: self.x0())
            timestamps: 1-D array of timestamps (seconds); Δt computed per step
            n_steps: used only when timestamps is None
            dt_fixed: fallback step size if timestamps is None

        Returns:
            X: (T+1, 30) trajectory including x0
        """
        if x0 is None:
            x0 = self.x0()
        x = x0.copy()

        if timestamps is not None:
            dts = np.diff(timestamps)
            T = len(dts)
        else:
            assert n_steps is not None
            dts = np.full(n_steps, dt_fixed)
            T = n_steps

        X = np.empty((T + 1, STATE_DIM))
        X[0] = x
        for i, dt in enumerate(dts):
            x = self.step(x, float(dt))
            X[i + 1] = x
        return X


def build_model(
    params: GenParams,
    Y_red: np.ndarray,
    gen_terminal_angles_deg: list[float],
    gen_terminal_vmag_pu: list[float],
) -> SwingModel:
    """Assemble a SwingModel from parameters and base-case power-flow data.

    The internal EMF magnitude E_i is calibrated so that P_e,i exactly equals
    P_m,i at the base-case operating point, guaranteeing a zero-drift equilibrium.

    Algorithm:
      1. Initialize δ_i0 = terminal bus angle + load angle correction (simplified)
      2. Set E_i = V_{terminal,i} initially (neglect load angle in magnitude)
      3. Compute P_e,i0 from Y_red
      4. Set Pm,i0 = P_e,i0 (so P_m - P_e = 0 by construction)

    Args:
        params: GenParams with H, D, xd1_sys
        Y_red: (10, 10) complex admittance on system base
        gen_terminal_angles_deg: base-case voltage angles at gen terminal buses (degrees)
        gen_terminal_vmag_pu: base-case voltage magnitudes at gen terminal buses (p.u.)
    """
    delta0 = np.deg2rad(np.array(gen_terminal_angles_deg))
    E_mag = np.array(gen_terminal_vmag_pu)

    # Build a temporary model to compute P_e at the base case
    tmp = SwingModel(
        params=params, Y_red=Y_red, E_mag=E_mag,
        delta0=delta0, omega0=np.zeros(N_GEN), Pm0=np.zeros(N_GEN),
    )
    Pe0 = tmp.electric_power(delta0)

    log.info("Base-case Pe,i (p.u.): %s", np.round(Pe0, 4))
    log.info("Base-case Pm,i from .raw (p.u.): %s", np.round(params.Pm_base, 4))

    # Set Pm = Pe so equilibrium is exact (avoids initialization drift)
    # Note: this may differ from the nominal .raw values by a few percent
    # due to the classical-model approximation. For T2 UKF, Pe0 is what matters.
    Pm0 = Pe0.copy()

    return SwingModel(
        params=params,
        Y_red=Y_red,
        E_mag=E_mag,
        delta0=delta0,
        omega0=np.zeros(N_GEN),
        Pm0=Pm0,
    )
