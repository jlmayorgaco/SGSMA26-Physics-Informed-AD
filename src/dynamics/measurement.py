"""Measurement function h(x) → 14 PMU channels × 8 buses.

Uses JAX for differentiability (needed by the T2 UKF Jacobian).
Falls back to NumPy if JAX is unavailable.

Channel layout per bus (14 channels):
  0  VA_ANG  (deg)     3  VB_MAG  (V)     7  IA_MAG  (A)    11  IC_MAG  (A)
  1  VA_MAG  (V)       4  VC_ANG  (deg)   8  IB_ANG  (deg)  12  Freq   (Hz)
  2  VB_ANG  (deg)     5  VC_MAG  (V)     9  IB_MAG  (A)    13  ROCOF  (Hz/s)
                       6  IA_ANG  (deg)  10  IC_ANG  (deg)

Physics:
  For a balanced positive-sequence system with PMU bus k at angle θ_k:
    VA_ANG_k = θ_k (degrees)
    VB_ANG_k = θ_k - 120°
    VC_ANG_k = θ_k + 120°
    V*_MAG   = |V_k| * V_base_LN  (volts, line-to-neutral)

  Angle at PMU bus k from the linearized DC-PF sensitivity:
    θ_k(t) = θ_k0 + Σ_i S_ki * (δ_i - δ_i0)   (radians)

  Bus frequency:
    f_k = f_0 + dθ_k/dt / (2π) = 60 + Σ_i S_ki * ω_i / (2π)

  ROCOF:
    ROCOF_k = df_k/dt = Σ_i S_ki * (dω_i/dt) / (2π)
    dω_i/dt = (ω_s / 2H_i) * (Pm_i - Pe_i) - (D_i / 2H_i) * ω_i   (from swing model)

  PMU current at bus k (positive-sequence, line current magnitude/angle):
    I_k = Σ_j Y_kj * V_j  (complex, using full network Ybus at PMU bus rows)
    At base case: I_k0 (precomputed from power-flow)
    Linearized: ΔI_k ≈ Σ_j Y_pmu_full[k,j] * ΔV_j
    For implementation: I_k ≈ I_k0 (magnitude, angle) + angle perturbation from ΔV

  Current base for unit conversion:
    S_base_per_phase = V_base_LN * I_base  →  I_base = S_base / (3 * V_base_LN)
    For S_base = 100 MVA, V_base_LN = 345e3/√3: I_base = 100e6 / (3 * 199185) ≈ 167.3 A

Hot-path design:
  - All bus indices are integer positions, never name lookups.
  - Precomputed arrays (S, I_k0, theta_k0, V_k0, ...) are stored on the model.
  - h(x) takes a flat (30,) JAX array and returns a (8, 14) JAX array.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)

# Try JAX; fall back to NumPy silently
try:
    import jax.numpy as jnp
    import jax
    _JAX_AVAILABLE = True
except ImportError:
    import numpy as jnp  # type: ignore[no-redef]
    _JAX_AVAILABLE = False
    log.warning("JAX not available — measurement function uses NumPy fallback")

N_GEN = 10
N_PMU = 8
N_CHAN = 14
F0 = 60.0                     # nominal frequency (Hz)
V_BASE_LL = 345e3             # line-to-line base voltage (V)
V_BASE_LN = V_BASE_LL / np.sqrt(3)   # 199,186 V (line-to-neutral)
S_BASE = 100e6                # 100 MVA
I_BASE = S_BASE / (3 * V_BASE_LN)    # ≈ 167.3 A per phase


@dataclass
class MeasurementModel:
    """Pre-assembled measurement model; hot path is h(x).

    Attributes:
        S:          (N_PMU, N_GEN) angle-sensitivity matrix
                    S[k,i] = ∂θ_pmu_k / ∂δ_gen_i  (radians/radian)
        theta0:     (N_PMU,) base-case bus voltage angles (radians)
        vmag0:      (N_PMU,) base-case voltage magnitudes (p.u.)
        I_ang0:     (N_PMU,) base-case current angles (radians, positive-sequence phase A)
        I_mag0:     (N_PMU,) base-case current magnitudes (p.u.)
        delta0:     (N_GEN,) base-case rotor angles (radians)
        H:          (N_GEN,) inertia constants (seconds)
        D:          (N_GEN,) damping coefficients
        omega_s:    synchronous angular velocity (rad/s)
        use_jax:    bool, whether to use JAX arrays
    """
    S: np.ndarray         # (8, 10)
    theta0: np.ndarray    # (8,) radians
    vmag0: np.ndarray     # (8,) p.u.
    I_ang0: np.ndarray    # (8,) radians
    I_mag0: np.ndarray    # (8,) p.u.
    delta0: np.ndarray    # (10,) radians
    H: np.ndarray         # (10,)
    D: np.ndarray         # (10,)
    omega_s: float = 2 * np.pi * F0
    use_jax: bool = False

    def __post_init__(self):
        if _JAX_AVAILABLE:
            self.use_jax = True
            # Convert arrays to JAX for the hot path
            self._S = jnp.array(self.S)
            self._theta0 = jnp.array(self.theta0)
            self._vmag0 = jnp.array(self.vmag0)
            self._I_ang0 = jnp.array(self.I_ang0)
            self._I_mag0 = jnp.array(self.I_mag0)
            self._delta0 = jnp.array(self.delta0)
            self._H = jnp.array(self.H)
            self._D = jnp.array(self.D)
        else:
            self._S = self.S
            self._theta0 = self.theta0
            self._vmag0 = self.vmag0
            self._I_ang0 = self.I_ang0
            self._I_mag0 = self.I_mag0
            self._delta0 = self.delta0
            self._H = self.H
            self._D = self.D

    def h(self, x) -> "jnp.ndarray":
        """Compute 14-channel PMU measurements for 8 buses.

        Args:
            x: (30,) state vector [δ(10), ω(10), Pm(10)]

        Returns:
            y: (8, 14) measurement array in physical units:
               columns 0,2,4,6,8,10 = angles (degrees)
               columns 1,3,5,7,9,11 = magnitudes (V or A)
               column 12 = frequency (Hz)
               column 13 = ROCOF (Hz/s)
        """
        xp = jnp if self.use_jax else np

        delta = x[:N_GEN]
        omega = x[N_GEN: 2 * N_GEN]
        Pm    = x[2 * N_GEN:]

        S  = self._S        # (8, 10)
        d0 = self._delta0   # (10,)
        H  = self._H        # (10,)
        D  = self._D        # (10,)

        # ── Voltage angles at PMU buses ────────────────────────────────────────
        # θ_k = θ_k0 + Σ_i S_ki * (δ_i - δ_i0)   [radians]
        d_delta = delta - d0                         # (10,)
        d_theta = S @ d_delta                        # (8,) radians
        theta_k = self._theta0 + d_theta             # (8,) radians
        theta_k_deg = xp.rad2deg(theta_k)            # (8,) degrees

        # ── Voltage magnitudes (approximately constant in classical model) ──────
        vmag_V = self._vmag0 * V_BASE_LN             # (8,) volts

        # ── Current: linearized around base case ─────────────────────────────
        # Current angle tracks voltage angle perturbation (impedance-load approx.)
        # For small deviations: ΔI_ang ≈ Δθ (voltage leads to current angle change)
        I_ang_k = self._I_ang0 + d_theta             # (8,) radians
        I_ang_k_deg = xp.rad2deg(I_ang_k)            # (8,) degrees
        I_mag_A = self._I_mag0 * I_BASE              # (8,) amperes (constant)

        # ── Frequency: f_k = 60 + (dθ_k/dt) / (2π) ──────────────────────────
        # dθ_k/dt = S @ ω   (since θ_k = θ_k0 + S @ (δ - δ_0) and dδ/dt = ω)
        d_theta_dt = S @ omega                       # (8,) rad/s
        freq_k = F0 + d_theta_dt / (2 * xp.pi)      # (8,) Hz

        # ── ROCOF: df/dt = (d²θ/dt²) / (2π) = (S @ dω/dt) / (2π) ──────────
        # dω/dt = (ω_s / 2H) * (Pm - Pe) - (D / 2H) * ω
        # Approximate Pe ≈ Pm at base case for ROCOF (small-signal): use Pm - Pe ≈ 0
        # More accurate: use Pe from swing model; but for measurement function
        # in the UKF hot path, use the analytical expression:
        #   dω_i/dt ≈ (ω_s / 2H_i) * (Pm_i - Pe_i) - (D_i / 2H_i) * ω_i
        # We don't recompute Pe here (expensive); use linear approx from ω only:
        #   dω_i/dt ≈ -(D_i / 2H_i) * ω_i    (valid near equilibrium)
        two_H = 2.0 * H
        d_omega_dt = -(D / two_H) * omega            # (10,) rad/s^2 (damping only)
        d2_theta_dt2 = S @ d_omega_dt                # (8,) rad/s^2
        rocof_k = d2_theta_dt2 / (2 * xp.pi)        # (8,) Hz/s

        # ── Three-phase symmetry (balanced, positive sequence) ────────────────
        VA_ang = theta_k_deg
        VA_mag = vmag_V
        VB_ang = theta_k_deg - 120.0
        VB_mag = vmag_V
        VC_ang = theta_k_deg + 120.0
        VC_mag = vmag_V
        IA_ang = I_ang_k_deg
        IA_mag = I_mag_A
        IB_ang = I_ang_k_deg - 120.0
        IB_mag = I_mag_A
        IC_ang = I_ang_k_deg + 120.0
        IC_mag = I_mag_A

        # ── Stack into (8, 14) ────────────────────────────────────────────────
        y = xp.stack([
            VA_ang, VA_mag, VB_ang, VB_mag, VC_ang, VC_mag,
            IA_ang, IA_mag, IB_ang, IB_mag, IC_ang, IC_mag,
            freq_k, rocof_k,
        ], axis=1)                                   # (8, 14)
        return y

    def h_flat(self, x) -> "jnp.ndarray":
        """Flat version: returns (112,) for the UKF."""
        return self.h(x).ravel()


def build_measurement_model(
    swing_model,           # SwingModel from swing.py
    Ybus: np.ndarray,      # (N, N) complex admittance (full network)
    pmu_indices: list[int],  # row indices of 8 PMU buses in Ybus
    gen_indices: list[int],  # row indices of 10 generator terminal buses in Ybus
    bus_theta0_rad: np.ndarray,  # (N,) base-case bus angles (radians)
    bus_vmag0_pu: np.ndarray,    # (N,) base-case voltage magnitudes (p.u.)
) -> MeasurementModel:
    """Build the MeasurementModel from grid case data.

    Computes:
      S[k, i] = B'^{-1}[pmu_k, gen_i]  (DC-PF angle sensitivity)
      I_k0 = Σ_j Y_kj * V_j  (base-case complex current at PMU bus k)

    where B' is the imaginary part of Ybus (DC power-flow B matrix).
    """
    N = Ybus.shape[0]

    # ── Sensitivity matrix S (8, 10) via DC-PF B' matrix ─────────────────────
    slack_idx = _find_slack_from_ybus(Ybus)
    non_slack = [i for i in range(N) if i != slack_idx]
    B_red = Ybus.imag[np.ix_(non_slack, non_slack)]

    try:
        B_inv = np.linalg.inv(B_red)
    except np.linalg.LinAlgError:
        B_inv = np.linalg.pinv(B_red)

    # Map PMU and gen indices to reduced (non-slack) index space
    def to_reduced(idx):
        return non_slack.index(idx) if idx in non_slack else None

    pmu_red = [to_reduced(i) for i in pmu_indices]
    gen_red = [to_reduced(i) for i in gen_indices]

    S = np.zeros((N_PMU, N_GEN))
    for row, pr in enumerate(pmu_red):
        for col, gr in enumerate(gen_red):
            if pr is not None and gr is not None:
                S[row, col] = B_inv[pr, gr]

    # ── Base-case current at PMU buses ────────────────────────────────────────
    # Complex voltage vector at all buses
    V0_complex = bus_vmag0_pu * np.exp(1j * bus_theta0_rad)   # (N,)
    # Complex current at PMU bus k = Σ_j Y_kj V_j
    I_complex = (Ybus[np.ix_(pmu_indices, np.arange(N))] @ V0_complex)  # (8,)
    I_ang0 = np.angle(I_complex)           # (8,) radians
    I_mag0 = np.abs(I_complex)             # (8,) p.u.

    # PMU bus base-case angles/magnitudes
    theta0 = bus_theta0_rad[pmu_indices]   # (8,) radians
    vmag0 = bus_vmag0_pu[pmu_indices]      # (8,) p.u.

    log.info(
        "MeasurementModel built: S norm=%s, I_mag0(p.u.)=%s",
        np.round(np.linalg.norm(S, axis=1), 4),
        np.round(I_mag0, 4),
    )

    return MeasurementModel(
        S=S,
        theta0=theta0,
        vmag0=vmag0,
        I_ang0=I_ang0,
        I_mag0=I_mag0,
        delta0=swing_model.delta0,
        H=swing_model.params.H,
        D=swing_model.params.D,
    )


def _find_slack_from_ybus(Ybus: np.ndarray) -> int:
    """Heuristic: slack bus row has the largest diagonal magnitude (most connections)."""
    return int(np.argmax(np.abs(np.diag(Ybus))))
