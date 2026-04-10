"""Calibrate Q and R for the UKF from first 60 s of normal operation.

R calibration:
  For each selected measurement channel, compute the empirical variance of
  (CSV_value) over the first 60 s (Event == 0). The offset between the model
  prediction h(x0) and the data mean is also recorded so the UKF can center its
  innovations correctly.

  R = diag(var_ch0, var_ch1, ..., var_ch_{n_z-1})  [diagonal only]

Q calibration:
  Physical reasoning (classical swing model):
    δ  noise: very small — angle accumulates ω*dt, not directly driven
    ω  noise: small — speed deviations are driven by power imbalances
    Pm noise: larger — mechanical power is a random-walk state

  Continuous-time spectral density (p.u., rad):
    q_delta = (1e-6)^2   → variance per step = q × dt ≈ 3e-13  (rad²)
    q_omega = (5e-5)^2   → variance per step ≈ 8e-11  (rad²/s²)
    q_Pm    = (5e-4)^2   → variance per step ≈ 8e-9   (p.u.²)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.io.load_csv import PMU_BUSES
from src.dynamics.parameters import OMEGA_S

log = logging.getLogger(__name__)

# Selected channels per bus for the UKF measurement vector
# Indices into the (8, 14) h(x) output (after flattening per bus first):
#   channel 1  = VA_MAG   (volts)
#   channel 7  = IA_MAG   (amperes)
#   channel 12 = Freq     (Hz)
#   channel 13 = ROCOF    (Hz/s)
MEAS_CHANNELS = [1, 7, 12, 13]  # 4 channels × 8 buses = 32
N_CHAN_PER_BUS = len(MEAS_CHANNELS)
N_Z = N_CHAN_PER_BUS * len(PMU_BUSES)  # 32

# Column names in the merged DataFrame for each selected channel, per bus
_CHAN_NAMES = {1: "VA_MAG", 7: "IA_MAG", 12: "Freq", 13: "ROCOF"}


def channel_cols() -> list[str]:
    """Return the ordered list of column names for the UKF measurement vector."""
    cols = []
    for bus in PMU_BUSES:
        for ch_idx in MEAS_CHANNELS:
            cols.append(f"BUS{bus}_{_CHAN_NAMES[ch_idx]}")
    return cols


def calibrate_R(
    df: pd.DataFrame,
    h0: np.ndarray,
    n_seconds: float = 60.0,
    min_variance: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit diagonal R and offset from the first n_seconds of normal data.

    Args:
        df: merged DataFrame with all 8 bus columns and TIMESTAMP.
        h0: (N_Z,) model prediction at base case [h(x0) for selected channels].
        n_seconds: seconds of data to use (Event == 0 region).
        min_variance: floor for diagonal R entries (prevents division by zero).

    Returns:
        R:      (N_Z, N_Z) diagonal measurement noise covariance.
        offset: (N_Z,) per-channel mean bias = mean(CSV) - h0
    """
    normal = df[(df["TIMESTAMP"] <= n_seconds) & (df["Event"] == 0)].copy()
    if len(normal) < 10:
        log.warning("Too few normal rows for calibration (%d); using defaults", len(normal))
        return _default_R(), np.zeros(N_Z)

    cols = channel_cols()
    available = [c for c in cols if c in normal.columns]
    if len(available) < len(cols):
        missing = set(cols) - set(available)
        log.warning("Missing calibration columns: %s", missing)

    variances = np.zeros(N_Z)
    offset = np.zeros(N_Z)

    for i, col in enumerate(cols):
        if col not in normal.columns:
            variances[i] = min_variance
            continue
        vals = normal[col].dropna().to_numpy(float)
        if len(vals) < 2:
            variances[i] = min_variance
            continue
        mean_val = vals.mean()
        var_val = vals.var(ddof=1)
        variances[i] = max(var_val, min_variance)
        offset[i] = mean_val - h0[i]

    R = np.diag(variances)
    log.info(
        "R calibrated: min_var=%.3e, max_var=%.3e, mean_offset=%.4g",
        variances.min(), variances.max(), np.abs(offset).mean(),
    )
    return R, offset


def _default_R() -> np.ndarray:
    """Fallback R based on typical measurement noise for the 39-bus system."""
    # Per-channel defaults (rough estimates)
    defaults = {
        "VA_MAG": 1000.0**2,  # ±1000 V
        "IA_MAG": 5.0**2,     # ±5 A
        "Freq":   0.01**2,    # ±0.01 Hz
        "ROCOF":  0.01**2,    # ±0.01 Hz/s
    }
    variances = []
    for bus in PMU_BUSES:
        for ch_idx in MEAS_CHANNELS:
            name = _CHAN_NAMES[ch_idx]
            variances.append(defaults[name])
    return np.diag(variances)


def build_Q(n_gen: int = 10, dt: float = 1.0 / 30.0) -> np.ndarray:
    """Build the discrete-time process noise covariance Q ∈ R^{30×30}.

    Physical reasoning (per CLAUDE.md §3):
      δ: accumulates ω*dt — very small direct noise
      ω: speed deviation — small process noise (damping handles it)
      Pm: mechanical power random walk — larger noise (models uncertainty in Pm)

    Returns Q in state order [δ(n_gen), ω(n_gen), Pm(n_gen)].
    """
    # Continuous-time spectral density (p.u.)
    q_delta = 1e-6**2    # (rad)² / s
    q_omega = 5e-5**2    # (rad/s)² / s
    q_Pm    = 5e-4**2    # (p.u.)² / s

    # Discrete-time Q = q_cont × dt (Euler approximation)
    q_delta_disc = q_delta * dt
    q_omega_disc = q_omega * dt
    q_Pm_disc    = q_Pm    * dt

    diag = np.concatenate([
        np.full(n_gen, q_delta_disc),
        np.full(n_gen, q_omega_disc),
        np.full(n_gen, q_Pm_disc),
    ])
    return np.diag(diag)


def build_P0(
    n_gen: int = 10,
    delta_std_deg: float = 5.0,
    omega_std_rad_per_s: float = 0.1,
    Pm_std_pu: float = 0.05,
) -> np.ndarray:
    """Build initial state covariance P0 (large = uncertain initial condition).

    Args:
        n_gen: number of generators.
        delta_std_deg: initial rotor angle uncertainty (degrees).
        omega_std_rad_per_s: initial speed deviation uncertainty (rad/s).
        Pm_std_pu: initial mechanical power uncertainty (p.u.).
    """
    delta_var = np.deg2rad(delta_std_deg)**2
    omega_var = omega_std_rad_per_s**2
    Pm_var = Pm_std_pu**2

    diag = np.concatenate([
        np.full(n_gen, delta_var),
        np.full(n_gen, omega_var),
        np.full(n_gen, Pm_var),
    ])
    return np.diag(diag)
