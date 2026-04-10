"""Unscented Kalman Filter with Merwe scaled sigma points.

Parameters: α=1e-3, β=2, κ=0 (standard UKF defaults for state estimation).

Implementation notes:
  • Standard (non-square-root) form with symmetric enforcement and Cholesky
    regularisation to maintain positive-definiteness over 162 k steps.
  • Sigma-point propagation through the swing model is VECTORISED — all 2n+1
    sigma points are processed in a single batch matrix operation.
  • Measurement function is evaluated batch-wise via h_batch() to avoid Python loops.
  • Innovation statistic η_t = ν_t' S_t^{-1} ν_t is computed every step and stored
    for the χ² calibration check.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import scipy.linalg

log = logging.getLogger(__name__)

# Merwe sigma-point parameters
ALPHA = 1e-3
BETA  = 2.0
KAPPA = 0.0

_EPS = 1e-10   # Cholesky regularisation floor


# ── Merwe Scaled Sigma Points ─────────────────────────────────────────────────

class MerweScaledSigmaPoints:
    """Generate and weight sigma points for the UKF.

    With n = state dimension, generates 2n+1 sigma points.
    """

    def __init__(self, n: int, alpha: float = ALPHA, beta: float = BETA, kappa: float = KAPPA):
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        lam = alpha**2 * (n + kappa) - n
        self.lam = lam
        self.c = n + lam            # scaling constant

        # Mean weights
        Wm = np.full(2 * n + 1, 0.5 / self.c)
        Wm[0] = lam / self.c
        self.Wm = Wm

        # Covariance weights
        Wc = Wm.copy()
        Wc[0] = Wm[0] + (1 - alpha**2 + beta)
        self.Wc = Wc

        self.sqrt_c = np.sqrt(abs(self.c))  # √(n + λ)

    def sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Generate (2n+1, n) sigma points from mean x and covariance P.

        Uses the Cholesky factor of P (with regularisation fallback).
        """
        L = _cholesky_safe(P)        # lower-triangular, shape (n, n)
        scaled = self.sqrt_c * L     # (n, n); each column = √((n+λ)P) direction

        X = np.empty((2 * self.n + 1, self.n))
        X[0] = x
        X[1: self.n + 1] = x + scaled.T      # x + columns of scaled
        X[self.n + 1:] = x - scaled.T        # x - columns of scaled
        return X

    def weighted_mean(self, X: np.ndarray) -> np.ndarray:
        """Weighted mean of (2n+1, dim) sigma points."""
        return self.Wm @ X   # (dim,)

    def weighted_cov(
        self, X: np.ndarray, x_mean: np.ndarray, noise: np.ndarray | None = None
    ) -> np.ndarray:
        """Weighted covariance of (2n+1, dim) sigma points around x_mean."""
        diff = X - x_mean                  # (2n+1, dim)
        P = (self.Wc[:, None] * diff).T @ diff   # (dim, dim)
        if noise is not None:
            P += noise
        return P

    def weighted_cross_cov(
        self,
        X: np.ndarray, x_mean: np.ndarray,
        Y: np.ndarray, y_mean: np.ndarray,
    ) -> np.ndarray:
        """Weighted cross-covariance P_xy of shape (dim_x, dim_y)."""
        dX = X - x_mean   # (2n+1, dim_x)
        dY = Y - y_mean   # (2n+1, dim_y)
        return (self.Wc[:, None] * dX).T @ dY  # (dim_x, dim_y)


# ── Swing model batch propagation ─────────────────────────────────────────────

def _f_batch(X: np.ndarray, swing) -> np.ndarray:
    """Vectorised RK4 step for all sigma points.

    Args:
        X: (batch, 30) sigma-point matrix
        swing: SwingModel (provides E_mag, Y_red, params.H, params.D)

    Returns:
        (batch, 30) propagated sigma points after one RK4 step.
        NOTE: dt is stored in swing._dt_current, set by the UKF before calling.
    """
    dt = swing._dt_current
    k1 = _f_rhs_batch(X, swing)
    k2 = _f_rhs_batch(X + 0.5 * dt * k1, swing)
    k3 = _f_rhs_batch(X + 0.5 * dt * k2, swing)
    k4 = _f_rhs_batch(X + dt * k3, swing)
    return X + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def _f_rhs_batch(X: np.ndarray, swing) -> np.ndarray:
    """Vectorised RHS of swing ODE for batch (batch, 30)."""
    from src.dynamics.parameters import OMEGA_S
    n_gen = 10
    delta = X[:, :n_gen]       # (batch, 10)
    omega = X[:, n_gen: 2*n_gen]
    Pm    = X[:, 2*n_gen:]

    # Electrical power batch: (batch, 10)
    E = swing.E_mag                            # (10,)
    G = swing.Y_red.real                       # (10, 10)
    B = swing.Y_red.imag

    # dij[b, i, j] = δ_bi - δ_bj
    dij = delta[:, :, None] - delta[:, None, :]   # (batch, 10, 10)
    Ei = E[None, :, None]   # (1, 10, 1)
    Ej = E[None, None, :]   # (1, 1, 10)
    Pe = np.sum(Ei * Ej * (G * np.cos(dij) + B * np.sin(dij)), axis=2)  # (batch, 10)

    H = swing.params.H   # (10,)
    D = swing.params.D
    two_H = 2.0 * H[None, :]  # (1, 10)

    d_delta = omega.copy()
    d_omega = (OMEGA_S / two_H) * (Pm - Pe) - (D[None, :] / two_H) * omega
    d_Pm    = np.zeros_like(Pm)

    return np.concatenate([d_delta, d_omega, d_Pm], axis=1)


def _h_batch(X: np.ndarray, meas_model, meas_channels: list[int]) -> np.ndarray:
    """Vectorised measurement function for batch (batch, 30) → (batch, n_z).

    Evaluates the linearised measurement model over all sigma points at once.
    Selects meas_channels from the 14-channel output per bus.
    """
    import numpy as np
    n_gen = 10
    n_pmu = 8
    n_chan = 14
    batch = X.shape[0]

    delta = X[:, :n_gen]   # (batch, 10)
    omega = X[:, n_gen: 2*n_gen]

    S    = meas_model.S        # (8, 10)
    d0   = meas_model.delta0   # (10,)
    theta0 = meas_model.theta0  # (8,) radians
    vmag0  = meas_model.vmag0   # (8,) p.u.
    I_ang0 = meas_model.I_ang0  # (8,) radians
    I_mag0 = meas_model.I_mag0  # (8,) p.u.
    H    = meas_model.H
    D    = meas_model.D

    from src.dynamics.measurement import F0, V_BASE_LN, I_BASE

    d_delta = delta - d0[None, :]               # (batch, 10)
    d_theta = d_delta @ S.T                     # (batch, 8) = (batch,10) @ (10,8)
    theta_k = theta0[None, :] + d_theta         # (batch, 8) radians
    theta_k_deg = np.rad2deg(theta_k)

    vmag_V   = np.broadcast_to(vmag0 * V_BASE_LN, (batch, n_pmu)).copy()   # (batch, 8) V
    I_ang_k_deg = np.rad2deg(I_ang0[None, :] + d_theta)                    # (batch, 8)
    I_mag_A  = np.broadcast_to(I_mag0 * I_BASE, (batch, n_pmu)).copy()    # (batch, 8) A

    d_theta_dt = omega @ S.T                    # (batch, 8)
    freq_k = F0 + d_theta_dt / (2 * np.pi)

    two_H = 2.0 * H[None, :]
    d_omega_dt = -(D[None, :] / two_H) * omega @ np.ones((10, 1))  # broadcast
    # ROCOF = S @ (dω/dt) / (2π): vectorised
    d_omega_arr = -(D[None, :] / two_H) * omega    # (batch, 10)
    rocof_k = (d_omega_arr @ S.T) / (2 * np.pi)   # (batch, 8)

    # Stack all 14 channels: shape (batch, 8, 14)
    y_all = np.stack([
        theta_k_deg,        # 0  VA_ANG (deg)
        vmag_V,             # 1  VA_MAG (V)
        theta_k_deg - 120., # 2  VB_ANG
        vmag_V,             # 3  VB_MAG
        theta_k_deg + 120., # 4  VC_ANG
        vmag_V,             # 5  VC_MAG
        I_ang_k_deg,        # 6  IA_ANG
        I_mag_A,            # 7  IA_MAG (A)
        I_ang_k_deg - 120., # 8  IB_ANG
        I_mag_A,            # 9  IB_MAG
        I_ang_k_deg + 120., # 10 IC_ANG
        I_mag_A,            # 11 IC_MAG
        freq_k,             # 12 Freq (Hz)
        rocof_k,            # 13 ROCOF (Hz/s)
    ], axis=2)   # (batch, 8, 14)

    # Select channels and flatten: (batch, n_pmu × n_selected)
    y_sel = y_all[:, :, meas_channels]  # (batch, 8, n_sel)
    return y_sel.reshape(batch, -1)     # (batch, n_z)


# ── UKF ───────────────────────────────────────────────────────────────────────

class UKF:
    """Standard UKF for the 30-state swing model with (4×8 = 32)-dim measurements.

    State layout:
        x[0:10]  = δ  (rotor angles, rad)
        x[10:20] = ω  (speed deviations, rad/s)
        x[20:30] = Pm (mechanical power, p.u.)

    Measurement vector z: flattened (n_pmu, n_meas_per_bus) array of selected channels.
    """

    def __init__(
        self,
        swing,            # SwingModel
        meas_model,       # MeasurementModel
        Q: np.ndarray,    # (30, 30) process noise
        R_base: np.ndarray,  # (n_z, n_z) base measurement noise (may be inflated per step)
        x0: np.ndarray,   # (30,) initial state
        P0: np.ndarray,   # (30, 30) initial covariance
        meas_channels: list[int] | None = None,  # channel indices into 14-ch output
        meas_offset: np.ndarray | None = None,   # (n_z,) bias correction
    ):
        self.swing = swing
        self.meas_model = meas_model
        self.Q = Q
        self.R_base = R_base
        self.x = x0.copy()
        self.P = P0.copy()
        self.meas_channels = meas_channels or [1, 7, 12, 13]
        self.n_z = len(PMU_BUSES_) * len(self.meas_channels)
        self.meas_offset = meas_offset if meas_offset is not None else np.zeros(self.n_z)

        n = len(x0)
        self.sp = MerweScaledSigmaPoints(n)

        # Storage for innovation diagnostics
        self.nu: np.ndarray | None = None       # last innovation
        self.S_innov: np.ndarray | None = None  # last innovation covariance
        self.eta: float = float("nan")          # normalised innovation statistic

    def predict(self, dt: float) -> None:
        """UKF predict step — propagate sigma points through swing RK4."""
        self.swing._dt_current = dt
        X = self.sp.sigma_points(self.x, self.P)   # (2n+1, 30)
        X_pred = _f_batch(X, self.swing)             # (2n+1, 30)

        x_pred = self.sp.weighted_mean(X_pred)
        P_pred = self.sp.weighted_cov(X_pred, x_pred, noise=self.Q)
        P_pred = _symmetrise(P_pred)

        self.x_pred = x_pred
        self.P_pred = P_pred
        self.X_pred = X_pred   # keep for update step (reuse same sigma points)

    def update(self, z: np.ndarray, R: np.ndarray | None = None) -> None:
        """UKF update step — incorporate measurement z.

        Args:
            z: (n_z,) measurement vector.
            R: (n_z, n_z) effective noise covariance (use R_base if None).
        """
        if R is None:
            R = self.R_base

        # Propagate cached sigma points through h
        Y = _h_batch(self.X_pred, self.meas_model, self.meas_channels)  # (2n+1, n_z)
        Y += self.meas_offset[None, :]   # apply bias correction

        y_pred = self.sp.weighted_mean(Y)             # (n_z,)
        S = self.sp.weighted_cov(Y, y_pred, noise=R)  # (n_z, n_z)
        Pxy = self.sp.weighted_cross_cov(self.X_pred, self.x_pred, Y, y_pred)  # (30, n_z)

        # Kalman gain: K = Pxy @ inv(S)
        try:
            S_chol, lower = scipy.linalg.cho_factor(S, lower=True, check_finite=False)
            K = scipy.linalg.cho_solve((S_chol, lower), Pxy.T, check_finite=False).T
        except (np.linalg.LinAlgError, scipy.linalg.LinAlgError):
            K = Pxy @ np.linalg.pinv(S)

        nu = z - y_pred
        self.x = self.x_pred + K @ nu
        self.P = _symmetrise(self.P_pred - K @ S @ K.T)
        self.P = _regularise(self.P)

        # Innovation statistics
        try:
            S_inv_nu = scipy.linalg.cho_solve((S_chol, lower), nu, check_finite=False)
            self.eta = float(nu @ S_inv_nu)
        except Exception:
            self.eta = float("nan")
        self.nu = nu
        self.S_innov = S

    def run(
        self,
        df: "pd.DataFrame",
        channel_cols: list[str],
        n_channels_per_bus: int,
        R_inflater: "Callable | None" = None,
    ) -> dict:
        """Run the UKF over the full DataFrame.

        Args:
            df: merged DataFrame (all 8 buses, TIMESTAMP, Event, BUSk_DATA_PRESENT).
            channel_cols: ordered list of column names matching the n_z measurement vector.
            n_channels_per_bus: channels per bus (used by R_inflater).
            R_inflater: optional callable (R_base, data_present_dict) → R_eff.

        Returns:
            dict with keys: x_est (T, 30), innovations (T, n_z), eta (T,), S_innov (T, n_z, n_z).
        """
        from src.estimator.nan_handling import data_present_flags, extract_z

        T = len(df)
        timestamps = df["TIMESTAMP"].to_numpy(float)
        x_est = np.empty((T, len(self.x)))
        innovations = np.empty((T, self.n_z))
        etas = np.empty(T)

        x_est[0] = self.x
        innovations[0] = np.zeros(self.n_z)
        etas[0] = 0.0

        for i in range(1, T):
            dt = float(timestamps[i] - timestamps[i - 1])
            if dt <= 0:
                dt = 1.0 / 30.0   # fallback for non-monotonic edge case

            self.predict(dt)

            row = df.iloc[i]
            z = extract_z(row, channel_cols)

            if R_inflater is not None:
                flags = data_present_flags(row)
                R_eff = R_inflater(self.R_base, flags, n_channels_per_bus)
            else:
                R_eff = None

            self.update(z, R_eff)

            x_est[i] = self.x
            innovations[i] = self.nu if self.nu is not None else np.zeros(self.n_z)
            etas[i] = self.eta

        return {"x_est": x_est, "innovations": innovations, "eta": etas}


# ── Helpers ───────────────────────────────────────────────────────────────────

# Imported here to avoid circular import in _h_batch
from src.io.load_csv import PMU_BUSES as PMU_BUSES_


def _symmetrise(P: np.ndarray) -> np.ndarray:
    return (P + P.T) * 0.5


def _regularise(P: np.ndarray, eps: float = _EPS) -> np.ndarray:
    """Add eps*I to ensure positive definiteness."""
    min_eig = np.linalg.eigvalsh(P).min()
    if min_eig < eps:
        P = P + (eps - min_eig + eps) * np.eye(P.shape[0])
    return P


def _cholesky_safe(P: np.ndarray) -> np.ndarray:
    """Cholesky with regularisation fallback."""
    try:
        return np.linalg.cholesky(P)
    except np.linalg.LinAlgError:
        n = P.shape[0]
        eps = abs(np.diag(P)).mean() * 1e-6 + _EPS
        for _ in range(10):
            try:
                return np.linalg.cholesky(P + eps * np.eye(n))
            except np.linalg.LinAlgError:
                eps *= 10
        # Final fallback: eigendecomposition
        w, v = np.linalg.eigh(P)
        w = np.maximum(w, _EPS)
        return v @ np.diag(np.sqrt(w))
