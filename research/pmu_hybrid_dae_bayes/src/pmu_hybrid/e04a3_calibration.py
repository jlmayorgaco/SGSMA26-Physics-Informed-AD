"""Leakage-safe uncertainty calibration utilities for E04-A3."""
from __future__ import annotations
import numpy as np


def nis_raw_normalized(innovation: np.ndarray, innovation_cov: np.ndarray) -> tuple[float, float]:
    """Return raw NIS and dimension-normalized NIS for one innovation."""
    v = np.asarray(innovation, dtype=float)
    s = np.asarray(innovation_cov, dtype=float)
    raw = float(v @ np.linalg.solve(s, v))
    return raw, raw / v.size


def gaussian_nll(errors: np.ndarray, covariances: np.ndarray, tau: float = 1.0,
                 block_tau: tuple[float, float] | None = None) -> float:
    """Mean 2-D Gaussian NLL under scalar or Re/Im block temperature."""
    e = np.asarray(errors, dtype=float)
    c = np.asarray(covariances, dtype=float)
    if block_tau is None:
        scales = np.full(e.shape[:-1] + (2,), float(tau))
    else:
        scales = np.empty(e.shape[:-1] + (2,)); scales[..., 0] = block_tau[0]; scales[..., 1] = block_tau[1]
    v = c * np.sqrt(scales[..., :, None] * scales[..., None, :])
    inv = np.linalg.inv(v); sign, logdet = np.linalg.slogdet(v)
    if np.any(sign <= 0): raise ValueError("temperature covariance is not positive definite")
    quad = np.einsum("...i,...ij,...j->...", e, inv, e)
    return float(np.mean(0.5 * (2 * np.log(2 * np.pi) + logdet + quad)))


def fit_scalar_temperature(errors: np.ndarray, covariances: np.ndarray) -> tuple[float, float, float]:
    """Fit scalar tau by NLL; return (tau_nll, tau_empirical, nll)."""
    e = np.asarray(errors, dtype=float); c = np.asarray(covariances, dtype=float)
    inv = np.linalg.inv(c); quad = np.einsum("...i,...ij,...j->...", e, inv, e)
    tau = max(float(np.mean(quad) / e.shape[-1]), 1e-12)
    return tau, tau, gaussian_nll(e, c, tau=tau)


def fit_block_temperature(errors: np.ndarray, covariances: np.ndarray) -> tuple[float, float, float]:
    """Fit independent Re/Im scales using marginal standardized squared errors."""
    e = np.asarray(errors, dtype=float); c = np.asarray(covariances, dtype=float)
    tau_re = max(float(np.mean(e[..., 0] ** 2 / c[..., 0, 0])), 1e-12)
    tau_im = max(float(np.mean(e[..., 1] ** 2 / c[..., 1, 1])), 1e-12)
    return tau_re, tau_im, gaussian_nll(e, c, block_tau=(tau_re, tau_im))


def coverage(errors: np.ndarray, covariances: np.ndarray, level: float = 0.95,
             tau: float = 1.0, block_tau: tuple[float, float] | None = None) -> tuple[float, float]:
    """Marginal Re/Im Gaussian coverage at a central interval level."""
    e = np.asarray(errors, dtype=float); c = np.asarray(covariances, dtype=float)
    if block_tau is None: scales = (tau, tau)
    else: scales = block_tau
    z = {0.50: 0.6744897501960817, 0.90: 1.6448536269514722, 0.95: 1.959963984540054}[round(level, 2)]
    sd = np.sqrt(np.maximum(np.stack([c[..., 0, 0] * scales[0], c[..., 1, 1] * scales[1]], axis=-1), 1e-30))
    hit = np.abs(e) <= z * sd
    return float(np.mean(hit[..., 0])), float(np.mean(hit[..., 1]))


def innovation_acf(series: np.ndarray, lags=(1, 2, 3, 5, 10)) -> dict[int, float]:
    """ACF of a scalar sequence, safely centered once."""
    x = np.asarray(series, dtype=float).ravel(); x = x - x.mean(); den = float(x @ x)
    return {int(k): (float(x[:-k] @ x[k:] / den) if den > 0 and k < len(x) else 0.0) for k in lags}


def low_rank_spectrum(innovations: np.ndarray, ranks=(1, 2, 4, 8, 16)) -> tuple[np.ndarray, np.ndarray]:
    """Return eigenvalues and cumulative variance explained at requested ranks."""
    x = np.asarray(innovations, dtype=float); x = x - x.mean(axis=0, keepdims=True)
    eig = np.linalg.eigvalsh((x.T @ x) / max(len(x) - 1, 1))[::-1]; cum = np.cumsum(eig) / max(eig.sum(), 1e-30)
    return eig, np.asarray([cum[min(int(r), len(cum)) - 1] for r in ranks])
