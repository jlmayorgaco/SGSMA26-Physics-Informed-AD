"""Residual distribution fitting and sampling utilities for m3."""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import ks_2samp, t as student_t_dist

from src.calibration.calibration_metrics import percentile_mismatch


def fit_residual_distribution(noise_stats, family, requested_model="auto"):
    if requested_model != "auto":
        return requested_model
    preferred = str(noise_stats.get("fitted_distribution_type", "gaussian"))
    kurt = float(noise_stats.get("kurtosis", 0.0))
    skewness = abs(float(noise_stats.get("skewness", 0.0)))
    residual_samples = noise_stats.get("residual_samples", []) or []
    if family in {"current_mag", "frequency"} and residual_samples and (preferred == "bootstrap" or kurt > 3.0):
        return "bootstrap"
    if family in {"current_mag", "frequency"} and (kurt > 1.5 or preferred == "student_t"):
        return "student_t"
    if skewness > 0.6 or preferred == "gmm":
        return "gmm"
    return "gaussian"


def candidate_noise_models(noise_stats, family):
    preferred = fit_residual_distribution(noise_stats, family, "auto")
    if family in {"current_mag", "frequency"}:
        modes = [preferred, "bootstrap", "student_t"]
        if abs(float(noise_stats.get("skewness", 0.0))) > 0.6:
            modes.append("gmm")
    else:
        modes = [preferred]
    out = []
    for mode in modes:
        if mode not in out:
            out.append(mode)
    return out


def sample_block_bootstrap(samples, n, rng, block=16):
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    if len(samples) == 0:
        return np.zeros(n)
    if len(samples) <= block:
        return rng.choice(samples, size=n, replace=True)
    out = []
    while len(out) < n:
        start = int(rng.integers(0, len(samples) - block + 1))
        out.extend(samples[start:start + block])
    return np.asarray(out[:n], dtype=float)


def sample_residual_process(noise_stats, n, rng, mode, noise_scale=1.0):
    if n <= 0:
        return np.array([], dtype=float)
    sigma = float(noise_stats.get("std_dev_abs", noise_stats.get("std_dev_raw", 1e-6))) * float(noise_scale)
    rho = float(np.clip(noise_stats.get("ar1_rho", 0.0), -0.999, 0.999))
    if mode == "bootstrap":
        innovations = sample_block_bootstrap(noise_stats.get("residual_samples", []), n, rng)
        if np.std(innovations) > 1e-12:
            innovations = innovations * (sigma / np.std(innovations))
    elif mode == "student_t":
        kurt = float(noise_stats.get("kurtosis", 3.0))
        if not np.isfinite(kurt):
            kurt = 3.0
        kurt = max(kurt, 0.1)
        df = float(np.clip(6.0 / kurt + 4.0, 3.0, 30.0))
        innovations = student_t_dist.rvs(df, size=n, random_state=rng)
        innovations = innovations / max(np.std(innovations), 1e-12) * sigma
    elif mode == "gmm":
        tail_prob = float(np.clip(noise_stats.get("p_outlier", 0.01), 0.01, 0.20))
        wide_sigma = max(float(noise_stats.get("outlier_mag_abs", 0.0)), 2.5 * sigma)
        mask = rng.random(n) < tail_prob
        innovations = rng.normal(0.0, sigma, n)
        innovations[mask] = rng.normal(0.0, wide_sigma, int(np.sum(mask)))
    else:
        innovations = rng.normal(0.0, sigma, n)

    white = innovations * math.sqrt(max(0.0, 1.0 - rho**2))
    noise = np.zeros(n)
    noise[0] = white[0]
    for i in range(1, n):
        noise[i] = rho * noise[i - 1] + white[i]

    p = float(np.clip(noise_stats.get("p_outlier", 0.0), 0.0, 1.0))
    mag = float(noise_stats.get("outlier_mag_abs", 0.0)) * float(noise_scale)
    if p > 0 and mag > 0:
        mask = rng.random(n) < p
        signs = rng.choice([-1.0, 1.0], size=n)
        spikes = mask * signs * np.abs(rng.normal(mag, max(mag * 0.25, 1e-12), n))
    else:
        spikes = 0.0
    return noise + spikes


def score_distribution_fit(real_residuals, sampled_residuals):
    real_residuals = np.asarray(real_residuals, dtype=float)
    sampled_residuals = np.asarray(sampled_residuals, dtype=float)
    real_residuals = real_residuals[np.isfinite(real_residuals)]
    sampled_residuals = sampled_residuals[np.isfinite(sampled_residuals)]
    if len(real_residuals) == 0 or len(sampled_residuals) == 0:
        return np.inf
    ks = float(ks_2samp(real_residuals, sampled_residuals).statistic)
    qerr = percentile_mismatch(real_residuals, sampled_residuals)
    return 0.7 * ks + 0.3 * qerr
