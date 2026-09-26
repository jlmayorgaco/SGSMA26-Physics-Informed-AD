"""Affine and quantile calibration fit helpers."""

from __future__ import annotations

import numpy as np


def fit_affine_distribution(sim, real, family):
    sim = np.asarray(sim, dtype=float)
    real = np.asarray(real, dtype=float)
    sim = sim[np.isfinite(sim)]
    real = real[np.isfinite(real)]
    if len(sim) == 0 or len(real) == 0:
        return 1.0, 0.0, "identity_empty"
    if np.std(sim) < 1e-12:
        return 0.0, float(np.median(real)), "median_constant"
    if family == "current_mag":
        qs = [1, 5, 50, 95, 99]
        w = np.asarray([1.5, 1.5, 1.0, 2.0, 2.5])
    else:
        qs = [5, 50, 95]
        w = np.ones(3)
    sx = np.percentile(sim, qs)
    ry = np.percentile(real, qs)
    xbar = np.average(sx, weights=w)
    ybar = np.average(ry, weights=w)
    denom = np.sum(w * (sx - xbar) ** 2)
    if denom < 1e-18:
        return 0.0, float(np.median(real)), "median_constant"
    a = float(np.sum(w * (sx - xbar) * (ry - ybar)) / denom)
    b = float(ybar - a * xbar)
    return a, b, "weighted_percentile_affine"


def fit_quantile_mapping(sim, real, n_quantiles=31):
    sim = np.asarray(sim, dtype=float)
    real = np.asarray(real, dtype=float)
    sim = sim[np.isfinite(sim)]
    real = real[np.isfinite(real)]
    if len(sim) < 8 or len(real) < 8:
        return None
    qs = np.linspace(0.01, 0.99, n_quantiles)
    src = np.quantile(sim, qs)
    dst = np.quantile(real, qs)
    keep = np.r_[True, np.diff(src) > 1e-12]
    if np.sum(keep) < 3:
        return None
    return {"src": src[keep].tolist(), "dst": dst[keep].tolist()}


def apply_quantile_mapping(values, mapping):
    if mapping is None:
        return values
    src = np.asarray(mapping["src"], dtype=float)
    dst = np.asarray(mapping["dst"], dtype=float)
    return np.interp(np.asarray(values, dtype=float), src, dst, left=dst[0], right=dst[-1])


def condition_prepared_to_chunk(prepared, real_chunk, family):
    if family not in {"voltage_mag", "current_mag", "frequency"}:
        out = dict(prepared)
        out["chunk_conditioning_used"] = "no"
        return out
    mapping = fit_quantile_mapping(prepared["sim_noisy"], real_chunk, n_quantiles=21)
    if mapping is None:
        out = dict(prepared)
        out["chunk_conditioning_used"] = "no"
        return out
    out = dict(prepared)
    out["sim_clean"] = apply_quantile_mapping(prepared["sim_clean"], mapping)
    out["sim_drifted"] = apply_quantile_mapping(prepared["sim_drifted"], mapping)
    out["sim_noisy"] = apply_quantile_mapping(prepared["sim_noisy"], mapping)
    out["fit_mode"] = f"{prepared['fit_mode']}+chunk_quantile_conditioning"
    out["chunk_conditioning_used"] = "yes"
    return out
