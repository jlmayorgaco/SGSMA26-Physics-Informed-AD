from __future__ import annotations

import numpy as np

from src.config.constants import (
    PHASE_CURRENT_MAG_COLUMNS,
    PHASE_VOLTAGE_ANG_COLUMNS,
    PHASE_VOLTAGE_MAG_COLUMNS,
)
from src.utils.signals import safe_gradient, unwrap_if_angle


def basic_stats(values: np.ndarray, dt: float, column: str) -> dict:
    raw = np.asarray(values, dtype=float)
    proc = unwrap_if_angle(raw, column)
    valid = proc[np.isfinite(proc)]

    out = {
        "n_total": int(len(raw)),
        "n_valid": int(np.isfinite(raw).sum()),
        "n_missing": int(np.isnan(raw).sum()),
        "missing_ratio": float(np.isnan(raw).mean()) if len(raw) else None,
    }

    if valid.size == 0:
        return out

    out.update({
        "mean": float(np.mean(valid)),
        "median": float(np.median(valid)),
        "std": float(np.std(valid, ddof=1)) if valid.size > 1 else 0.0,
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "peak_to_peak": float(np.ptp(valid)),
        "rms": float(np.sqrt(np.mean(valid ** 2))),
        "energy": float(np.sum(valid ** 2) * dt),
        "q01": float(np.quantile(valid, 0.01)),
        "q05": float(np.quantile(valid, 0.05)),
        "q25": float(np.quantile(valid, 0.25)),
        "q75": float(np.quantile(valid, 0.75)),
        "q95": float(np.quantile(valid, 0.95)),
        "q99": float(np.quantile(valid, 0.99)),
    })

    centered = valid - np.mean(valid)
    std = np.std(valid, ddof=0)
    if std > 1e-12:
        out["skew"] = float(np.mean((centered / std) ** 3))
        out["kurtosis_excess"] = float(np.mean((centered / std) ** 4) - 3.0)

    deriv = safe_gradient(proc, dt)
    deriv = deriv[np.isfinite(deriv)]
    if deriv.size:
        out["derivative_mean"] = float(np.mean(deriv))
        out["derivative_std"] = float(np.std(deriv, ddof=1)) if deriv.size > 1 else 0.0
        out["derivative_abs_max"] = float(np.max(np.abs(deriv)))

    return out


def phase_balance_metrics(df) -> dict:
    out = {}

    if all(col in df.columns for col in PHASE_VOLTAGE_MAG_COLUMNS):
        mags = df[PHASE_VOLTAGE_MAG_COLUMNS].to_numpy(dtype=float)
        mean_mag = np.nanmean(mags, axis=1)
        dev = np.nanmax(np.abs(mags - mean_mag[:, None]), axis=1)
        ratio = dev / np.maximum(np.abs(mean_mag), 1e-12)
        out["voltage_phase_imbalance_ratio_mean"] = float(np.nanmean(ratio))
        out["voltage_phase_imbalance_ratio_p95"] = float(np.nanquantile(ratio, 0.95))

    if all(col in df.columns for col in PHASE_CURRENT_MAG_COLUMNS):
        mags = df[PHASE_CURRENT_MAG_COLUMNS].to_numpy(dtype=float)
        mean_mag = np.nanmean(mags, axis=1)
        dev = np.nanmax(np.abs(mags - mean_mag[:, None]), axis=1)
        ratio = dev / np.maximum(np.abs(mean_mag), 1e-12)
        out["current_phase_imbalance_ratio_mean"] = float(np.nanmean(ratio))
        out["current_phase_imbalance_ratio_p95"] = float(np.nanquantile(ratio, 0.95))

    if all(col in df.columns for col in PHASE_VOLTAGE_ANG_COLUMNS):
        va = unwrap_if_angle(df["VA_ang"].to_numpy(dtype=float), "VA_ang")
        vb = unwrap_if_angle(df["VB_ang"].to_numpy(dtype=float), "VB_ang")
        vc = unwrap_if_angle(df["VC_ang"].to_numpy(dtype=float), "VC_ang")

        sep_ab = np.mod(va - vb, 360.0)
        sep_bc = np.mod(vb - vc, 360.0)
        sep_ca = np.mod(vc - va, 360.0)

        err = np.vstack([
            np.abs(sep_ab - 120.0),
            np.abs(sep_bc - 120.0),
            np.abs(sep_ca - 120.0),
        ])

        out["voltage_angle_separation_error_mean_deg"] = float(np.nanmean(err))
        out["voltage_angle_separation_error_p95_deg"] = float(np.nanquantile(err, 0.95))

    return out