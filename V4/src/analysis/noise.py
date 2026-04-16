from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.spectral import spectral_summary
from src.utils.numeric import autocorr_lag1, mad, robust_zscore
from src.utils.signals import rolling_trend, unwrap_if_angle


def distribution_fit_summary(residual: np.ndarray) -> dict:
    x = np.asarray(residual, dtype=float)
    x = x[np.isfinite(x)]

    out = {}
    if x.size < 20:
        return out

    mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=1))
    med = float(np.median(x))
    b = float(np.mean(np.abs(x - med)))

    out["gaussian"] = {"mu": mu, "sigma": sigma}
    out["laplace"] = {"mu": med, "b": b}

    # Lightweight best-family approximation without scipy.
    # You can upgrade later with scipy.stats fits.
    centered = x - mu
    std = max(np.std(x), 1e-12)
    kurt = float(np.mean((centered / std) ** 4) - 3.0)

    if kurt > 1.0:
        best = "student_t"
    elif kurt > 0.2:
        best = "laplace"
    else:
        best = "gaussian"

    out["moment_summary"] = {"excess_kurtosis": kurt}
    out["best_family"] = best
    return out


def noise_model(values: np.ndarray, dt: float, column: str, trend_window_seconds: float) -> dict:
    raw = np.asarray(values, dtype=float)
    proc = unwrap_if_angle(raw, column)

    y = pd.Series(proc)
    window_samples = max(5, int(round(trend_window_seconds / max(dt, 1e-9))))
    trend = rolling_trend(y, window_samples).to_numpy(dtype=float)
    residual = proc - trend

    valid_res = residual[np.isfinite(residual)]
    valid_trend = trend[np.isfinite(trend)]

    if valid_res.size == 0:
        return {}

    res_std = float(np.std(valid_res, ddof=1)) if valid_res.size > 1 else 0.0
    trend_std = float(np.std(valid_trend, ddof=1)) if valid_trend.size > 1 else 0.0

    spectral = spectral_summary(valid_res, 1.0 / dt)

    return {
        "trend_std": trend_std,
        "residual_std": res_std,
        "residual_mad": float(mad(valid_res)),
        "lag1_autocorr": autocorr_lag1(valid_res),
        "snr_db": float(20.0 * np.log10(max(trend_std, 1e-12) / max(res_std, 1e-12))),
        "outlier_rate_abs_z_gt_3": float(np.mean(np.abs(robust_zscore(valid_res)) > 3.0)),
        "outlier_rate_abs_z_gt_5": float(np.mean(np.abs(robust_zscore(valid_res)) > 5.0)),
        "distribution_fit": distribution_fit_summary(valid_res),
        "spectral": spectral,
    }