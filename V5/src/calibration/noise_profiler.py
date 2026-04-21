"""Raw-signal noise profiling core logic (m2 migrated implementation)."""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew

from src.calibration.residual_models import autocorr, choose_distribution_type, quantile_summary, robust_sigma, sample_residuals
from src.calibration.spectral_features import infer_sample_rate_hz, spectral_summary
from src.calibration.trend_estimation import estimate_trend


def signal_family(column_name: str) -> str:
    """Infer signal family from exact raw column name."""
    name = column_name.upper()
    if name.endswith("_MAG"):
        if any(tag in name for tag in ["_IA_MAG", "_IB_MAG", "_IC_MAG"]):
            return "current_mag"
        return "voltage_mag"
    if name.endswith("_ANG"):
        if any(tag in name for tag in ["_IA_ANG", "_IB_ANG", "_IC_ANG"]):
            return "angle_current"
        return "angle_voltage"
    if name.endswith("_FREQ"):
        return "frequency"
    if name.endswith("_ROCOF"):
        return "rocof"
    return "other"


def analyze_raw_signal(
    values: np.ndarray,
    column_name: str,
    timestamps: np.ndarray | None = None,
) -> dict | None:
    """Build one raw-signal profile from one chunk slice."""
    family = signal_family(column_name)
    y = np.asarray(values, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) < 8:
        return None

    fs = infer_sample_rate_hz(timestamps)
    trend, trend_method = estimate_trend(y, family)
    residual = y - trend
    residual = residual[np.isfinite(residual)]
    if len(residual) == 0:
        return None

    sigma_abs = robust_sigma(residual)
    abs_res = np.abs(residual)
    threshold = max(3.5 * sigma_abs, float(np.quantile(abs_res, 0.995)))
    outlier_mask = abs_res > threshold
    outliers = residual[outlier_mask]
    clipped = np.clip(residual, np.quantile(residual, 0.01), np.quantile(residual, 0.99))

    fitted_type = choose_distribution_type(residual, family)
    sig_spec = spectral_summary(y, fs)
    res_spec = spectral_summary(residual, fs)
    res_skew = float(skew(residual, bias=False)) if len(residual) > 2 else 0.0
    res_kurt = float(kurtosis(residual, fisher=True, bias=False)) if len(residual) > 3 else 0.0

    eda_stats = {
        "mean": float(np.mean(y)),
        "std": float(np.std(y)),
        "median": float(np.median(y)),
        "min": float(np.min(y)),
        "max": float(np.max(y)),
        "n_samples": int(len(y)),
        **quantile_summary(y),
    }

    noise_model = {
        "std_dev_abs": float(sigma_abs),
        "std_dev_raw": float(sigma_abs),
        "ar1_rho": autocorr(clipped, 1),
        "ar5_rho": autocorr(clipped, 5),
        "p_outlier": float(np.mean(outlier_mask)),
        "outlier_mag_abs": float(np.mean(np.abs(outliers))) if len(outliers) else 0.0,
        "outlier_threshold_abs": float(threshold),
        "skewness": res_skew,
        "kurtosis": res_kurt,
        "fitted_distribution_type": fitted_type,
        "signal_family": family,
        "psd_summary": res_spec,
        "fft_low_freq_summary": {
            "dominant_frequency_hz": res_spec["dominant_frequency_hz"],
            "dominant_amplitude": res_spec["dominant_amplitude"],
            "lowfreq_power_ratio": res_spec["lowfreq_power_ratio"],
            "top_frequencies_hz": res_spec["top_frequencies_hz"],
            "top_amplitudes": res_spec["top_amplitudes"],
        },
        "quantile_summary": quantile_summary(residual),
        "residual_samples": sample_residuals(residual),
        "profile_source": "raw_event_chunk",
        "trend_method": trend_method,
    }

    diagnostics = {
        "residual_mean": float(np.mean(residual)),
        "residual_std": float(np.std(residual)),
        "residual_skew": res_skew,
        "residual_kurtosis": res_kurt,
        "residual_lag1": autocorr(residual, 1),
        "residual_lag5": autocorr(residual, 5),
        "signal_psd_summary": sig_spec,
    }

    profile = {
        "eda_stats": eda_stats,
        "noise_model": noise_model,
        "diagnostics": diagnostics,
        "sample_count": int(len(y)),
    }
    profile.update(noise_model)
    return profile
