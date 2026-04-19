"""Raw PMU noise profiler for RAW0001 event chunks.

This module is intentionally separate from ``m2_noise_profiling.py`` so the
normalized workflow remains untouched. The output is keyed by exact raw CSV
column names:

    event_type -> bus_id -> raw_column_name -> profile

The profile contains both classic residual noise parameters and richer
distribution/spectral summaries used by ``m3_andes_calibration_raw.py``.
"""

from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, welch
from scipy.stats import kurtosis, skew


CHUNKS_DIR = "output/SCENARIO_RAW0001/chunks"
PROFILE_OUT = "output/SCENARIO_RAW0001/dataset_profiles_raw.json"
RESIDUAL_SAMPLE_LIMIT = 2048


def signal_family(column_name: str) -> str:
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


def infer_sample_rate_hz(timestamps: np.ndarray | None) -> float:
    if timestamps is None:
        return 30.0
    t = pd.to_numeric(pd.Series(timestamps), errors="coerce").dropna().to_numpy(float)
    if len(t) < 3:
        return 30.0
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return 30.0
    return float(1.0 / np.median(dt))


def safe_odd_window(n: int, target: int) -> int:
    if n < 5:
        return max(1, n)
    win = min(target, n)
    if win % 2 == 0:
        win -= 1
    return max(5, win)


def estimate_trend(y: np.ndarray, family: str) -> tuple[np.ndarray, str]:
    if len(y) < 7:
        return np.full_like(y, np.nanmedian(y)), "median_short_series"

    if family in {"current_mag", "voltage_mag", "frequency"}:
        win = safe_odd_window(len(y), 301)
        trend = (
            pd.Series(y)
            .rolling(window=win, center=True, min_periods=max(5, win // 5))
            .median()
            .bfill()
            .ffill()
            .to_numpy(dtype=float)
        )
        return trend, f"rolling_median_w{win}"

    win = safe_odd_window(len(y), 101)
    poly = 3 if win >= 7 else 2
    return savgol_filter(y, win, polyorder=poly), f"savgol_w{win}_p{poly}"


def robust_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 0.0
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    sigma = 1.4826 * mad
    return float(sigma if sigma > 1e-12 else np.std(x))


def autocorr(x: np.ndarray, lag: int) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) <= lag or np.std(x) < 1e-15:
        return 0.0
    return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])


def quantile_summary(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {f"p{p:02d}": 0.0 for p in [1, 5, 50, 95, 99]}
    vals = np.percentile(x, [1, 5, 50, 95, 99])
    return {name: float(val) for name, val in zip(["p01", "p05", "p50", "p95", "p99"], vals)}


def spectral_summary(x: np.ndarray, fs: float) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 8 or np.std(x) < 1e-15:
        return {
            "sample_rate_hz": float(fs),
            "lowfreq_power": 0.0,
            "total_power": 0.0,
            "lowfreq_power_ratio": 0.0,
            "spectral_centroid_hz": 0.0,
            "dominant_frequency_hz": 0.0,
            "dominant_amplitude": 0.0,
            "top_frequencies_hz": [],
            "top_amplitudes": [],
        }

    centered = x - np.mean(x)
    nperseg = min(512, len(centered))
    freqs, psd = welch(centered, fs=fs, nperseg=nperseg, detrend="constant")
    total = float(np.trapezoid(psd, freqs)) if len(freqs) > 1 else float(np.sum(psd))
    low_mask = (freqs > 0.0) & (freqs <= 0.25)
    low = float(np.trapezoid(psd[low_mask], freqs[low_mask])) if np.any(low_mask) else 0.0
    denom = max(float(np.sum(psd)), 1e-30)
    centroid = float(np.sum(freqs * psd) / denom)

    fft_freq = np.fft.rfftfreq(len(centered), d=1.0 / fs)
    fft_mag = np.abs(np.fft.rfft(centered)) / max(len(centered), 1)
    if len(fft_mag) > 1:
        order = np.argsort(fft_mag[1:])[-5:][::-1] + 1
        dominant = int(order[0])
    else:
        order = np.array([], dtype=int)
        dominant = 0

    return {
        "sample_rate_hz": float(fs),
        "lowfreq_power": low,
        "total_power": total,
        "lowfreq_power_ratio": float(low / max(total, 1e-30)),
        "spectral_centroid_hz": centroid,
        "dominant_frequency_hz": float(fft_freq[dominant]) if dominant else 0.0,
        "dominant_amplitude": float(fft_mag[dominant]) if dominant else 0.0,
        "top_frequencies_hz": [float(fft_freq[i]) for i in order],
        "top_amplitudes": [float(fft_mag[i]) for i in order],
    }


def choose_distribution_type(residual: np.ndarray, family: str) -> str:
    residual = residual[np.isfinite(residual)]
    if len(residual) < 32:
        return "gaussian"
    sk = abs(float(skew(residual, bias=False)))
    ku = float(kurtosis(residual, fisher=True, bias=False))
    tail_ratio = np.quantile(np.abs(residual), 0.99) / max(np.quantile(np.abs(residual), 0.75), 1e-12)
    if family in {"current_mag", "frequency"} and (ku > 4.0 or tail_ratio > 4.0):
        return "bootstrap"
    if ku > 2.0:
        return "student_t"
    if sk > 0.75:
        return "gmm"
    return "gaussian"


def sample_residuals(residual: np.ndarray, limit: int = RESIDUAL_SAMPLE_LIMIT) -> list[float]:
    residual = np.asarray(residual, dtype=float)
    residual = residual[np.isfinite(residual)]
    if len(residual) == 0:
        return []
    if len(residual) <= limit:
        sample = residual
    else:
        idx = np.linspace(0, len(residual) - 1, limit).astype(int)
        sample = residual[idx]
    return [float(x) for x in sample]


def analyze_raw_signal(values: np.ndarray, column_name: str, timestamps: np.ndarray | None = None) -> dict | None:
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


def weighted_average(items: list[dict], key: str, weights: np.ndarray) -> float:
    vals = np.asarray([float(item.get(key, 0.0)) for item in items], dtype=float)
    if len(vals) == 0:
        return 0.0
    return float(np.average(vals, weights=weights)) if np.sum(weights) > 0 else float(np.mean(vals))


def merge_spectral(items: list[dict], key: str, weights: np.ndarray) -> dict:
    specs = [item.get(key, {}) or {} for item in items]
    numeric = [
        "sample_rate_hz",
        "lowfreq_power",
        "total_power",
        "lowfreq_power_ratio",
        "spectral_centroid_hz",
        "dominant_frequency_hz",
        "dominant_amplitude",
    ]
    out = {name: weighted_average(specs, name, weights) for name in numeric}
    first_top_f = next((s.get("top_frequencies_hz", []) for s in specs if s.get("top_frequencies_hz")), [])
    first_top_a = next((s.get("top_amplitudes", []) for s in specs if s.get("top_amplitudes")), [])
    out["top_frequencies_hz"] = [float(x) for x in first_top_f]
    out["top_amplitudes"] = [float(x) for x in first_top_a]
    return out


def aggregate_entries(entries: list[dict]) -> dict:
    weights = np.asarray([max(1, e.get("sample_count", 1)) for e in entries], dtype=float)
    eda_keys = entries[0]["eda_stats"].keys()
    diag_keys = [k for k in entries[0]["diagnostics"].keys() if k != "signal_psd_summary"]
    noise_keys = [
        "std_dev_abs",
        "std_dev_raw",
        "ar1_rho",
        "ar5_rho",
        "p_outlier",
        "outlier_mag_abs",
        "outlier_threshold_abs",
        "skewness",
        "kurtosis",
    ]

    eda = {k: weighted_average([e["eda_stats"] for e in entries], k, weights) for k in eda_keys}
    diag = {k: weighted_average([e["diagnostics"] for e in entries], k, weights) for k in diag_keys}
    diag["signal_psd_summary"] = merge_spectral([e["diagnostics"] for e in entries], "signal_psd_summary", weights)

    residual_samples: list[float] = []
    for entry in entries:
        residual_samples.extend(entry.get("noise_model", {}).get("residual_samples", []))
    if len(residual_samples) > RESIDUAL_SAMPLE_LIMIT:
        idx = np.linspace(0, len(residual_samples) - 1, RESIDUAL_SAMPLE_LIMIT).astype(int)
        residual_samples = [float(residual_samples[i]) for i in idx]

    noise = {k: weighted_average([e["noise_model"] for e in entries], k, weights) for k in noise_keys}
    families = [e["noise_model"]["signal_family"] for e in entries]
    distributions = [e["noise_model"]["fitted_distribution_type"] for e in entries]
    trends = [e["noise_model"]["trend_method"] for e in entries]
    noise["signal_family"] = max(set(families), key=families.count)
    noise["fitted_distribution_type"] = max(set(distributions), key=distributions.count)
    noise["trend_method"] = max(set(trends), key=trends.count)
    noise["profile_source"] = "raw_event_chunk_aggregate"
    noise["psd_summary"] = merge_spectral([e["noise_model"] for e in entries], "psd_summary", weights)
    noise["fft_low_freq_summary"] = merge_spectral([e["noise_model"] for e in entries], "fft_low_freq_summary", weights)
    noise["quantile_summary"] = {
        k: weighted_average([e["noise_model"]["quantile_summary"] for e in entries], k, weights)
        for k in entries[0]["noise_model"]["quantile_summary"].keys()
    }
    noise["residual_samples"] = residual_samples

    profile = {"eda_stats": eda, "noise_model": noise, "diagnostics": diag, "sample_count": int(np.sum(weights))}
    profile.update(noise)
    return profile


def generate_raw_profiles(chunks_dir: str = CHUNKS_DIR, out_path: str = PROFILE_OUT) -> dict:
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    chunk_dirs = sorted(glob.glob(os.path.join(chunks_dir, "chunk*_event_*")))
    for chunk_dir in chunk_dirs:
        name = os.path.basename(chunk_dir)
        try:
            event_type = str(int(name.split("_")[2]))
        except (IndexError, ValueError):
            continue

        for csv_path in sorted(glob.glob(os.path.join(chunk_dir, "Bus*.csv"))):
            bus_id = os.path.basename(csv_path).replace(".csv", "").replace("Bus", "")
            df = pd.read_csv(csv_path)
            timestamps = df["TIMESTAMP"].to_numpy() if "TIMESTAMP" in df.columns else None
            for col in df.columns:
                if col in {"TIMESTAMP", "DATA_PRESENT", "Event"}:
                    continue
                values = pd.to_numeric(df[col], errors="coerce").to_numpy()
                analysis = analyze_raw_signal(values, col, timestamps)
                if analysis is not None:
                    acc[event_type][bus_id][col].append(analysis)

    profiles = {}
    for event_type, buses in acc.items():
        profiles[event_type] = {}
        for bus_id, signals in buses.items():
            profiles[event_type][bus_id] = {}
            for col, entries in signals.items():
                profiles[event_type][bus_id][col] = aggregate_entries(entries)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(profiles, fh, indent=2)
    return profiles


class RawPMUNoiseLayer:
    """Resolver for exact raw profiles with explicit fallback status."""

    def __init__(self, profile_path: str = PROFILE_OUT):
        with open(profile_path, "r", encoding="utf-8") as fh:
            self.profiles = json.load(fh)
        self.missing = set()

    def resolve_stats(self, bus_id, event_type, raw_column_name):
        bus_id, event_type = str(bus_id), str(event_type)
        if event_type in self.profiles and bus_id in self.profiles[event_type]:
            sigs = self.profiles[event_type][bus_id]
            if raw_column_name in sigs:
                return sigs[raw_column_name], "exact_match"
        if "0" in self.profiles and bus_id in self.profiles["0"]:
            sigs = self.profiles["0"][bus_id]
            if raw_column_name in sigs:
                return sigs[raw_column_name], "fallback_to_0"

        key = (bus_id, event_type, raw_column_name)
        if key not in self.missing:
            print(f"[WARN] raw noise profile missing: bus={bus_id}, event={event_type}, signal={raw_column_name}")
            self.missing.add(key)
        return {
            "std_dev_abs": 1e-6,
            "std_dev_raw": 1e-6,
            "ar1_rho": 0.0,
            "ar5_rho": 0.0,
            "p_outlier": 0.0,
            "outlier_mag_abs": 0.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
            "fitted_distribution_type": "gaussian",
            "signal_family": "unknown",
            "psd_summary": {},
            "fft_low_freq_summary": {},
            "quantile_summary": {},
            "residual_samples": [],
            "profile_source": "default",
        }, "default_fallback"

    def apply(self, bus_id, event_type, t, y, raw_column_name):
        stats, status = self.resolve_stats(bus_id, event_type, raw_column_name)
        y = np.asarray(y, dtype=float)
        n = len(y)
        if n == 0:
            return y, status

        sigma = float(stats.get("std_dev_abs", stats.get("std_dev_raw", 1e-6)))
        rho = float(np.clip(stats.get("ar1_rho", 0.0), -0.999, 0.999))
        white = np.random.normal(0.0, sigma * np.sqrt(max(0.0, 1.0 - rho**2)), n)
        noise = np.zeros(n)
        noise[0] = white[0]
        for i in range(1, n):
            noise[i] = rho * noise[i - 1] + white[i]

        p = float(np.clip(stats.get("p_outlier", 0.0), 0.0, 1.0))
        mag = float(stats.get("outlier_mag_abs", 0.0))
        if p > 0 and mag > 0:
            mask = np.random.choice([0, 1], size=n, p=[1.0 - p, p])
            spikes = mask * np.random.choice([-1.0, 1.0], size=n) * np.abs(
                np.random.normal(mag, max(mag * 0.25, 1e-12), n)
            )
        else:
            spikes = 0.0
        return y + noise + spikes, status


if __name__ == "__main__":
    profiles = generate_raw_profiles()
    n_event0 = sum(len(signals) for signals in profiles.get("0", {}).values())
    print(f"Raw profiles saved to {PROFILE_OUT}")
    print(f"Event 0 raw signal profiles: {n_event0}")
