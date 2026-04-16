#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy import stats
    from scipy.signal import welch
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False
    stats = None
    welch = None


MEASUREMENT_COLUMNS = [
    "VA_mag", "VA_ang", "VB_mag", "VB_ang", "VC_mag", "VC_ang",
    "IA_mag", "IA_ang", "IB_mag", "IB_ang", "IC_mag", "IC_ang",
    "Frequency", "ROCOF",
]
OPTIONAL_COLUMNS = ["DATA_PRESENT", "Event"]
ANGLE_COLUMNS = {"VA_ang", "VB_ang", "VC_ang", "IA_ang", "IB_ang", "IC_ang"}
DEFAULT_EVENT_LABELS = {
    0: "Normal operation",
    1: "Fault",
    2: "Line outage",
    3: "Generation change/outage",
    4: "Load change/drop",
    5: "Missing data",
    6: "Missing data + physical event",
    7: "Bad data",
    8: "Unknown event",
}


@dataclass
class BusData:
    bus_id: str
    path: Path
    df: pd.DataFrame
    sampling_rate_hz: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose the 8 PMU CSV files from the SGSMA IEEE-39 package and export a fully "
            "parameterized realism model JSON for post-processing ANDES simulations."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing Bus*_Competition_Data*.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where diagnostics, plots, and JSON outputs will be stored.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="Bus*_Competition_Data*.csv",
        help="Glob pattern for PMU CSV files.",
    )
    parser.add_argument(
        "--trend-window-seconds",
        type=float,
        default=1.0,
        help="Rolling-median window in seconds used to estimate the slow trend for noise modeling.",
    )
    parser.add_argument(
        "--event-baseline-seconds",
        type=float,
        default=1.0,
        help="Baseline window length before event onset for event-profile statistics.",
    )
    parser.add_argument(
        "--event-post-seconds",
        type=float,
        default=1.0,
        help="Post-event recovery window length for event-profile statistics.",
    )
    parser.add_argument(
        "--artifact-z-threshold",
        type=float,
        default=6.0,
        help="Robust z-score threshold for artifact detection on residuals and derivatives.",
    )
    parser.add_argument(
        "--save-per-bus-json",
        action="store_true",
        help="Save per-bus summary JSON files in addition to the super-model JSON.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_py(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_py(v) for v in obj]
    return obj


def infer_bus_id(path: Path) -> str:
    m = re.search(r"Bus(\d+)", path.stem, re.IGNORECASE)
    if m:
        return f"Bus{int(m.group(1))}"
    return path.stem


def normalize_raw_column_name(col: str) -> str:
    col = str(col).replace("\ufeff", "").strip()
    col = re.sub(r"\s+", "_", col)
    return col


def canonicalize_pmu_columns(df: pd.DataFrame, path: Optional[Path] = None) -> pd.DataFrame:
    """
    Convert headers like:
      BUS2_VA_ANG, BUS2_VA_MAG, ..., BUS2_Freq, BUS2_ROCOF
    into the internal canonical names:
      VA_ang, VA_mag, ..., Frequency, ROCOF

    Keeps TIMESTAMP, DATA_PRESENT, Event as-is.
    Also validates that one CSV does not mix multiple BUS prefixes.
    """
    original_cols = list(df.columns)
    cols = [normalize_raw_column_name(c) for c in original_cols]

    detected_bus_prefixes = set()
    renamed: Dict[str, str] = {}

    canonical_map = {
        "VA_MAG": "VA_mag",
        "VA_ANG": "VA_ang",
        "VB_MAG": "VB_mag",
        "VB_ANG": "VB_ang",
        "VC_MAG": "VC_mag",
        "VC_ANG": "VC_ang",
        "IA_MAG": "IA_mag",
        "IA_ANG": "IA_ang",
        "IB_MAG": "IB_mag",
        "IB_ANG": "IB_ang",
        "IC_MAG": "IC_mag",
        "IC_ANG": "IC_ang",
        "FREQ": "Frequency",
        "FREQUENCY": "Frequency",
        "ROCOF": "ROCOF",
    }

    for raw_col, col in zip(original_cols, cols):
        upper = col.upper()

        if upper == "TIMESTAMP":
            renamed[raw_col] = "TIMESTAMP"
            continue
        if upper == "DATA_PRESENT":
            renamed[raw_col] = "DATA_PRESENT"
            continue
        if upper == "EVENT":
            renamed[raw_col] = "Event"
            continue

        m = re.match(r"^(BUS\d+)_(VA|VB|VC|IA|IB|IC)_(ANG|MAG)$", upper)
        if m:
            bus_prefix, ph, suffix = m.groups()
            detected_bus_prefixes.add(bus_prefix)
            renamed[raw_col] = f"{ph}_{suffix.lower()}"
            continue

        m = re.match(r"^(BUS\d+)_(FREQ|FREQUENCY)$", upper)
        if m:
            bus_prefix, _ = m.groups()
            detected_bus_prefixes.add(bus_prefix)
            renamed[raw_col] = "Frequency"
            continue

        m = re.match(r"^(BUS\d+)_ROCOF$", upper)
        if m:
            bus_prefix = m.group(1)
            detected_bus_prefixes.add(bus_prefix)
            renamed[raw_col] = "ROCOF"
            continue

        if upper in canonical_map:
            renamed[raw_col] = canonical_map[upper]
            continue

        renamed[raw_col] = col

    if len(detected_bus_prefixes) > 1:
        raise ValueError(
            f"CSV appears to contain multiple BUS prefixes {sorted(detected_bus_prefixes)}"
            + (f" in file {path}" if path is not None else "")
        )

    return df.rename(columns=renamed)


def required_columns_present(df: pd.DataFrame, path: Optional[Path] = None) -> None:
    missing = [c for c in MEASUREMENT_COLUMNS if c not in df.columns]
    if missing:
        where = f" in file {path}" if path is not None else ""
        raise ValueError(
            f"Missing required columns{where}: {missing}\n"
            f"Detected columns: {list(df.columns)}"
        )
    if "TIMESTAMP" not in df.columns:
        raise ValueError(
            "Missing TIMESTAMP column" + (f" in file {path}" if path is not None else "")
        )


def load_bus_csv(path: Path) -> BusData:
    df = pd.read_csv(path, sep=None, engine="python")
    df = canonicalize_pmu_columns(df, path=path)
    required_columns_present(df, path=path)

    df = df.sort_values("TIMESTAMP").reset_index(drop=True)

    numeric_cols = ["TIMESTAMP"] + MEASUREMENT_COLUMNS + [c for c in OPTIONAL_COLUMNS if c in df.columns]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "DATA_PRESENT" not in df.columns:
        df["DATA_PRESENT"] = 1
    if "Event" not in df.columns:
        df["Event"] = 0

    t = df["TIMESTAMP"].to_numpy(dtype=float)
    dt = np.diff(t)
    valid_dt = dt[np.isfinite(dt) & (dt > 0)]
    sampling_rate_hz = float(1.0 / np.median(valid_dt)) if valid_dt.size else float("nan")

    return BusData(bus_id=infer_bus_id(path), path=path, df=df, sampling_rate_hz=sampling_rate_hz)


def load_all_buses(input_dir: Path, pattern: str) -> List[BusData]:
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found in {input_dir} with pattern {pattern!r}")
    buses = [load_bus_csv(p) for p in files]
    buses = sorted(buses, key=lambda b: int(re.sub(r"\D", "", b.bus_id) or "0"))
    return buses


def contiguous_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    if mask.size == 0:
        return []
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    diff = np.diff(mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, len(mask)]
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def robust_zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    scale = mad(x)
    if not np.isfinite(scale) or scale < 1e-12:
        std = np.nanstd(x)
        if not np.isfinite(std) or std < 1e-12:
            return np.zeros_like(x, dtype=float)
        return (x - med) / std
    return 0.6744897501960817 * (x - med) / scale


def unwrap_if_angle(values: np.ndarray, column: str) -> np.ndarray:
    y = np.asarray(values, dtype=float).copy()
    if column not in ANGLE_COLUMNS:
        return y
    mask = np.isfinite(y)
    if mask.sum() < 2:
        return y
    out = y.copy()
    out[mask] = np.rad2deg(np.unwrap(np.deg2rad(y[mask])))
    return out


def rolling_trend(y: pd.Series, window_samples: int) -> pd.Series:
    window_samples = max(3, int(window_samples) | 1)
    trend = y.rolling(window=window_samples, center=True, min_periods=max(3, window_samples // 5)).median()
    trend = trend.interpolate(limit_direction="both")
    trend = trend.ffill().bfill()
    return trend


def safe_corrcoef(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return None
    x2, y2 = x[mask], y[mask]
    sx, sy = np.std(x2), np.std(y2)
    if sx < 1e-12 or sy < 1e-12:
        return None
    return float(np.corrcoef(x2, y2)[0, 1])


def autocorr_lag1(x: np.ndarray) -> Optional[float]:
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(x)
    x = x[mask]
    if x.size < 3:
        return None
    return safe_corrcoef(x[:-1], x[1:])


def distribution_fit_summary(residual: np.ndarray) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    x = np.asarray(residual, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 20:
        return out
    mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=1))
    b = float(np.mean(np.abs(x - np.median(x))))
    out["gaussian"] = {"mu": mu, "sigma": sigma}
    out["laplace"] = {"mu": float(np.median(x)), "b": b}
    if SCIPY_AVAILABLE:
        try:
            df_t, loc_t, scale_t = stats.t.fit(x)
            out["student_t"] = {"df": float(df_t), "loc": float(loc_t), "scale": float(scale_t)}
        except Exception as exc:
            out["student_t_error"] = str(exc)
        try:
            k2, p = stats.normaltest(x)
            out["normaltest"] = {"statistic": float(k2), "pvalue": float(p)}
        except Exception:
            pass
    best = "gaussian"
    if SCIPY_AVAILABLE and "student_t" in out:
        try:
            ll_gauss = float(np.sum(stats.norm.logpdf(x, loc=mu, scale=max(sigma, 1e-9))))
            ll_lap = float(np.sum(stats.laplace.logpdf(x, loc=out["laplace"]["mu"], scale=max(b, 1e-9))))
            st = out["student_t"]
            ll_t = float(np.sum(stats.t.logpdf(x, df=max(st["df"], 1e-6), loc=st["loc"], scale=max(st["scale"], 1e-9))))
            aics = {
                "gaussian": 2 * 2 - 2 * ll_gauss,
                "laplace": 2 * 2 - 2 * ll_lap,
                "student_t": 2 * 3 - 2 * ll_t,
            }
            out["aic"] = {k: float(v) for k, v in aics.items()}
            best = min(aics, key=aics.get)
        except Exception:
            pass
    out["best_family"] = best
    return out


def spectral_summary(x: np.ndarray, fs: float) -> Dict[str, Any]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 16 or not np.isfinite(fs) or fs <= 0:
        return {}
    if SCIPY_AVAILABLE and welch is not None:
        nperseg = min(4096, max(64, int(2 ** np.floor(np.log2(min(len(x), 2048))))))
        f, pxx = welch(x, fs=fs, nperseg=nperseg, detrend="constant")
    else:
        x0 = x - np.mean(x)
        n = len(x0)
        pxx = np.abs(np.fft.rfft(x0)) ** 2 / max(n, 1)
        f = np.fft.rfftfreq(n, d=1.0 / fs)
    if len(f) == 0:
        return {}
    order = np.argsort(pxx[1:])[::-1][:5] + 1 if len(f) > 1 else np.array([0])
    bands = {
        "0_to_0p1_hz": (0.0, 0.1),
        "0p1_to_1_hz": (0.1, 1.0),
        "1_to_5_hz": (1.0, 5.0),
        "5_to_nyquist_hz": (5.0, fs / 2.0),
    }
    band_energy: Dict[str, float] = {}
    for name, (lo, hi) in bands.items():
        mask = (f >= lo) & (f < hi)
        band_energy[name] = float(np.trapezoid(pxx[mask], f[mask])) if mask.any() else 0.0
    return {
        "dominant_frequencies_hz": [float(f[i]) for i in order],
        "dominant_psd": [float(pxx[i]) for i in order],
        "band_energy": band_energy,
    }


def basic_stats(values: np.ndarray, dt: float, column: str) -> Dict[str, Any]:
    raw = np.asarray(values, dtype=float)
    proc = unwrap_if_angle(raw, column)
    valid = proc[np.isfinite(proc)]
    out: Dict[str, Any] = {
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
    if SCIPY_AVAILABLE:
        try:
            out["skew"] = float(stats.skew(valid, bias=False, nan_policy="omit"))
            out["kurtosis_excess"] = float(stats.kurtosis(valid, fisher=True, bias=False, nan_policy="omit"))
        except Exception:
            pass
    else:
        centered = valid - np.mean(valid)
        std = np.std(valid, ddof=0)
        if std > 1e-12:
            out["skew"] = float(np.mean((centered / std) ** 3))
            out["kurtosis_excess"] = float(np.mean((centered / std) ** 4) - 3.0)
    deriv = np.gradient(proc, dt) if len(proc) >= 2 else np.array([], dtype=float)
    deriv = deriv[np.isfinite(deriv)]
    if deriv.size:
        out["derivative_mean"] = float(np.mean(deriv))
        out["derivative_std"] = float(np.std(deriv, ddof=1)) if deriv.size > 1 else 0.0
        out["derivative_abs_max"] = float(np.max(np.abs(deriv)))
    return out


def noise_model(values: np.ndarray, dt: float, column: str, trend_window_seconds: float) -> Dict[str, Any]:
    raw = np.asarray(values, dtype=float)
    proc = unwrap_if_angle(raw, column)
    y = pd.Series(proc)
    window_samples = max(5, int(round(trend_window_seconds / max(dt, 1e-9))))
    trend = rolling_trend(y, window_samples).to_numpy(dtype=float)
    residual = proc - trend
    valid_res = residual[np.isfinite(residual)]
    valid_trend = trend[np.isfinite(trend)]
    out: Dict[str, Any] = {}
    if valid_res.size == 0:
        return out
    res_std = float(np.std(valid_res, ddof=1)) if valid_res.size > 1 else 0.0
    trend_std = float(np.std(valid_trend, ddof=1)) if valid_trend.size > 1 else 0.0
    out.update({
        "trend_std": trend_std,
        "residual_std": res_std,
        "residual_mad": float(mad(valid_res)),
        "snr_db": float(20.0 * np.log10(max(trend_std, 1e-12) / max(res_std, 1e-12))),
        "lag1_autocorr": autocorr_lag1(valid_res),
        "outlier_rate_abs_z_gt_3": float(np.mean(np.abs(robust_zscore(valid_res)) > 3.0)),
        "outlier_rate_abs_z_gt_5": float(np.mean(np.abs(robust_zscore(valid_res)) > 5.0)),
        "distribution_fit": distribution_fit_summary(valid_res),
        "spectral": spectral_summary(valid_res, 1.0 / dt),
    })
    if SCIPY_AVAILABLE:
        try:
            out["residual_skew"] = float(stats.skew(valid_res, bias=False, nan_policy="omit"))
            out["residual_kurtosis_excess"] = float(stats.kurtosis(valid_res, fisher=True, bias=False, nan_policy="omit"))
        except Exception:
            pass
    return out


def phase_balance_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if all(col in df.columns for col in ["VA_mag", "VB_mag", "VC_mag"]):
        mags = df[["VA_mag", "VB_mag", "VC_mag"]].to_numpy(dtype=float)
        mean_mag = np.nanmean(mags, axis=1)
        dev = np.nanmax(np.abs(mags - mean_mag[:, None]), axis=1)
        ratio = dev / np.maximum(np.abs(mean_mag), 1e-9)
        out["voltage_phase_imbalance_ratio_mean"] = float(np.nanmean(ratio))
        out["voltage_phase_imbalance_ratio_p95"] = float(np.nanquantile(ratio, 0.95))
    if all(col in df.columns for col in ["IA_mag", "IB_mag", "IC_mag"]):
        mags = df[["IA_mag", "IB_mag", "IC_mag"]].to_numpy(dtype=float)
        mean_mag = np.nanmean(mags, axis=1)
        dev = np.nanmax(np.abs(mags - mean_mag[:, None]), axis=1)
        ratio = dev / np.maximum(np.abs(mean_mag), 1e-9)
        out["current_phase_imbalance_ratio_mean"] = float(np.nanmean(ratio))
        out["current_phase_imbalance_ratio_p95"] = float(np.nanquantile(ratio, 0.95))
    if all(col in df.columns for col in ["VA_ang", "VB_ang", "VC_ang"]):
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


def timestamp_integrity(df: pd.DataFrame) -> Dict[str, Any]:
    t = df["TIMESTAMP"].to_numpy(dtype=float)
    dt = np.diff(t)
    valid_dt = dt[np.isfinite(dt)]
    if valid_dt.size == 0:
        return {}
    return {
        "row_count": int(len(t)),
        "timestamp_start": float(t[0]),
        "timestamp_end": float(t[-1]),
        "dt_mean_s": float(np.mean(valid_dt)),
        "dt_median_s": float(np.median(valid_dt)),
        "dt_std_s": float(np.std(valid_dt, ddof=1)) if valid_dt.size > 1 else 0.0,
        "sampling_rate_hz": float(1.0 / np.median(valid_dt)) if np.median(valid_dt) > 0 else None,
        "duplicate_timestamp_count": int(pd.Series(t).duplicated().sum()),
        "non_monotonic_count": int(np.sum(valid_dt <= 0)),
        "timestamp_jitter_std_s": float(np.std(valid_dt - np.median(valid_dt), ddof=1)) if valid_dt.size > 1 else 0.0,
    }


def missing_profile(df: pd.DataFrame) -> Dict[str, Any]:
    measurement_nan_mask = df[MEASUREMENT_COLUMNS].isna().any(axis=1).to_numpy(dtype=bool)
    data_present_mask = None
    if "DATA_PRESENT" in df.columns:
        data_present_mask = (df["DATA_PRESENT"].fillna(0).to_numpy(dtype=float) <= 0.5)
    combined_missing = measurement_nan_mask.copy()
    if data_present_mask is not None:
        combined_missing = combined_missing | data_present_mask
    runs = contiguous_runs(combined_missing)
    t = df["TIMESTAMP"].to_numpy(dtype=float)
    gaps = []
    for s, e in runs:
        gaps.append({
            "start_idx": s,
            "end_idx": e - 1,
            "start_time": float(t[s]),
            "end_time": float(t[e - 1]),
            "duration_s": float(t[e - 1] - t[s]) if e - s > 1 else 0.0,
        })
    durations = [g["duration_s"] for g in gaps]
    return {
        "missing_frame_count": int(combined_missing.sum()),
        "missing_ratio": float(combined_missing.mean()),
        "gap_count": int(len(runs)),
        "gaps": gaps,
        "nan_frame_count": int(measurement_nan_mask.sum()),
        "data_present_zero_count": int(data_present_mask.sum()) if data_present_mask is not None else None,
        "mask_consistency_disagreement_count": int(np.sum(measurement_nan_mask != data_present_mask)) if data_present_mask is not None else None,
        "max_gap_seconds": float(max(durations)) if durations else 0.0,
        "median_gap_seconds": float(np.median(durations)) if durations else 0.0,
    }


def event_spans(df: pd.DataFrame, label_map: Dict[int, str]) -> List[Dict[str, Any]]:
    if "Event" not in df.columns:
        return []
    ev = df["Event"].fillna(0).to_numpy(dtype=int)
    t = df["TIMESTAMP"].to_numpy(dtype=float)
    spans: List[Dict[str, Any]] = []
    for event_id in sorted(set(ev.tolist())):
        if event_id == 0:
            continue
        runs = contiguous_runs(ev == event_id)
        for s, e in runs:
            spans.append({
                "event_id": int(event_id),
                "label": label_map.get(int(event_id), f"Event {event_id}"),
                "start_idx": s,
                "end_idx": e - 1,
                "start_time": float(t[s]),
                "end_time": float(t[e - 1]),
                "duration_s": float(t[e - 1] - t[s]) if e - s > 1 else 0.0,
            })
    spans.sort(key=lambda d: (d["start_time"], d["event_id"]))
    return spans


def summarize_event_spans(spans: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[int, Dict[str, Any]] = {}
    by_event: Dict[int, List[float]] = defaultdict(list)
    labels: Dict[int, str] = {}
    for item in spans:
        by_event[int(item["event_id"])].append(float(item["duration_s"]))
        labels[int(item["event_id"])] = str(item["label"])
    for event_id, durations in by_event.items():
        grouped[event_id] = {
            "label": labels[event_id],
            "count": int(len(durations)),
            "durations_s": [float(x) for x in durations],
            "mean_duration_s": float(np.mean(durations)),
            "median_duration_s": float(np.median(durations)),
            "p95_duration_s": float(np.quantile(durations, 0.95)) if len(durations) > 1 else float(durations[0]),
        }
    return {str(k): v for k, v in grouped.items()}


def detect_artifacts(values: np.ndarray, dt: float, column: str, artifact_z_threshold: float) -> Dict[str, Any]:
    raw = np.asarray(values, dtype=float)
    proc = unwrap_if_angle(raw, column)
    y = pd.Series(proc)
    trend = rolling_trend(y, max(5, int(round(1.0 / max(dt, 1e-9)))))
    residual = proc - trend.to_numpy(dtype=float)
    rz = np.abs(robust_zscore(residual))
    deriv = np.gradient(proc, dt) if len(proc) >= 2 else np.zeros_like(proc)
    dz = np.abs(robust_zscore(deriv))

    spike_mask = rz > artifact_z_threshold
    deriv_spike_mask = dz > artifact_z_threshold

    win = max(3, int(round(0.2 / max(dt, 1e-9))))
    left = pd.Series(proc).rolling(win, min_periods=win).mean().shift(1)
    right = pd.Series(proc).rolling(win, min_periods=win).mean().shift(-win)
    step_signal = np.abs((right - left).to_numpy(dtype=float))
    step_mask = np.abs(robust_zscore(step_signal)) > max(4.0, artifact_z_threshold - 1.0)

    diff_abs = np.abs(np.diff(proc, prepend=proc[0]))
    finite_diff = diff_abs[np.isfinite(diff_abs)]
    eps = max(np.nanmedian(finite_diff) * 0.1, 1e-9) if finite_diff.size else 1e-9
    nearly_constant = diff_abs <= eps
    flat_runs = contiguous_runs(nearly_constant)
    flat_runs = [(s, e) for s, e in flat_runs if (e - s) >= max(3, int(round(0.5 / max(dt, 1e-9))))]

    artifact_union = spike_mask | deriv_spike_mask | step_mask
    bursts = contiguous_runs(artifact_union)
    return {
        "spike_count": int(spike_mask.sum()),
        "derivative_spike_count": int(deriv_spike_mask.sum()),
        "step_count": int(step_mask.sum()),
        "artifact_burst_count": int(len(bursts)),
        "artifact_burst_durations_s": [float((e - s - 1) * dt) for s, e in bursts],
        "flatline_count": int(len(flat_runs)),
        "flatline_durations_s": [float((e - s - 1) * dt) for s, e in flat_runs],
        "artifact_rate_per_sample": float(np.mean(artifact_union)),
        "thresholds": {
            "robust_z_threshold": float(artifact_z_threshold),
            "flatline_eps": float(eps),
        },
    }


def event_channel_profile(df: pd.DataFrame, column: str, spans: List[Dict[str, Any]], dt: float, baseline_seconds: float, post_seconds: float) -> Dict[str, Any]:
    if not spans or column not in df.columns:
        return {}
    y = unwrap_if_angle(df[column].to_numpy(dtype=float), column)
    t = df["TIMESTAMP"].to_numpy(dtype=float)
    baseline_n = max(1, int(round(baseline_seconds / max(dt, 1e-9))))
    post_n = max(1, int(round(post_seconds / max(dt, 1e-9))))
    by_event: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for sp in spans:
        s, e = int(sp["start_idx"]), int(sp["end_idx"]) + 1
        bs = max(0, s - baseline_n)
        pe = min(len(y), e + post_n)
        baseline = y[bs:s]
        during = y[s:e]
        post = y[e:pe]
        if np.isfinite(baseline).sum() == 0 or np.isfinite(during).sum() == 0:
            continue
        base_med = float(np.nanmedian(baseline))
        peak_abs_delta = float(np.nanmax(np.abs(during - base_med)))
        peak_signed_delta = float(during[np.nanargmax(np.abs(during - base_med))] - base_med)
        settle_value = float(np.nanmedian(post)) if np.isfinite(post).sum() else None
        by_event[int(sp["event_id"])].append({
            "start_time": float(t[s]),
            "end_time": float(t[e - 1]),
            "duration_s": float(sp["duration_s"]),
            "baseline_median": base_med,
            "peak_abs_delta": peak_abs_delta,
            "peak_signed_delta": peak_signed_delta,
            "post_median": settle_value,
        })
    summarized: Dict[str, Any] = {}
    for event_id, rows in by_event.items():
        peak_abs = [r["peak_abs_delta"] for r in rows]
        peak_signed = [r["peak_signed_delta"] for r in rows]
        durations = [r["duration_s"] for r in rows]
        summarized[str(event_id)] = {
            "count": int(len(rows)),
            "peak_abs_delta_mean": float(np.mean(peak_abs)),
            "peak_abs_delta_median": float(np.median(peak_abs)),
            "peak_signed_delta_mean": float(np.mean(peak_signed)),
            "duration_mean_s": float(np.mean(durations)),
            "instances": rows,
        }
    return summarized


def diagnose_bus(bus: BusData, trend_window_seconds: float, baseline_seconds: float, post_seconds: float, artifact_z_threshold: float, label_map: Dict[int, str]) -> Dict[str, Any]:
    df = bus.df
    t = df["TIMESTAMP"].to_numpy(dtype=float)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else float("nan")
    spans = event_spans(df, label_map)
    result: Dict[str, Any] = {
        "bus_id": bus.bus_id,
        "descriptor": bus.bus_id,
        "csv_path": str(bus.path),
        "sampling_rate_hz": bus.sampling_rate_hz,
        "row_count": int(len(df)),
        "integrity": timestamp_integrity(df),
        "event_spans": spans,
        "event_summary": summarize_event_spans(spans),
        "missing_data": missing_profile(df),
        "phase_balance": phase_balance_metrics(df),
        "channels": {},
        "event_channel_profiles": {},
        "notes": {
            "harmonics_note": (
                "The CSVs contain PMU phasor features sampled at ~30 fps, not the raw 60 Hz waveform. "
                "Therefore, the spectral analysis refers to low-frequency dynamics and PMU-observation effects, "
                "not line-frequency waveform harmonics."
            )
        },
    }
    for col in MEASUREMENT_COLUMNS:
        values = df[col].to_numpy(dtype=float)
        result["channels"][col] = {
            "basic": basic_stats(values, dt, col),
            "noise": noise_model(values, dt, col, trend_window_seconds),
            "artifacts": detect_artifacts(values, dt, col, artifact_z_threshold),
            "spectral": spectral_summary(unwrap_if_angle(values, col), bus.sampling_rate_hz),
        }
        result["event_channel_profiles"][col] = event_channel_profile(
            df=df,
            column=col,
            spans=spans,
            dt=dt,
            baseline_seconds=baseline_seconds,
            post_seconds=post_seconds,
        )
    return result


def align_integrity(buses: Sequence[BusData]) -> Dict[str, Any]:
    if not buses:
        return {}
    ref = buses[0].df["TIMESTAMP"].to_numpy(dtype=float)
    out: Dict[str, Any] = {
        "reference_bus": buses[0].bus_id,
        "reference_rows": int(len(ref)),
        "per_bus_alignment": {},
    }
    for bus in buses:
        t = bus.df["TIMESTAMP"].to_numpy(dtype=float)
        min_len = min(len(ref), len(t))
        exact_match_len = int(np.sum(np.isclose(ref[:min_len], t[:min_len], atol=1e-9, rtol=0.0)))
        common = len(np.intersect1d(ref, t))
        out["per_bus_alignment"][bus.bus_id] = {
            "row_count": int(len(t)),
            "common_timestamp_count": int(common),
            "exact_prefix_match_count": exact_match_len,
            "prefix_match_ratio": float(exact_match_len / max(min_len, 1)),
            "full_length_equal": bool(len(ref) == len(t) and exact_match_len == len(ref)),
        }
    return out


def make_merged_frame(buses: Sequence[BusData]) -> pd.DataFrame:
    merged: Optional[pd.DataFrame] = None
    for bus in buses:
        cols = ["TIMESTAMP"] + [c for c in MEASUREMENT_COLUMNS if c in bus.df.columns]
        bus_df = bus.df[cols].copy()
        bus_df = bus_df.rename(columns={c: f"{bus.bus_id}__{c}" for c in cols if c != "TIMESTAMP"})
        merged = bus_df if merged is None else merged.merge(bus_df, on="TIMESTAMP", how="inner")
    assert merged is not None
    return merged


def cross_bus_rankings(buses: Sequence[BusData], label_map: Dict[int, str]) -> Dict[str, Any]:
    if not buses or "Event" not in buses[0].df.columns:
        return {}
    ref = buses[0].df
    spans = event_spans(ref, label_map)
    outputs: Dict[str, List[Dict[str, Any]]] = {c: [] for c in MEASUREMENT_COLUMNS}
    for sp in spans:
        s, e = int(sp["start_idx"]), int(sp["end_idx"]) + 1
        for col in MEASUREMENT_COLUMNS:
            ranked = []
            for bus in buses:
                series = unwrap_if_angle(bus.df[col].to_numpy(dtype=float), col)
                baseline = series[max(0, s - 60):s]
                during = series[s:e]
                if np.isfinite(baseline).sum() < 3 or np.isfinite(during).sum() < 1:
                    score = None
                else:
                    mu = float(np.nanmedian(baseline))
                    sigma = float(mad(baseline))
                    sigma = sigma if np.isfinite(sigma) and sigma > 1e-12 else float(np.nanstd(baseline))
                    sigma = sigma if np.isfinite(sigma) and sigma > 1e-12 else 1.0
                    score = float(np.nanmax(np.abs((during - mu) / sigma)))
                ranked.append({"bus": bus.bus_id, "score": score})
            ranked = [r for r in ranked if r["score"] is not None]
            ranked.sort(key=lambda d: d["score"], reverse=True)
            outputs[col].append({
                "event_id": int(sp["event_id"]),
                "label": sp["label"],
                "start_time": float(sp["start_time"]),
                "end_time": float(sp["end_time"]),
                "value_suffix": col,
                "ranked_buses_by_peak_abs_zscore": ranked,
            })
    return outputs


def cross_bus_correlations(buses: Sequence[BusData]) -> Dict[str, Any]:
    merged = make_merged_frame(buses)
    out: Dict[str, Any] = {}
    for col in ["VA_mag", "IA_mag", "Frequency", "ROCOF"]:
        cols = [f"{bus.bus_id}__{col}" for bus in buses if f"{bus.bus_id}__{col}" in merged.columns]
        frame = merged[cols]
        out[col] = frame.corr().replace({np.nan: None}).to_dict()
    return out


def aggregate_event_library(bus_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    durations_by_event: Dict[int, List[float]] = defaultdict(list)
    labels_by_event: Dict[int, str] = {}
    buses_by_event: Dict[int, set] = defaultdict(set)
    for bus in bus_results:
        for sp in bus.get("event_spans", []):
            eid = int(sp["event_id"])
            durations_by_event[eid].append(float(sp["duration_s"]))
            labels_by_event[eid] = str(sp["label"])
            buses_by_event[eid].add(bus["bus_id"])
    out: Dict[str, Any] = {}
    for eid, durations in durations_by_event.items():
        out[str(eid)] = {
            "label": labels_by_event[eid],
            "bus_count_with_event": int(len(buses_by_event[eid])),
            "bus_ids": sorted(list(buses_by_event[eid])),
            "count": int(len(durations)),
            "duration_mean_s": float(np.mean(durations)),
            "duration_median_s": float(np.median(durations)),
            "duration_std_s": float(np.std(durations, ddof=1)) if len(durations) > 1 else 0.0,
            "duration_p95_s": float(np.quantile(durations, 0.95)) if len(durations) > 1 else float(durations[0]),
            "durations_s": [float(x) for x in durations],
        }
    return out


def simulation_copy_recipe(bus_results: Sequence[Dict[str, Any]], event_library: Dict[str, Any], rankings: Dict[str, Any]) -> Dict[str, Any]:
    recipe: Dict[str, Any] = {
        "global": {
            "intended_use": (
                "Use these parameters as the post-PMU observation and corruption layer after ANDES physics. "
                "Do not inject them into the differential equations; apply them to the exported PMU-like signals."
            ),
            "recommended_pipeline_order": [
                "ANDES clean physics",
                "PMU observation/filter layer",
                "bus/channel baseline offsets",
                "colored stochastic noise",
                "heavy-tail residual shaping",
                "artifact / bad-data bursts",
                "missing-data bursts",
                "event-duration relabeling at PMU-visible level",
            ],
            "event_library": event_library,
        },
        "per_bus": {},
        "rankings": rankings,
    }
    for bus in bus_results:
        bus_block: Dict[str, Any] = {
            "bus_id": bus["bus_id"],
            "sampling_rate_hz": bus["sampling_rate_hz"],
            "phase_balance": bus.get("phase_balance", {}),
            "missing_model": bus.get("missing_data", {}),
            "channels": {},
        }
        for col, info in bus["channels"].items():
            basic = info.get("basic", {})
            noise = info.get("noise", {})
            art = info.get("artifacts", {})
            dist = noise.get("distribution_fit", {})
            band_energy = noise.get("spectral", {}).get("band_energy", {})
            event_profiles = bus.get("event_channel_profiles", {}).get(col, {})
            bus_block["channels"][col] = {
                "baseline": {
                    "mean": basic.get("mean"),
                    "median": basic.get("median"),
                    "std": basic.get("std"),
                    "q05": basic.get("q05"),
                    "q95": basic.get("q95"),
                },
                "noise_model": {
                    "trend_std": noise.get("trend_std"),
                    "residual_std": noise.get("residual_std"),
                    "residual_mad": noise.get("residual_mad"),
                    "lag1_autocorr": noise.get("lag1_autocorr"),
                    "snr_db": noise.get("snr_db"),
                    "best_distribution_family": dist.get("best_family"),
                    "distribution_fit": dist,
                    "low_frequency_band_energy": band_energy.get("0_to_0p1_hz"),
                    "mid_frequency_band_energy": band_energy.get("0p1_to_1_hz"),
                    "electromechanical_band_energy": band_energy.get("1_to_5_hz"),
                },
                "artifact_model": {
                    "artifact_rate_per_sample": art.get("artifact_rate_per_sample"),
                    "spike_count": art.get("spike_count"),
                    "derivative_spike_count": art.get("derivative_spike_count"),
                    "step_count": art.get("step_count"),
                    "flatline_count": art.get("flatline_count"),
                    "artifact_burst_durations_s": art.get("artifact_burst_durations_s"),
                    "flatline_durations_s": art.get("flatline_durations_s"),
                },
                "event_response_model": event_profiles,
            }
        recipe["per_bus"][bus["bus_id"]] = bus_block
    return recipe


def plot_bus_overview(bus: BusData, out_dir: Path) -> None:
    df = bus.df
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    axes[0].plot(df["TIMESTAMP"], df["VA_mag"], label="VA_mag")
    axes[0].plot(df["TIMESTAMP"], df["VB_mag"], label="VB_mag", alpha=0.7)
    axes[0].plot(df["TIMESTAMP"], df["VC_mag"], label="VC_mag", alpha=0.7)
    axes[0].set_ylabel("Voltage mag")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["TIMESTAMP"], df["IA_mag"], label="IA_mag")
    axes[1].plot(df["TIMESTAMP"], df["IB_mag"], label="IB_mag", alpha=0.7)
    axes[1].plot(df["TIMESTAMP"], df["IC_mag"], label="IC_mag", alpha=0.7)
    axes[1].set_ylabel("Current mag")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(df["TIMESTAMP"], df["Frequency"], label="Frequency")
    axes[2].plot(df["TIMESTAMP"], df["ROCOF"], label="ROCOF", alpha=0.8)
    axes[2].set_ylabel("Freq / ROCOF")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    if "DATA_PRESENT" in df.columns:
        axes[3].step(df["TIMESTAMP"], df["DATA_PRESENT"], where="post", label="DATA_PRESENT")
    if "Event" in df.columns:
        axes[3].step(df["TIMESTAMP"], df["Event"], where="post", label="Event", alpha=0.8)
    axes[3].set_ylabel("Meta")
    axes[3].set_xlabel("Time [s]")
    axes[3].legend(loc="upper right")
    axes[3].grid(True, alpha=0.3)

    fig.suptitle(f"{bus.bus_id} overview")
    fig.tight_layout()
    fig.savefig(out_dir / f"{bus.bus_id}_overview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_bus_distribution_panels(bus_result: Dict[str, Any], bus: BusData, out_dir: Path) -> None:
    df = bus.df
    fig, axes = plt.subplots(4, 2, figsize=(14, 14))
    chosen = ["VA_mag", "IA_mag", "Frequency", "ROCOF"]
    for r, col in enumerate(chosen):
        values = df[col].to_numpy(dtype=float)
        axes[r, 0].hist(values[np.isfinite(values)], bins=100)
        axes[r, 0].set_title(f"{bus.bus_id} | {col} histogram")
        axes[r, 0].grid(True, alpha=0.25)
        spec = bus_result["channels"][col].get("spectral", {})
        freqs = spec.get("dominant_frequencies_hz", [])
        amps = spec.get("dominant_psd", [])
        if freqs and amps:
            axes[r, 1].stem(freqs, amps, basefmt=" ")
        axes[r, 1].set_title(f"{bus.bus_id} | {col} dominant spectral lines")
        axes[r, 1].set_xlabel("Frequency [Hz]")
        axes[r, 1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / f"{bus.bus_id}_distribution_panels.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_event_zooms(buses: Sequence[BusData], label_map: Dict[int, str], out_dir: Path) -> None:
    if not buses or "Event" not in buses[0].df.columns:
        return
    ref = buses[0].df
    spans = event_spans(ref, label_map)
    for sp in spans:
        fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
        t0 = sp["start_time"] - 1.0
        t1 = sp["end_time"] + 1.0
        for bus in buses:
            df = bus.df
            mask = (df["TIMESTAMP"] >= t0) & (df["TIMESTAMP"] <= t1)
            axes[0].plot(df.loc[mask, "TIMESTAMP"], df.loc[mask, "Frequency"], label=bus.bus_id)
            axes[1].plot(df.loc[mask, "TIMESTAMP"], df.loc[mask, "VA_mag"], label=bus.bus_id)
        for ax in axes:
            ax.axvline(sp["start_time"], linestyle="--", linewidth=1.2)
            ax.axvline(sp["end_time"], linestyle="--", linewidth=1.2)
            ax.axvspan(sp["start_time"], sp["end_time"], alpha=0.15)
            ax.grid(True, alpha=0.3)
        axes[0].set_title(f"Frequency overlay | event {sp['event_id']} ({sp['label']})")
        axes[0].set_ylabel("Frequency")
        axes[1].set_title(f"VA_mag overlay | event {sp['event_id']} ({sp['label']})")
        axes[1].set_ylabel("VA_mag")
        axes[1].set_xlabel("Time [s]")
        axes[0].legend(ncol=4, loc="upper right")
        fig.tight_layout()
        safe_label = sp["label"].replace(" ", "_").replace("/", "_")
        fig.savefig(out_dir / f"event_{sp['event_id']}_{safe_label}_overlay.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_py(payload), f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    out_root = args.output_dir
    ensure_dir(out_root)
    json_dir = out_root / "json"
    plots_dir = out_root / "plots"
    per_bus_json_dir = out_root / "per_bus_json"
    ensure_dir(json_dir)
    ensure_dir(plots_dir)
    if args.save_per_bus_json:
        ensure_dir(per_bus_json_dir)

    buses = load_all_buses(args.input_dir, args.pattern)
    label_map = DEFAULT_EVENT_LABELS.copy()

    dataset_integrity = {
        "input_dir": str(args.input_dir),
        "file_count": int(len(buses)),
        "bus_ids": [bus.bus_id for bus in buses],
        "alignment": align_integrity(buses),
        "expected_columns": MEASUREMENT_COLUMNS + OPTIONAL_COLUMNS,
    }

    bus_results: List[Dict[str, Any]] = []
    for bus in buses:
        result = diagnose_bus(
            bus=bus,
            trend_window_seconds=args.trend_window_seconds,
            baseline_seconds=args.event_baseline_seconds,
            post_seconds=args.event_post_seconds,
            artifact_z_threshold=args.artifact_z_threshold,
            label_map=label_map,
        )
        bus_results.append(result)
        plot_bus_overview(bus, plots_dir)
        plot_bus_distribution_panels(result, bus, plots_dir)
        if args.save_per_bus_json:
            save_json(per_bus_json_dir / f"{bus.bus_id}_summary.json", result)

    rankings = cross_bus_rankings(buses, label_map)
    event_library = aggregate_event_library(bus_results)
    cross_bus = {
        "rankings": rankings,
        "correlations": cross_bus_correlations(buses),
    }
    recipe = simulation_copy_recipe(bus_results, event_library, rankings)

    super_model = {
        "metadata": {
            "tool": "pmu_realism_diagnoser",
            "version": "1.1.0",
            "input_dir": str(args.input_dir),
            "output_dir": str(args.output_dir),
            "bus_count": int(len(buses)),
            "notes": [
                "This JSON is meant to parameterize a post-ANDES PMU realism layer.",
                "Event durations here are PMU-visible durations, not necessarily raw physical durations.",
                "Noise and artifact parameters are empirical and bus/channel specific.",
                "This version supports SGSMA CSV headers like BUS2_VA_ANG, BUS2_Freq, BUS2_ROCOF.",
            ],
        },
        "dataset_integrity": dataset_integrity,
        "event_library": event_library,
        "cross_bus": cross_bus,
        "buses": {bus_result["bus_id"]: bus_result for bus_result in bus_results},
        "simulation_copy_recipe": recipe,
    }

    plot_event_zooms(buses, label_map, plots_dir)

    save_json(json_dir / "dataset_integrity.json", dataset_integrity)
    save_json(json_dir / "event_library.json", event_library)
    save_json(json_dir / "cross_bus_profile.json", cross_bus)
    save_json(json_dir / "simulation_copy_recipe.json", recipe)
    save_json(json_dir / "pmu_realism_super_model.json", super_model)

    summary_lines = [
        f"Processed {len(buses)} PMU CSV files.",
        f"Buses: {', '.join(bus.bus_id for bus in buses)}",
        f"Main output: {json_dir / 'pmu_realism_super_model.json'}",
    ]
    (out_root / "README_outputs.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("Done.")
    for line in summary_lines:
        print(line)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        main()