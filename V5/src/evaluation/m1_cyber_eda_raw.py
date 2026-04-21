from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import ks_2samp, wasserstein_distance
except Exception:  # pragma: no cover - fallback for minimal envs
    ks_2samp = None  # type: ignore[assignment]
    wasserstein_distance = None  # type: ignore[assignment]

try:
    from sklearn.feature_selection import mutual_info_classif
except Exception:  # pragma: no cover - fallback for minimal envs
    mutual_info_classif = None  # type: ignore[assignment]


EVENT_NAMES: dict[int, str] = {
    0: "normal",
    1: "fault",
    2: "line_outage",
    3: "generation_change_outage",
    4: "load_change",
    5: "missing_data",
    6: "missing_data_plus_physical",
    7: "bad_data",
    8: "unknown",
}
TARGET_EVENTS = (0, 5, 7)
COMPARISONS = ((5, 0), (7, 0), (5, 7))

BUS_ORDER = ["BUS39", "BUS29", "BUS10", "BUS22", "BUS19", "BUS2", "BUS5", "BUS6"]
MEASUREMENT_CHANNELS = [
    "VA_ANG",
    "VA_MAG",
    "VB_ANG",
    "VB_MAG",
    "VC_ANG",
    "VC_MAG",
    "IA_ANG",
    "IA_MAG",
    "IB_ANG",
    "IB_MAG",
    "IC_ANG",
    "IC_MAG",
    "FREQ",
    "ROCOF",
]
ANGLE_CHANNELS = ["VA_ANG", "VB_ANG", "VC_ANG", "IA_ANG", "IB_ANG", "IC_ANG"]
FREQ_CHANNELS = ["FREQ", "ROCOF"]
EVENT_COLORS = {
    0: "#4d4d4d",
    5: "#d95f02",
    7: "#1b9e77",
}


@dataclass(slots=True)
class ChunkBusRecord:
    chunk_dir: Path
    chunk_name: str
    chunk_index: int
    chunk_event_id: int
    chunk_event_label: str
    bus: str
    frame: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M1 RAW cyber EDA for Event 5/7 vs Event 0")
    parser.add_argument(
        "--chunks-root",
        type=Path,
        default=Path("output/M0_RAW0001_NEWARCH/chunks"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output/M1_CYBER_EDA_RAW0001"),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
    )
    return parser.parse_args()


def _safe_float(value: float | int | np.number | None) -> float:
    if value is None:
        return float("nan")
    try:
        out = float(value)
    except Exception:
        return float("nan")
    if not np.isfinite(out):
        return float("nan")
    return out


def _safe_mean(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def _safe_std(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.std(arr))


def _safe_median(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.median(arr))


def _safe_iqr(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.quantile(arr, 0.75) - np.quantile(arr, 0.25))


def _robust_sigma(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    return float(1.4826 * mad)


def _contiguous_true_runs(mask: np.ndarray) -> list[int]:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    runs: list[int] = []
    count = 0
    for item in mask:
        if item:
            count += 1
        elif count > 0:
            runs.append(count)
            count = 0
    if count > 0:
        runs.append(count)
    return runs


def _contiguous_true_run_positions(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    out: list[tuple[int, int]] = []
    start: int | None = None
    for idx, item in enumerate(mask):
        if item and start is None:
            start = idx
        elif not item and start is not None:
            out.append((start, idx - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


def _wrapped_deg(angle_deg: np.ndarray) -> np.ndarray:
    vals = np.asarray(angle_deg, dtype=float)
    return ((vals + 180.0) % 360.0) - 180.0


def _wrapped_diff_deg(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return np.zeros((0,), dtype=float)
    raw = np.diff(vals)
    return _wrapped_deg(raw)


def _acf(values: np.ndarray, lag: int) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= lag or lag <= 0:
        return float("nan")
    x = arr - np.mean(arr)
    a = x[:-lag]
    b = x[lag:]
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(a * b) / denom)


def _spectral_summary(values: np.ndarray, dt: float) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 8 or not np.isfinite(dt) or dt <= 0.0:
        return {
            "psd_lowfreq_ratio": float("nan"),
            "psd_highfreq_ratio": float("nan"),
            "dominant_frequency_hz": float("nan"),
        }
    centered = arr - np.mean(arr)
    spec = np.fft.rfft(centered)
    power = np.abs(spec) ** 2
    freqs = np.fft.rfftfreq(arr.size, d=dt)
    total = float(np.sum(power))
    if total <= 1e-12:
        return {
            "psd_lowfreq_ratio": float("nan"),
            "psd_highfreq_ratio": float("nan"),
            "dominant_frequency_hz": float("nan"),
        }
    nyquist = float(0.5 / dt)
    low_mask = freqs <= min(0.5, nyquist)
    high_mask = freqs >= max(0.5 * nyquist, 0.0)
    low_ratio = float(np.sum(power[low_mask]) / total) if np.any(low_mask) else float("nan")
    high_ratio = float(np.sum(power[high_mask]) / total) if np.any(high_mask) else float("nan")
    dominant_idx = int(np.argmax(power[1:]) + 1) if power.size > 1 else 0
    return {
        "psd_lowfreq_ratio": low_ratio,
        "psd_highfreq_ratio": high_ratio,
        "dominant_frequency_hz": float(freqs[dominant_idx]),
    }


def _parse_chunk_name(chunk_name: str) -> tuple[int, int, str]:
    match = re.match(r"^chunk(\d+)_event_(\d+)_(.+)$", chunk_name, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Could not parse chunk folder name: {chunk_name}")
    chunk_idx = int(match.group(1))
    event_id = int(match.group(2))
    label = str(match.group(3))
    return chunk_idx, event_id, label


def _canonicalize_bus_frame(df: pd.DataFrame, bus: str) -> pd.DataFrame:
    out = df.copy()
    bus_prefix = bus.upper() + "_"
    rename: dict[str, str] = {}
    for col in out.columns:
        upper = str(col).upper()
        if upper.startswith(bus_prefix):
            rename[col] = upper[len(bus_prefix) :]
        else:
            rename[col] = upper
    out = out.rename(columns=rename)
    required = ["TIMESTAMP", *MEASUREMENT_CHANNELS, "DATA_PRESENT", "EVENT"]
    for col in required:
        if col not in out.columns:
            out[col] = np.nan
    out = out[required].copy()
    out["TIMESTAMP"] = pd.to_numeric(out["TIMESTAMP"], errors="coerce")
    out = out.dropna(subset=["TIMESTAMP"]).sort_values("TIMESTAMP").reset_index(drop=True)
    for col in MEASUREMENT_CHANNELS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["DATA_PRESENT"] = pd.to_numeric(out["DATA_PRESENT"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    out["EVENT"] = pd.to_numeric(out["EVENT"], errors="coerce").fillna(0).astype(int)
    return out


def load_chunk_bus_records(chunks_root: Path) -> list[ChunkBusRecord]:
    root = chunks_root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Chunks root does not exist: {root}")
    chunk_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    records: list[ChunkBusRecord] = []
    for chunk_dir in chunk_dirs:
        chunk_idx, chunk_event, chunk_label = _parse_chunk_name(chunk_dir.name)
        for csv_path in sorted(chunk_dir.glob("Bus*.csv")):
            bus = csv_path.stem.upper()
            raw = pd.read_csv(csv_path)
            frame = _canonicalize_bus_frame(raw, bus=bus)
            records.append(
                ChunkBusRecord(
                    chunk_dir=chunk_dir,
                    chunk_name=chunk_dir.name,
                    chunk_index=chunk_idx,
                    chunk_event_id=chunk_event,
                    chunk_event_label=chunk_label,
                    bus=bus,
                    frame=frame,
                )
            )
    if not records:
        raise RuntimeError(f"No Bus*.csv files found under {root}")
    return records


def records_to_dataframe(records: list[ChunkBusRecord]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for rec in records:
        frame = rec.frame.copy()
        frame["BUS"] = rec.bus
        frame["CHUNK_NAME"] = rec.chunk_name
        frame["CHUNK_INDEX"] = rec.chunk_index
        frame["CHUNK_EVENT_ID"] = rec.chunk_event_id
        frame["CHUNK_EVENT_LABEL"] = rec.chunk_event_label
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True)
    merged["EVENT_NAME"] = merged["EVENT"].map(EVENT_NAMES).fillna("other")
    return merged


def build_chunk_inventory(records: list[ChunkBusRecord]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rec in records:
        frame = rec.frame
        ts = frame["TIMESTAMP"].to_numpy(dtype=float)
        unique_ts = int(np.unique(ts).size)
        rows.append(
            {
                "chunk_name": rec.chunk_name,
                "chunk_index": rec.chunk_index,
                "chunk_event_id": rec.chunk_event_id,
                "chunk_event_name": EVENT_NAMES.get(rec.chunk_event_id, "other"),
                "chunk_event_label": rec.chunk_event_label,
                "bus": rec.bus,
                "rows": int(len(frame)),
                "timestamp_start": _safe_float(ts.min() if ts.size else float("nan")),
                "timestamp_end": _safe_float(ts.max() if ts.size else float("nan")),
                "duration_s": _safe_float((ts.max() - ts.min()) if ts.size > 1 else 0.0),
                "unique_timestamps": unique_ts,
                "duplicate_timestamps": int(len(frame) - unique_ts),
                "rows_event0": int((frame["EVENT"] == 0).sum()),
                "rows_event5": int((frame["EVENT"] == 5).sum()),
                "rows_event7": int((frame["EVENT"] == 7).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["chunk_index", "bus"]).reset_index(drop=True)


def _estimate_expected_dt(frame: pd.DataFrame) -> float:
    dt = np.diff(frame["TIMESTAMP"].to_numpy(dtype=float))
    dt = dt[np.isfinite(dt) & (dt > 0.0)]
    if dt.size == 0:
        return 0.033
    return float(np.median(dt))


def compute_timestamp_gap_stats(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gap_rows: list[dict[str, Any]] = []
    dt_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []

    for (chunk_name, bus), group in data.groupby(["CHUNK_NAME", "BUS"], sort=True):
        g = group.sort_values("TIMESTAMP").reset_index(drop=True)
        ts = g["TIMESTAMP"].to_numpy(dtype=float)
        dt = np.diff(ts, prepend=np.nan)
        expected_dt = _estimate_expected_dt(g)
        dp = g["DATA_PRESENT"].to_numpy(dtype=float)
        dp_bin = (dp >= 0.5).astype(int)
        prev_dp = np.roll(dp_bin, 1)
        prev_dp[0] = dp_bin[0]
        transitions = prev_dp * 10 + dp_bin

        for event_id in TARGET_EVENTS:
            event_mask = g["EVENT"].to_numpy(dtype=int) == int(event_id)
            if int(np.sum(event_mask)) == 0:
                continue
            event_dt = dt[event_mask]
            event_dt = event_dt[np.isfinite(event_dt)]
            if event_dt.size == 0:
                continue
            missing_frames = np.maximum(np.round(event_dt / max(expected_dt, 1e-6)) - 1.0, 0.0)
            duplicate_count = int(np.sum(event_dt <= 1e-12))
            irregular = np.abs(event_dt - expected_dt) > (0.15 * max(expected_dt, 1e-6))
            event_transition_mask = event_mask & (np.arange(len(g)) > 0)
            tr_values = transitions[event_transition_mask]
            tr_10 = int(np.sum(tr_values == 10))
            tr_1 = int(np.sum(tr_values == 1))
            transition_rows.append(
                {
                    "chunk_name": chunk_name,
                    "bus": bus,
                    "event": int(event_id),
                    "event_name": EVENT_NAMES.get(int(event_id), "other"),
                    "transition_1_to_0": tr_10,
                    "transition_0_to_1": tr_1,
                    "transition_total": tr_10 + tr_1,
                }
            )
            gap_rows.append(
                {
                    "chunk_name": chunk_name,
                    "bus": bus,
                    "event": int(event_id),
                    "event_name": EVENT_NAMES.get(int(event_id), "other"),
                    "n_samples": int(event_dt.size),
                    "expected_dt_s": expected_dt,
                    "dt_mean_s": _safe_mean(event_dt),
                    "dt_std_s": _safe_std(event_dt),
                    "dt_median_s": _safe_median(event_dt),
                    "dt_iqr_s": _safe_iqr(event_dt),
                    "dt_p95_s": _safe_float(np.quantile(event_dt, 0.95)),
                    "duplicate_timestamps": duplicate_count,
                    "irregular_spacing_rate": _safe_float(np.mean(irregular)),
                    "missing_frame_count_est": int(np.sum(missing_frames)),
                    "missing_frame_rate_est": _safe_float(np.sum(missing_frames) / max(event_dt.size, 1)),
                    "jitter_std_s": _safe_std(event_dt - expected_dt),
                    "data_present_transition_1_to_0": tr_10,
                    "data_present_transition_0_to_1": tr_1,
                }
            )
            sample_dt = event_dt[: min(4000, event_dt.size)]
            for val in sample_dt:
                dt_rows.append(
                    {
                        "chunk_name": chunk_name,
                        "bus": bus,
                        "event": int(event_id),
                        "event_name": EVENT_NAMES.get(int(event_id), "other"),
                        "dt_s": float(val),
                    }
                )

    gap_df = pd.DataFrame(gap_rows)
    dt_df = pd.DataFrame(dt_rows)
    tr_df = pd.DataFrame(transition_rows)
    return gap_df, dt_df, tr_df


def _channel_family(channel: str) -> str:
    name = str(channel).upper()
    if name in {"FREQ"}:
        return "frequency"
    if name in {"ROCOF"}:
        return "rocof"
    if name.startswith("V"):
        return "voltage"
    if name.startswith("I"):
        return "current"
    return "other"


def compute_missingness_stats(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    stats_rows: list[dict[str, Any]] = []
    burst_rows: list[dict[str, Any]] = []

    for (chunk_name, bus), group in data.groupby(["CHUNK_NAME", "BUS"], sort=True):
        g = group.sort_values("TIMESTAMP").reset_index(drop=True)
        for event_id in TARGET_EVENTS:
            e = g.loc[g["EVENT"] == int(event_id)].copy()
            if e.empty:
                continue
            row_nan_fraction = e[MEASUREMENT_CHANNELS].isna().mean(axis=1).to_numpy(dtype=float)
            full_dropout_mask = row_nan_fraction >= 0.95
            partial_dropout_mask = (row_nan_fraction > 0.0) & (row_nan_fraction < 0.95)
            dp_missing_mask = e["DATA_PRESENT"].to_numpy(dtype=float) < 0.5
            dp_runs = _contiguous_true_run_positions(dp_missing_mask)
            for start, end in dp_runs:
                burst_rows.append(
                    {
                        "chunk_name": chunk_name,
                        "bus": bus,
                        "event": int(event_id),
                        "event_name": EVENT_NAMES.get(int(event_id), "other"),
                        "burst_type": "data_present_zero",
                        "channel": "__ROW__",
                        "start_idx": int(start),
                        "end_idx": int(end),
                        "burst_length": int(end - start + 1),
                    }
                )

            stats_rows.append(
                {
                    "chunk_name": chunk_name,
                    "bus": bus,
                    "event": int(event_id),
                    "event_name": EVENT_NAMES.get(int(event_id), "other"),
                    "channel": "__ROW__",
                    "channel_family": "row",
                    "nan_fraction": _safe_mean(row_nan_fraction),
                    "missing_burst_count": int(len(dp_runs)),
                    "missing_burst_mean_len": _safe_mean(np.array([end - start + 1 for start, end in dp_runs], dtype=float)),
                    "missing_burst_max_len": _safe_float(
                        np.max(np.array([end - start + 1 for start, end in dp_runs], dtype=float)) if dp_runs else np.nan
                    ),
                    "partial_dropout_rate": _safe_float(np.mean(partial_dropout_mask)),
                    "full_dropout_rate": _safe_float(np.mean(full_dropout_mask)),
                }
            )

            for channel in MEASUREMENT_CHANNELS:
                vals = e[channel].to_numpy(dtype=float)
                nan_mask = ~np.isfinite(vals)
                nan_runs = _contiguous_true_run_positions(nan_mask)
                for start, end in nan_runs:
                    burst_rows.append(
                        {
                            "chunk_name": chunk_name,
                            "bus": bus,
                            "event": int(event_id),
                            "event_name": EVENT_NAMES.get(int(event_id), "other"),
                            "burst_type": "channel_nan",
                            "channel": channel,
                            "start_idx": int(start),
                            "end_idx": int(end),
                            "burst_length": int(end - start + 1),
                        }
                    )
                stats_rows.append(
                    {
                        "chunk_name": chunk_name,
                        "bus": bus,
                        "event": int(event_id),
                        "event_name": EVENT_NAMES.get(int(event_id), "other"),
                        "channel": channel,
                        "channel_family": _channel_family(channel),
                        "nan_fraction": _safe_float(np.mean(nan_mask)),
                        "missing_burst_count": int(len(nan_runs)),
                        "missing_burst_mean_len": _safe_mean(np.array([end - start + 1 for start, end in nan_runs], dtype=float)),
                        "missing_burst_max_len": _safe_float(
                            np.max(np.array([end - start + 1 for start, end in nan_runs], dtype=float)) if nan_runs else np.nan
                        ),
                        "partial_dropout_rate": float("nan"),
                        "full_dropout_rate": float("nan"),
                    }
                )

    return pd.DataFrame(stats_rows), pd.DataFrame(burst_rows)


def _series_stats(values: np.ndarray, timestamps: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    ts = np.asarray(timestamps, dtype=float)
    finite = np.isfinite(arr) & np.isfinite(ts)
    arr = arr[finite]
    ts = ts[finite]
    if arr.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "iqr": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "robust_sigma": float("nan"),
            "outlier_rate": float("nan"),
            "increment_std": float("nan"),
            "increment_iqr": float("nan"),
            "derivative_std": float("nan"),
            "second_derivative_std": float("nan"),
            "jump_rate": float("nan"),
            "stuck_value_rate": float("nan"),
            "max_repeat_run": float("nan"),
            "clipping_low_rate": float("nan"),
            "clipping_high_rate": float("nan"),
            "drift_slope_per_s": float("nan"),
            "acf_lag1": float("nan"),
            "acf_lag5": float("nan"),
            "psd_lowfreq_ratio": float("nan"),
            "psd_highfreq_ratio": float("nan"),
            "dominant_frequency_hz": float("nan"),
        }
    med = np.median(arr)
    iqr = _safe_iqr(arr)
    sigma = _robust_sigma(arr)
    z = np.abs((arr - med) / max(sigma, 1e-9))
    outlier_rate = float(np.mean(z > 3.5))

    if arr.size > 1:
        dx = np.diff(arr)
        dts = np.diff(ts)
        valid_dx = np.isfinite(dx) & np.isfinite(dts) & (np.abs(dts) > 1e-9)
        dx = dx[np.isfinite(dx)]
        deriv = (np.diff(arr)[valid_dx] / dts[valid_dx]) if np.any(valid_dx) else np.zeros((0,), dtype=float)
        ddx = np.diff(np.diff(arr)) if arr.size > 2 else np.zeros((0,), dtype=float)
    else:
        dx = np.zeros((0,), dtype=float)
        deriv = np.zeros((0,), dtype=float)
        ddx = np.zeros((0,), dtype=float)

    abs_dx = np.abs(dx[np.isfinite(dx)])
    jump_threshold = np.quantile(abs_dx, 0.95) if abs_dx.size else np.nan
    jump_rate = float(np.mean(abs_dx >= jump_threshold)) if abs_dx.size and np.isfinite(jump_threshold) else float("nan")

    tol = max(1e-9, 0.001 * max(float(iqr), 1e-6))
    stuck_mask = np.abs(np.diff(arr)) <= tol if arr.size > 1 else np.zeros((0,), dtype=bool)
    repeat_runs = _contiguous_true_runs(stuck_mask)
    q01 = np.quantile(arr, 0.01)
    q99 = np.quantile(arr, 0.99)
    low_clip_rate = float(np.mean(arr <= q01))
    high_clip_rate = float(np.mean(arr >= q99))

    if arr.size >= 3 and np.unique(ts).size >= 3:
        try:
            slope = float(np.polyfit(ts, arr, 1)[0])
        except Exception:
            slope = float("nan")
    else:
        slope = float("nan")

    dt = np.diff(ts)
    dt = dt[np.isfinite(dt) & (dt > 0.0)]
    dt_median = float(np.median(dt)) if dt.size else 0.033
    spec = _spectral_summary(arr, dt=dt_median)

    return {
        "mean": _safe_mean(arr),
        "std": _safe_std(arr),
        "median": med,
        "iqr": iqr,
        "min": _safe_float(np.min(arr)),
        "max": _safe_float(np.max(arr)),
        "robust_sigma": sigma,
        "outlier_rate": outlier_rate,
        "increment_std": _safe_std(dx),
        "increment_iqr": _safe_iqr(dx),
        "derivative_std": _safe_std(deriv),
        "second_derivative_std": _safe_std(ddx),
        "jump_rate": jump_rate,
        "stuck_value_rate": _safe_float(np.mean(stuck_mask)) if stuck_mask.size else float("nan"),
        "max_repeat_run": _safe_float(np.max(repeat_runs) if repeat_runs else np.nan),
        "clipping_low_rate": low_clip_rate,
        "clipping_high_rate": high_clip_rate,
        "drift_slope_per_s": slope,
        "acf_lag1": _acf(arr, lag=1),
        "acf_lag5": _acf(arr, lag=5),
        "psd_lowfreq_ratio": spec["psd_lowfreq_ratio"],
        "psd_highfreq_ratio": spec["psd_highfreq_ratio"],
        "dominant_frequency_hz": spec["dominant_frequency_hz"],
    }


def compute_signal_stats(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    event_rows: list[dict[str, Any]] = []
    pmu_rows: list[dict[str, Any]] = []
    derivative_rows: list[dict[str, Any]] = []

    for event_id in TARGET_EVENTS:
        event_df = data.loc[data["EVENT"] == int(event_id)].copy()
        if event_df.empty:
            continue
        for channel in MEASUREMENT_CHANNELS:
            vals = event_df[channel].to_numpy(dtype=float)
            ts = event_df["TIMESTAMP"].to_numpy(dtype=float)
            stats = _series_stats(vals, ts)
            event_rows.append(
                {
                    "event": int(event_id),
                    "event_name": EVENT_NAMES.get(int(event_id), "other"),
                    "channel": channel,
                    "channel_family": _channel_family(channel),
                    "n_samples": int(np.isfinite(vals).sum()),
                    **stats,
                }
            )

    for (event_id, bus), group in data.groupby(["EVENT", "BUS"], sort=True):
        if int(event_id) not in TARGET_EVENTS:
            continue
        for channel in MEASUREMENT_CHANNELS:
            vals = group[channel].to_numpy(dtype=float)
            ts = group["TIMESTAMP"].to_numpy(dtype=float)
            stats = _series_stats(vals, ts)
            pmu_rows.append(
                {
                    "event": int(event_id),
                    "event_name": EVENT_NAMES.get(int(event_id), "other"),
                    "bus": str(bus),
                    "channel": channel,
                    "channel_family": _channel_family(channel),
                    "n_samples": int(np.isfinite(vals).sum()),
                    **stats,
                }
            )
            finite = np.isfinite(vals) & np.isfinite(ts)
            vals2 = vals[finite]
            ts2 = ts[finite]
            if vals2.size > 2:
                dx = np.diff(vals2)
                for v in dx[: min(len(dx), 2000)]:
                    derivative_rows.append(
                        {
                            "event": int(event_id),
                            "event_name": EVENT_NAMES.get(int(event_id), "other"),
                            "bus": str(bus),
                            "channel": channel,
                            "delta": float(v),
                        }
                    )

    return pd.DataFrame(event_rows), pd.DataFrame(pmu_rows), pd.DataFrame(derivative_rows)


def _phase_error_to_120(diff_deg: np.ndarray) -> np.ndarray:
    diff = _wrapped_deg(np.asarray(diff_deg, dtype=float))
    return np.minimum(np.abs(diff - 120.0), np.abs(diff + 120.0))


def compute_angle_stats(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (event_id, bus), group in data.groupby(["EVENT", "BUS"], sort=True):
        if int(event_id) not in TARGET_EVENTS:
            continue
        g = group.sort_values("TIMESTAMP")
        for channel in ANGLE_CHANNELS:
            arr = g[channel].to_numpy(dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue
            rad = np.deg2rad(arr)
            sin_mean = float(np.mean(np.sin(rad)))
            cos_mean = float(np.mean(np.cos(rad)))
            r = float(np.sqrt(sin_mean * sin_mean + cos_mean * cos_mean))
            wrapped_diff = _wrapped_diff_deg(arr)
            rows.append(
                {
                    "event": int(event_id),
                    "event_name": EVENT_NAMES.get(int(event_id), "other"),
                    "bus": str(bus),
                    "channel": channel,
                    "row_type": "channel",
                    "n_samples": int(arr.size),
                    "circular_mean_deg": float(np.rad2deg(np.arctan2(sin_mean, cos_mean))),
                    "circular_variance": float(1.0 - r),
                    "wrapped_diff_std_deg": _safe_std(wrapped_diff),
                    "abrupt_jump_rate_30deg": _safe_float(np.mean(np.abs(wrapped_diff) > 30.0)) if wrapped_diff.size else float("nan"),
                    "abrupt_jump_rate_90deg": _safe_float(np.mean(np.abs(wrapped_diff) > 90.0)) if wrapped_diff.size else float("nan"),
                    "sin_mean": sin_mean,
                    "cos_mean": cos_mean,
                    "phase_error_mean_deg": float("nan"),
                    "vi_phase_diff_std_deg": float("nan"),
                }
            )

        phase_cols = ["VA_ANG", "VB_ANG", "VC_ANG", "IA_ANG", "IB_ANG", "IC_ANG"]
        sub = g[phase_cols].copy()
        sub = sub.dropna()
        if not sub.empty:
            v_ab = _phase_error_to_120(sub["VA_ANG"].to_numpy(dtype=float) - sub["VB_ANG"].to_numpy(dtype=float))
            v_bc = _phase_error_to_120(sub["VB_ANG"].to_numpy(dtype=float) - sub["VC_ANG"].to_numpy(dtype=float))
            v_ca = _phase_error_to_120(sub["VC_ANG"].to_numpy(dtype=float) - sub["VA_ANG"].to_numpy(dtype=float))
            vi_a = _wrapped_deg(sub["VA_ANG"].to_numpy(dtype=float) - sub["IA_ANG"].to_numpy(dtype=float))
            vi_b = _wrapped_deg(sub["VB_ANG"].to_numpy(dtype=float) - sub["IB_ANG"].to_numpy(dtype=float))
            vi_c = _wrapped_deg(sub["VC_ANG"].to_numpy(dtype=float) - sub["IC_ANG"].to_numpy(dtype=float))
            rows.append(
                {
                    "event": int(event_id),
                    "event_name": EVENT_NAMES.get(int(event_id), "other"),
                    "bus": str(bus),
                    "channel": "__PHASE__",
                    "row_type": "summary",
                    "n_samples": int(len(sub)),
                    "circular_mean_deg": float("nan"),
                    "circular_variance": float("nan"),
                    "wrapped_diff_std_deg": float("nan"),
                    "abrupt_jump_rate_30deg": float("nan"),
                    "abrupt_jump_rate_90deg": float("nan"),
                    "sin_mean": float("nan"),
                    "cos_mean": float("nan"),
                    "phase_error_mean_deg": _safe_mean(np.concatenate([v_ab, v_bc, v_ca])),
                    "vi_phase_diff_std_deg": _safe_std(np.concatenate([vi_a, vi_b, vi_c])),
                }
            )
    return pd.DataFrame(rows)


def compute_freq_rocof_stats(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (event_id, bus), group in data.groupby(["EVENT", "BUS"], sort=True):
        if int(event_id) not in TARGET_EVENTS:
            continue
        g = group.sort_values("TIMESTAMP")
        freq = g["FREQ"].to_numpy(dtype=float)
        rocof = g["ROCOF"].to_numpy(dtype=float)
        ts = g["TIMESTAMP"].to_numpy(dtype=float)
        f_stats = _series_stats(freq, ts)
        r_stats = _series_stats(rocof, ts)
        finite = np.isfinite(freq) & np.isfinite(rocof)
        corr = float(np.corrcoef(freq[finite], rocof[finite])[0, 1]) if int(np.sum(finite)) > 4 else float("nan")
        rows.append(
            {
                "event": int(event_id),
                "event_name": EVENT_NAMES.get(int(event_id), "other"),
                "bus": str(bus),
                "n_samples": int(np.isfinite(freq).sum()),
                "freq_mean": f_stats["mean"],
                "freq_std": f_stats["std"],
                "freq_iqr": f_stats["iqr"],
                "freq_drift_slope_per_s": f_stats["drift_slope_per_s"],
                "freq_outlier_rate": f_stats["outlier_rate"],
                "freq_jump_rate": f_stats["jump_rate"],
                "freq_psd_lowfreq_ratio": f_stats["psd_lowfreq_ratio"],
                "rocof_mean": r_stats["mean"],
                "rocof_std": r_stats["std"],
                "rocof_iqr": r_stats["iqr"],
                "rocof_drift_slope_per_s": r_stats["drift_slope_per_s"],
                "rocof_outlier_rate": r_stats["outlier_rate"],
                "rocof_jump_rate": r_stats["jump_rate"],
                "rocof_psd_lowfreq_ratio": r_stats["psd_lowfreq_ratio"],
                "freq_rocof_corr": corr,
            }
        )
    return pd.DataFrame(rows)


def build_sample_feature_table(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (chunk_name, bus), group in data.groupby(["CHUNK_NAME", "BUS"], sort=True):
        g = group.sort_values("TIMESTAMP")
        for event_id in TARGET_EVENTS:
            e = g.loc[g["EVENT"] == int(event_id)].copy()
            if len(e) < 8:
                continue
            ts = e["TIMESTAMP"].to_numpy(dtype=float)
            dt = np.diff(ts)
            dt = dt[np.isfinite(dt) & (dt > 0.0)]
            dt_median = float(np.median(dt)) if dt.size else 0.033
            missing_frames = np.maximum(np.round(dt / max(dt_median, 1e-6)) - 1.0, 0.0)
            irregular_rate = float(np.mean(np.abs(dt - dt_median) > 0.15 * max(dt_median, 1e-6))) if dt.size else float("nan")

            channel_stats = []
            jump_rates = []
            stuck_rates = []
            outlier_rates = []
            angle_jump_rates = []
            angle_vars = []
            psd_low = []
            for channel in MEASUREMENT_CHANNELS:
                s = _series_stats(e[channel].to_numpy(dtype=float), ts)
                channel_stats.append(s)
                jump_rates.append(s["jump_rate"])
                stuck_rates.append(s["stuck_value_rate"])
                outlier_rates.append(s["outlier_rate"])
                psd_low.append(s["psd_lowfreq_ratio"])
            for channel in ANGLE_CHANNELS:
                arr = e[channel].to_numpy(dtype=float)
                arr = arr[np.isfinite(arr)]
                if arr.size < 3:
                    continue
                rad = np.deg2rad(arr)
                sin_mean = np.mean(np.sin(rad))
                cos_mean = np.mean(np.cos(rad))
                r = float(np.sqrt(sin_mean * sin_mean + cos_mean * cos_mean))
                angle_vars.append(float(1.0 - r))
                wd = _wrapped_diff_deg(arr)
                if wd.size:
                    angle_jump_rates.append(float(np.mean(np.abs(wd) > 30.0)))

            row_nan_fraction = e[MEASUREMENT_CHANNELS].isna().mean(axis=1).to_numpy(dtype=float)
            full_dropout = row_nan_fraction >= 0.95
            partial_dropout = (row_nan_fraction > 0.0) & (row_nan_fraction < 0.95)

            phase = e[["VA_ANG", "VB_ANG", "VC_ANG", "IA_ANG", "IB_ANG", "IC_ANG"]].dropna()
            if not phase.empty:
                v_ab = _phase_error_to_120(phase["VA_ANG"].to_numpy(dtype=float) - phase["VB_ANG"].to_numpy(dtype=float))
                v_bc = _phase_error_to_120(phase["VB_ANG"].to_numpy(dtype=float) - phase["VC_ANG"].to_numpy(dtype=float))
                v_ca = _phase_error_to_120(phase["VC_ANG"].to_numpy(dtype=float) - phase["VA_ANG"].to_numpy(dtype=float))
                phase_error_mean = _safe_mean(np.concatenate([v_ab, v_bc, v_ca]))
            else:
                phase_error_mean = float("nan")

            freq = e["FREQ"].to_numpy(dtype=float)
            rocof = e["ROCOF"].to_numpy(dtype=float)
            fr_mask = np.isfinite(freq) & np.isfinite(rocof)
            fr_corr = float(np.corrcoef(freq[fr_mask], rocof[fr_mask])[0, 1]) if int(np.sum(fr_mask)) > 4 else float("nan")

            rows.append(
                {
                    "chunk_name": chunk_name,
                    "bus": bus,
                    "event": int(event_id),
                    "event_name": EVENT_NAMES.get(int(event_id), "other"),
                    "n_rows": int(len(e)),
                    "duration_s": _safe_float(ts.max() - ts.min()) if ts.size > 1 else 0.0,
                    "dt_median_s": dt_median,
                    "dt_std_s": _safe_std(dt),
                    "irregular_spacing_rate": irregular_rate,
                    "missing_frame_rate_est": _safe_float(np.sum(missing_frames) / max(dt.size, 1)),
                    "duplicate_timestamp_rate": _safe_float(np.mean(dt <= 1e-12)) if dt.size else float("nan"),
                    "data_present_zero_rate": _safe_float(np.mean(e["DATA_PRESENT"].to_numpy(dtype=float) < 0.5)),
                    "nan_fraction_mean": _safe_mean(row_nan_fraction),
                    "nan_fraction_max": _safe_float(np.max(row_nan_fraction)) if row_nan_fraction.size else float("nan"),
                    "full_dropout_rate": _safe_float(np.mean(full_dropout)),
                    "partial_dropout_rate": _safe_float(np.mean(partial_dropout)),
                    "jump_rate_mean": _safe_mean(np.asarray(jump_rates, dtype=float)),
                    "stuck_rate_mean": _safe_mean(np.asarray(stuck_rates, dtype=float)),
                    "outlier_rate_mean": _safe_mean(np.asarray(outlier_rates, dtype=float)),
                    "angle_jump_rate_mean": _safe_mean(np.asarray(angle_jump_rates, dtype=float)),
                    "angle_circular_var_mean": _safe_mean(np.asarray(angle_vars, dtype=float)),
                    "phase_error_mean_deg": phase_error_mean,
                    "freq_mean": _safe_mean(freq),
                    "freq_std": _safe_std(freq),
                    "rocof_mean": _safe_mean(rocof),
                    "rocof_std": _safe_std(rocof),
                    "freq_rocof_corr": fr_corr,
                    "psd_lowfreq_ratio_mean": _safe_mean(np.asarray(psd_low, dtype=float)),
                }
            )
    return pd.DataFrame(rows)


def _ks_statistic(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size < 2 or y.size < 2:
        return float("nan"), float("nan")
    if ks_2samp is None:
        x_sorted = np.sort(x)
        y_sorted = np.sort(y)
        points = np.sort(np.concatenate([x_sorted, y_sorted]))
        cdf_x = np.searchsorted(x_sorted, points, side="right") / max(x_sorted.size, 1)
        cdf_y = np.searchsorted(y_sorted, points, side="right") / max(y_sorted.size, 1)
        stat = float(np.max(np.abs(cdf_x - cdf_y)))
        return stat, float("nan")
    out = ks_2samp(x, y)
    return float(out.statistic), float(out.pvalue)


def _wasserstein(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size < 2 or y.size < 2:
        return float("nan")
    if wasserstein_distance is None:
        return float(abs(np.mean(x) - np.mean(y)))
    return float(wasserstein_distance(x, y))


def _effect_size_cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size < 2 or y.size < 2:
        return float("nan")
    mean_diff = float(np.mean(x) - np.mean(y))
    var_x = float(np.var(x, ddof=1))
    var_y = float(np.var(y, ddof=1))
    pooled = ((x.size - 1) * var_x + (y.size - 1) * var_y) / max((x.size + y.size - 2), 1)
    denom = math.sqrt(max(pooled, 1e-12))
    return float(mean_diff / denom)


def rank_discriminative_features(sample_features: pd.DataFrame, event_a: int, event_b: int, random_seed: int) -> pd.DataFrame:
    pair = sample_features.loc[sample_features["event"].isin([event_a, event_b])].copy()
    if pair.empty:
        return pd.DataFrame(
            columns=[
                "feature",
                "event_a",
                "event_b",
                "mean_a",
                "mean_b",
                "effect_size",
                "effect_size_abs",
                "ks_stat",
                "ks_pvalue",
                "wasserstein",
                "wasserstein_norm",
                "mutual_info",
                "rank_score",
            ]
        )

    feature_cols = [
        col
        for col in pair.columns
        if col
        not in {
            "chunk_name",
            "bus",
            "event",
            "event_name",
        }
    ]
    feature_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(pair[col])]
    y = (pair["event"].to_numpy(dtype=int) == int(event_a)).astype(int)
    mi_map: dict[str, float] = {}
    if mutual_info_classif is not None and pair.shape[0] > 6:
        for col in feature_cols:
            values = pair[col].to_numpy(dtype=float)
            finite = np.isfinite(values)
            if int(np.sum(finite)) < 6:
                mi_map[col] = float("nan")
                continue
            x = values[finite].reshape(-1, 1)
            yy = y[finite]
            if np.unique(yy).size < 2:
                mi_map[col] = float("nan")
                continue
            try:
                mi_val = float(mutual_info_classif(x, yy, random_state=random_seed)[0])
            except Exception:
                mi_val = float("nan")
            mi_map[col] = mi_val
    else:
        for col in feature_cols:
            mi_map[col] = float("nan")

    rows: list[dict[str, Any]] = []
    for feature in feature_cols:
        a_vals = pair.loc[pair["event"] == int(event_a), feature].to_numpy(dtype=float)
        b_vals = pair.loc[pair["event"] == int(event_b), feature].to_numpy(dtype=float)
        a_vals = a_vals[np.isfinite(a_vals)]
        b_vals = b_vals[np.isfinite(b_vals)]
        if a_vals.size < 3 or b_vals.size < 3:
            continue
        d = _effect_size_cohens_d(a_vals, b_vals)
        ks_stat, ks_p = _ks_statistic(a_vals, b_vals)
        w = _wasserstein(a_vals, b_vals)
        scale = max(_safe_std(np.concatenate([a_vals, b_vals])), 1e-9)
        w_norm = w / scale if np.isfinite(w) else float("nan")
        mi = mi_map.get(feature, float("nan"))
        score = 0.0
        for item in [abs(d), ks_stat, w_norm, mi]:
            if np.isfinite(item):
                score += float(item)
        rows.append(
            {
                "feature": feature,
                "event_a": int(event_a),
                "event_b": int(event_b),
                "mean_a": _safe_mean(a_vals),
                "mean_b": _safe_mean(b_vals),
                "effect_size": d,
                "effect_size_abs": abs(d) if np.isfinite(d) else float("nan"),
                "ks_stat": ks_stat,
                "ks_pvalue": ks_p,
                "wasserstein": w,
                "wasserstein_norm": w_norm,
                "mutual_info": mi,
                "rank_score": score,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["rank_score", "effect_size_abs", "ks_stat"], ascending=False).reset_index(drop=True)


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        df = df.copy()
    df.to_csv(path, index=False)


def _plot_chunk_duration_histograms(chunk_inventory: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, event_id in zip(axes, [0, 5, 7]):
        subset = chunk_inventory.loc[chunk_inventory["chunk_event_id"] == event_id, "duration_s"].to_numpy(dtype=float)
        if subset.size == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
        else:
            ax.hist(subset, bins=min(20, max(5, subset.size)), color=EVENT_COLORS.get(event_id, "#808080"), alpha=0.8)
        ax.set_title(f"Event {event_id} ({EVENT_NAMES.get(event_id, 'other')})")
        ax.set_xlabel("Chunk duration [s]")
        ax.set_ylabel("Count")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_timestamp_gap_distributions(dt_samples: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = False
    for event_id in [0, 5, 7]:
        vals = dt_samples.loc[dt_samples["event"] == event_id, "dt_s"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        bins = np.linspace(max(0.0, float(np.quantile(vals, 0.01))), float(np.quantile(vals, 0.99)), 60)
        ax.hist(vals, bins=bins, density=True, alpha=0.35, label=f"Event {event_id}", color=EVENT_COLORS.get(event_id, "#777777"))
        plotted = True
    if not plotted:
        ax.text(0.5, 0.5, "No timestamp-gap samples", ha="center", va="center")
    ax.set_title("Timestamp Gap Distributions")
    ax.set_xlabel("dt [s]")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_data_present_transition_heatmap(transitions: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mat = np.zeros((3, 2), dtype=float)
    event_to_row = {0: 0, 5: 1, 7: 2}
    for event_id, group in transitions.groupby("event"):
        if int(event_id) not in event_to_row:
            continue
        row = event_to_row[int(event_id)]
        mat[row, 0] = float(group["transition_1_to_0"].sum())
        mat[row, 1] = float(group["transition_0_to_1"].sum())

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(mat, cmap="magma")
    ax.set_xticks([0, 1], ["1->0", "0->1"])
    ax.set_yticks([0, 1, 2], ["Event 0", "Event 5", "Event 7"])
    ax.set_title("DATA_PRESENT Transition Counts")
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            ax.text(c, r, f"{mat[r, c]:.0f}", ha="center", va="center", color="white")
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_nan_fraction_heatmap(missingness: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for event_id in [0, 5, 7]:
        sub = missingness.loc[
            (missingness["event"] == event_id) & (missingness["channel"] != "__ROW__"),
            ["bus", "channel", "nan_fraction"],
        ].copy()
        if sub.empty:
            continue
        pivot = sub.pivot_table(index="bus", columns="channel", values="nan_fraction", aggfunc="mean")
        pivot = pivot.reindex(index=BUS_ORDER, fill_value=np.nan)
        rows.append((event_id, pivot))
    if not rows:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No missingness data", ha="center", va="center")
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return
    fig, axes = plt.subplots(1, len(rows), figsize=(6 * len(rows), 5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (event_id, pivot) in zip(axes, rows):
        matrix = pivot.to_numpy(dtype=float)
        im = ax.imshow(matrix, cmap="viridis", aspect="auto")
        ax.set_title(f"Event {event_id} NaN Fraction")
        ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=90)
        ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_burst_length_histograms(burst_stats: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, event_id in zip(axes, [0, 5, 7]):
        vals = burst_stats.loc[burst_stats["event"] == event_id, "burst_length"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            ax.hist(vals, bins=min(30, max(8, int(np.sqrt(vals.size)))), color=EVENT_COLORS.get(event_id, "#777777"), alpha=0.85)
        else:
            ax.text(0.5, 0.5, "No bursts", ha="center", va="center")
        ax.set_title(f"Event {event_id} Burst Length")
        ax.set_xlabel("Length [frames]")
        ax.set_ylabel("Count")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_signal_distribution_panels(data: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    channels = ["VA_MAG", "IA_MAG", "FREQ", "ROCOF"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for ax, channel in zip(axes, channels):
        for event_id in [0, 5, 7]:
            vals = data.loc[data["EVENT"] == event_id, channel].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            lo = float(np.quantile(vals, 0.01))
            hi = float(np.quantile(vals, 0.99))
            bins = np.linspace(lo, hi, 60) if hi > lo else 40
            ax.hist(vals, bins=bins, density=True, alpha=0.35, color=EVENT_COLORS.get(event_id, "#777777"), label=f"E{event_id}")
        ax.set_title(channel)
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle("Signal Distribution Panels")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_derivative_distribution_panels(derivatives: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    channels = ["VA_MAG", "IA_MAG", "FREQ", "ROCOF"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for ax, channel in zip(axes, channels):
        sub = derivatives.loc[derivatives["channel"] == channel].copy()
        for event_id in [0, 5, 7]:
            vals = sub.loc[sub["event"] == event_id, "delta"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            lo = float(np.quantile(vals, 0.01))
            hi = float(np.quantile(vals, 0.99))
            bins = np.linspace(lo, hi, 60) if hi > lo else 40
            ax.hist(vals, bins=bins, density=True, alpha=0.35, color=EVENT_COLORS.get(event_id, "#777777"), label=f"E{event_id}")
        ax.set_title(f"d({channel})")
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle("Derivative Distribution Panels")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_angle_circular_panels(data: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    channels = ["VA_ANG", "IA_ANG"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={"projection": "polar"})
    axes = np.atleast_1d(axes)
    bins = np.linspace(-np.pi, np.pi, 36)
    for ax, channel in zip(axes, channels):
        for event_id in [0, 5, 7]:
            vals = data.loc[data["EVENT"] == event_id, channel].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            rad = np.deg2rad(_wrapped_deg(vals))
            hist, edges = np.histogram(rad, bins=bins, density=True)
            centers = 0.5 * (edges[:-1] + edges[1:])
            ax.plot(centers, hist, color=EVENT_COLORS.get(event_id, "#777777"), label=f"E{event_id}")
        ax.set_title(channel)
        ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    fig.suptitle("Angle Circular Panels")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_freq_rocof_panels(data: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True)
    for ax, event_id in zip(axes, [0, 5, 7]):
        sub = data.loc[data["EVENT"] == event_id, ["FREQ", "ROCOF"]].dropna()
        if sub.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
        else:
            sample = sub.sample(n=min(2000, len(sub)), random_state=42) if len(sub) > 2000 else sub
            ax.scatter(sample["FREQ"], sample["ROCOF"], s=8, alpha=0.35, color=EVENT_COLORS.get(event_id, "#777777"))
        ax.set_title(f"Event {event_id}")
        ax.grid(alpha=0.2)
        ax.set_xlabel("FREQ")
    axes[0].set_ylabel("ROCOF")
    fig.suptitle("Freq/ROCOF Panels")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_psd_panels(data: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, channel in zip(axes, ["FREQ", "VA_MAG"]):
        for event_id in [0, 5, 7]:
            sub = data.loc[data["EVENT"] == event_id, ["TIMESTAMP", channel]].dropna()
            if len(sub) < 16:
                continue
            vals = sub[channel].to_numpy(dtype=float)
            ts = sub["TIMESTAMP"].to_numpy(dtype=float)
            dt = np.diff(ts)
            dt = dt[np.isfinite(dt) & (dt > 0)]
            step = float(np.median(dt)) if dt.size else 0.033
            centered = vals - np.mean(vals)
            spec = np.fft.rfft(centered)
            power = np.abs(spec) ** 2
            freq = np.fft.rfftfreq(len(centered), d=step)
            if power.size > 1:
                ax.plot(freq[1:], power[1:], alpha=0.7, label=f"E{event_id}", color=EVENT_COLORS.get(event_id, "#777777"))
        ax.set_title(f"PSD - {channel}")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Power")
        ax.set_yscale("log")
        ax.grid(alpha=0.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_pmu_comparison_panels(signal_by_pmu: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Use summary over channels for interpretable PMU comparison.
    agg = (
        signal_by_pmu.groupby(["event", "bus"], as_index=False)[["outlier_rate", "jump_rate", "stuck_value_rate", "std"]]
        .mean(numeric_only=True)
        .rename(columns={"std": "avg_std"})
    )
    metrics = ["avg_std", "jump_rate", "outlier_rate"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
    for ax, metric in zip(axes, metrics):
        width = 0.25
        x = np.arange(len(BUS_ORDER))
        for idx, event_id in enumerate([0, 5, 7]):
            vals = []
            for bus in BUS_ORDER:
                sub = agg.loc[(agg["event"] == event_id) & (agg["bus"] == bus), metric]
                vals.append(float(sub.iloc[0]) if not sub.empty else np.nan)
            ax.bar(x + (idx - 1) * width, vals, width=width, label=f"E{event_id}", color=EVENT_COLORS.get(event_id, "#777777"), alpha=0.85)
        ax.set_title(metric)
        ax.set_xticks(x, BUS_ORDER, rotation=45)
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle("PMU Comparison Panels")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_discriminative_bars(df: pd.DataFrame, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    if df.empty:
        ax.text(0.5, 0.5, "No discriminative features", ha="center", va="center")
        ax.set_axis_off()
    else:
        top = df.head(15).iloc[::-1]
        ax.barh(top["feature"], top["rank_score"], color="#2a9d8f", alpha=0.9)
        ax.set_xlabel("Rank score")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _comparison_top_features(df: pd.DataFrame, n: int = 8) -> list[str]:
    if df.empty:
        return []
    return [str(item) for item in df["feature"].head(n).tolist()]


def _comparison_main_differences(
    sample_features: pd.DataFrame,
    event_a: int,
    event_b: int,
) -> list[str]:
    a = sample_features.loc[sample_features["event"] == int(event_a)]
    b = sample_features.loc[sample_features["event"] == int(event_b)]
    if a.empty or b.empty:
        return [f"Insufficient samples for Event {event_a} vs Event {event_b}."]

    messages: list[str] = []

    def delta(feature: str) -> float:
        return _safe_mean(a[feature].to_numpy(dtype=float)) - _safe_mean(b[feature].to_numpy(dtype=float))

    def describe_delta(feature: str, label: str) -> str:
        value = delta(feature)
        if np.isfinite(value):
            return f"{label}: {value:+.4f}"
        if feature in {"jump_rate_mean", "outlier_rate_mean", "phase_error_mean_deg", "freq_std"}:
            if _safe_mean(a["nan_fraction_mean"].to_numpy(dtype=float)) > 0.95:
                return f"{label}: not informative because Event {event_a} is dominated by complete dropout"
            if _safe_mean(b["nan_fraction_mean"].to_numpy(dtype=float)) > 0.95:
                return f"{label}: not informative because Event {event_b} is dominated by complete dropout"
        return f"{label}: insufficient finite samples"

    messages.append(describe_delta("nan_fraction_mean", "Missingness shift (nan_fraction_mean)"))
    messages.append(describe_delta("full_dropout_rate", "Full-dropout shift (full_dropout_rate)"))
    messages.append(describe_delta("jump_rate_mean", "Jumpiness shift (jump_rate_mean)"))
    messages.append(describe_delta("outlier_rate_mean", "Outlier-rate shift (outlier_rate_mean)"))
    messages.append(describe_delta("irregular_spacing_rate", "Timestamp irregularity shift"))
    messages.append(describe_delta("phase_error_mean_deg", "Phase-consistency shift (phase_error_mean_deg)"))
    messages.append(describe_delta("freq_std", "Frequency-noise shift (freq_std)"))
    return messages


def _stochastic_hypothesis(event_id: int) -> list[str]:
    if event_id == 5:
        return [
            "Two-state missingness process with burst persistence (ON/OFF dropout Markov chain).",
            "Separate full-dropout and partial-dropout states with PMU-specific transition probabilities.",
            "Burst-length and inter-burst intervals sampled from empirical heavy-tailed distributions.",
        ]
    if event_id == 7:
        return [
            "Mixture bad-data process combining spike, bias-drift, and stuck-value corruption modes.",
            "Mode switching with short persistence and PMU/channel-dependent activation rates.",
            "Amplitude corruption sampled conditionally on signal family (angle, magnitude, frequency, ROCOF).",
        ]
    return ["No hypothesis configured."]


def _should_model_separately(
    diff_50: pd.DataFrame,
    diff_70: pd.DataFrame,
    diff_57: pd.DataFrame,
) -> bool:
    if diff_57.empty:
        return True
    top57 = _comparison_top_features(diff_57, n=10)
    top70 = _comparison_top_features(diff_70, n=10)
    has_missingness_separation = any("nan_fraction" in item or "dropout" in item or "data_present" in item for item in top57)
    has_bad_data_signal_signature = any("jump" in item or "outlier" in item or "phase" in item or "freq_std" in item for item in top70)
    if has_missingness_separation and has_bad_data_signal_signature:
        return True
    # Conservative default: if event5-vs-event7 is strongly separable, keep separate models.
    top_score = _safe_float(diff_57["rank_score"].iloc[0]) if not diff_57.empty else float("nan")
    return bool(np.isfinite(top_score) and top_score > 1.0)


def _render_comparison_md(
    *,
    event_a: int,
    event_b: int,
    ranking: pd.DataFrame,
    main_differences: list[str],
    hypothesis: list[str],
) -> str:
    lines = [
        f"# Event {event_a} vs Event {event_b}",
        "",
        "## Main Differences",
    ]
    for item in main_differences:
        lines.append(f"- {item}")
    lines.extend(["", "## Most Discriminative Features"])
    top = ranking.head(15)
    if top.empty:
        lines.append("- No stable discriminative ranking available.")
    else:
        for _, row in top.iterrows():
            lines.append(
                "- "
                + f"{row['feature']}: score={_safe_float(row['rank_score']):.4f}, "
                + f"effect={_safe_float(row['effect_size']):+.4f}, "
                + f"KS={_safe_float(row['ks_stat']):.4f}, "
                + f"W={_safe_float(row['wasserstein_norm']):.4f}"
            )
    lines.extend(["", "## Stochastic Modeling Hypothesis"])
    for item in hypothesis:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def run_cyber_eda(chunks_root: Path, output_root: Path, random_seed: int = 42) -> dict[str, Any]:
    out_root = output_root.resolve()
    metrics_dir = out_root / "metrics"
    report_dir = out_root / "report"
    plots_dir = out_root / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    records = load_chunk_bus_records(chunks_root)
    data = records_to_dataframe(records)

    chunk_inventory = build_chunk_inventory(records)
    timestamp_gap_stats, dt_samples, transitions = compute_timestamp_gap_stats(data)
    missingness_stats, burst_stats = compute_missingness_stats(data)
    signal_stats_by_event, signal_stats_by_pmu, derivative_samples = compute_signal_stats(data)
    angle_circular_stats = compute_angle_stats(data)
    freq_rocof_stats = compute_freq_rocof_stats(data)
    sample_features = build_sample_feature_table(data)

    disc_5v0 = rank_discriminative_features(sample_features, event_a=5, event_b=0, random_seed=random_seed)
    disc_7v0 = rank_discriminative_features(sample_features, event_a=7, event_b=0, random_seed=random_seed)
    disc_5v7 = rank_discriminative_features(sample_features, event_a=5, event_b=7, random_seed=random_seed)

    _save_csv(chunk_inventory, metrics_dir / "chunk_inventory.csv")
    _save_csv(timestamp_gap_stats, metrics_dir / "timestamp_gap_stats.csv")
    _save_csv(missingness_stats, metrics_dir / "missingness_stats.csv")
    _save_csv(signal_stats_by_event, metrics_dir / "signal_stats_by_event.csv")
    _save_csv(signal_stats_by_pmu, metrics_dir / "signal_stats_by_pmu.csv")
    _save_csv(angle_circular_stats, metrics_dir / "angle_circular_stats.csv")
    _save_csv(freq_rocof_stats, metrics_dir / "freq_rocof_stats.csv")
    _save_csv(burst_stats, metrics_dir / "burst_stats.csv")
    _save_csv(disc_5v0, metrics_dir / "discriminative_features_event5_vs_event0.csv")
    _save_csv(disc_7v0, metrics_dir / "discriminative_features_event7_vs_event0.csv")
    _save_csv(disc_5v7, metrics_dir / "discriminative_features_event5_vs_event7.csv")

    _plot_chunk_duration_histograms(chunk_inventory, plots_dir / "chunk_duration_histograms.png")
    _plot_timestamp_gap_distributions(dt_samples, plots_dir / "timestamp_gap_distributions.png")
    _plot_data_present_transition_heatmap(transitions, plots_dir / "data_present_transition_heatmap.png")
    _plot_nan_fraction_heatmap(missingness_stats, plots_dir / "nan_fraction_heatmap.png")
    _plot_burst_length_histograms(burst_stats, plots_dir / "burst_length_histograms.png")
    _plot_signal_distribution_panels(data, plots_dir / "signal_distribution_panels.png")
    _plot_derivative_distribution_panels(derivative_samples, plots_dir / "derivative_distribution_panels.png")
    _plot_angle_circular_panels(data, plots_dir / "angle_circular_panels.png")
    _plot_freq_rocof_panels(data, plots_dir / "freq_rocof_panels.png")
    _plot_psd_panels(data, plots_dir / "psd_panels.png")
    _plot_pmu_comparison_panels(signal_stats_by_pmu, plots_dir / "pmu_comparison_panels.png")
    _plot_discriminative_bars(
        disc_5v0,
        plots_dir / "discriminative_feature_bars_event5_vs_event0.png",
        title="Top Discriminative Features: Event 5 vs Event 0",
    )
    _plot_discriminative_bars(
        disc_7v0,
        plots_dir / "discriminative_feature_bars_event7_vs_event0.png",
        title="Top Discriminative Features: Event 7 vs Event 0",
    )
    _plot_discriminative_bars(
        disc_5v7,
        plots_dir / "discriminative_feature_bars_event5_vs_event7.png",
        title="Top Discriminative Features: Event 5 vs Event 7",
    )

    diff_5v0 = _comparison_main_differences(sample_features, event_a=5, event_b=0)
    diff_7v0 = _comparison_main_differences(sample_features, event_a=7, event_b=0)
    diff_5v7 = _comparison_main_differences(sample_features, event_a=5, event_b=7)

    model_separate = _should_model_separately(disc_5v0, disc_7v0, disc_5v7)

    pmu_findings: dict[str, dict[str, float]] = {}
    pmu_summary = (
        signal_stats_by_pmu.loc[signal_stats_by_pmu["channel"].isin(["FREQ", "ROCOF", "VA_MAG", "IA_MAG"])]
        .groupby(["bus", "event"], as_index=False)[["jump_rate", "outlier_rate", "stuck_value_rate", "std"]]
        .mean(numeric_only=True)
    )
    for bus in BUS_ORDER:
        bus_rows = pmu_summary.loc[pmu_summary["bus"] == bus]
        pmu_findings[bus] = {}
        for event_id in [0, 5, 7]:
            row = bus_rows.loc[bus_rows["event"] == event_id]
            if row.empty:
                continue
            pmu_findings[bus][f"event_{event_id}_jump_rate"] = _safe_float(row["jump_rate"].iloc[0])
            pmu_findings[bus][f"event_{event_id}_outlier_rate"] = _safe_float(row["outlier_rate"].iloc[0])
            pmu_findings[bus][f"event_{event_id}_stuck_value_rate"] = _safe_float(row["stuck_value_rate"].iloc[0])
            pmu_findings[bus][f"event_{event_id}_avg_std"] = _safe_float(row["std"].iloc[0])

    report_json: dict[str, Any] = {
        "data_inventory": {
            "chunks_root": str(chunks_root.resolve()),
            "chunk_count": int(chunk_inventory["chunk_name"].nunique()),
            "chunk_count_by_event_folder": {
                str(k): int(v)
                for k, v in chunk_inventory.groupby("chunk_event_id")["chunk_name"].nunique().to_dict().items()
            },
            "rows_by_event_label": {
                str(k): int(v)
                for k, v in data.groupby("EVENT")["EVENT"].count().to_dict().items()
            },
            "rows_by_target_event": {
                "0": int((data["EVENT"] == 0).sum()),
                "5": int((data["EVENT"] == 5).sum()),
                "7": int((data["EVENT"] == 7).sum()),
            },
            "buses": sorted(data["BUS"].dropna().astype(str).unique().tolist()),
        },
        "event5_vs_event0": {
            "main_differences": diff_5v0,
            "most_discriminative_features": _comparison_top_features(disc_5v0, n=12),
            "stochastic_model_hypothesis": _stochastic_hypothesis(5),
        },
        "event7_vs_event0": {
            "main_differences": diff_7v0,
            "most_discriminative_features": _comparison_top_features(disc_7v0, n=12),
            "stochastic_model_hypothesis": _stochastic_hypothesis(7),
        },
        "event5_vs_event7": {
            "main_differences": diff_5v7,
            "should_be_modeled_separately": bool(model_separate),
            "recommended_separate_processes": [
                "Event 5: bursty missingness/dropout process with PMU-conditioned burst parameters.",
                "Event 7: bad-data corruption mixture (spike + drift/bias + stuck/replay-like modes).",
            ],
            "most_discriminative_features": _comparison_top_features(disc_5v7, n=12),
        },
        "pmu_specific_findings": pmu_findings,
        "recommended_next_modeling_step": {
            "event5_process": "Hidden Markov dropout process with states {normal, partial_dropout, full_dropout}, fitted per PMU and channel family.",
            "event7_process": "Switching corruption process with latent mode {spike, bias_drift, stuck, replay_like} and PMU/channel-conditioned amplitudes.",
            "noise_model": "Robust heavy-tailed residual model (Student-t or Gaussian-mixture) with channel-family specific scale.",
            "state_model": "Two-layer state model: global cyber state machine + local PMU mode transitions.",
            "parameters_to_fit_from_raw": [
                "dropout burst-length distribution",
                "inter-burst interval distribution",
                "DATA_PRESENT transition matrix",
                "event7 spike amplitude/width distributions",
                "event7 drift slope distribution",
                "stuck-run length distribution",
                "angle jump threshold exceedance probabilities",
                "frequency/ROCOF coupling and noise floor",
                "PMU-specific scaling factors",
            ],
        },
    }

    (report_dir / "event5_vs_event0.md").write_text(
        _render_comparison_md(
            event_a=5,
            event_b=0,
            ranking=disc_5v0,
            main_differences=diff_5v0,
            hypothesis=_stochastic_hypothesis(5),
        ),
        encoding="utf-8",
    )
    (report_dir / "event7_vs_event0.md").write_text(
        _render_comparison_md(
            event_a=7,
            event_b=0,
            ranking=disc_7v0,
            main_differences=diff_7v0,
            hypothesis=_stochastic_hypothesis(7),
        ),
        encoding="utf-8",
    )
    (report_dir / "event5_vs_event7.md").write_text(
        _render_comparison_md(
            event_a=5,
            event_b=7,
            ranking=disc_5v7,
            main_differences=diff_5v7,
            hypothesis=[
                "Event 5 and Event 7 should be represented by different stochastic corruption families.",
                "Event 5 is governed by availability and burst process dynamics.",
                "Event 7 is governed by value corruption dynamics (spike/bias/stuck/replay-like).",
            ],
        ),
        encoding="utf-8",
    )

    rec_md = [
        "# Stochastic Modeling Recommendations",
        "",
        "## Event 5 (Missing Data)",
        "- Fit PMU-wise Markov dropout transitions from DATA_PRESENT and NaN burst traces.",
        "- Use separate partial/full dropout states and empirical burst-length models.",
        "- Preserve per-channel family dropout tendencies (voltage/current/frequency/ROCOF).",
        "",
        "## Event 7 (Bad Data)",
        "- Use a mixture corruption model with latent modes: spike, bias-drift, stuck-value, replay-like.",
        "- Calibrate each mode by PMU and channel family.",
        "- Capture timestamp-irregularity coupling where present.",
        "",
        "## Shared",
        "- Keep baseline Event 0 residual model robust (heavy tails, PMU-conditioned scale).",
        "- Use a hierarchical cyber state model to generate realistic temporal persistence.",
        "",
    ]
    (report_dir / "stochastic_modeling_recommendations.md").write_text("\n".join(rec_md), encoding="utf-8")

    summary_md = [
        "# Cyber EDA Report",
        "",
        f"- Chunks analyzed: {int(chunk_inventory['chunk_name'].nunique())}",
        f"- Rows Event 0: {int((data['EVENT'] == 0).sum())}",
        f"- Rows Event 5: {int((data['EVENT'] == 5).sum())}",
        f"- Rows Event 7: {int((data['EVENT'] == 7).sum())}",
        "",
        "## Event 5 vs Event 0",
    ]
    summary_md.extend([f"- {line}" for line in diff_5v0[:6]])
    summary_md.append("")
    summary_md.append("## Event 7 vs Event 0")
    summary_md.extend([f"- {line}" for line in diff_7v0[:6]])
    summary_md.append("")
    summary_md.append("## Event 5 vs Event 7")
    summary_md.extend([f"- {line}" for line in diff_5v7[:6]])
    summary_md.extend(
        [
            "",
            f"- Should be modeled separately: {bool(model_separate)}",
            "",
            "## Recommended Next Step",
            f"- Event 5 process: {report_json['recommended_next_modeling_step']['event5_process']}",
            f"- Event 7 process: {report_json['recommended_next_modeling_step']['event7_process']}",
            f"- Noise model: {report_json['recommended_next_modeling_step']['noise_model']}",
            f"- State model: {report_json['recommended_next_modeling_step']['state_model']}",
            "",
        ]
    )
    (report_dir / "cyber_eda_report.md").write_text("\n".join(summary_md), encoding="utf-8")
    (report_dir / "cyber_eda_report.json").write_text(json.dumps(report_json, indent=2), encoding="utf-8")

    return report_json


def main() -> int:
    args = parse_args()
    run_cyber_eda(args.chunks_root, args.output_root, random_seed=args.random_seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
