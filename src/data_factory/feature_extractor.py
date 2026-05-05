from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ANGLE_SUFFIX = "_ANG"
MAG_SUFFIX = "_MAG"
PMU_BUSES = ("BUS2", "BUS5", "BUS6", "BUS10", "BUS19", "BUS22", "BUS29", "BUS39")
PHASE_GROUPS = {
    "V_MAG": ("VA_MAG", "VB_MAG", "VC_MAG"),
    "I_MAG": ("IA_MAG", "IB_MAG", "IC_MAG"),
    "V_ANG": ("VA_ANG", "VB_ANG", "VC_ANG"),
    "I_ANG": ("IA_ANG", "IB_ANG", "IC_ANG"),
}


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def _unwrap_angle_deg(values: np.ndarray) -> np.ndarray:
    return np.rad2deg(np.unwrap(np.deg2rad(values.astype(float))))


def _safe_nanmedian(values: np.ndarray, fallback: float = 0.0) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return fallback
    value = float(np.nanmedian(finite))
    return value if np.isfinite(value) else fallback


def _robust_scale(values: np.ndarray, fallback: float = 1.0) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) < 3:
        return fallback
    median = float(np.nanmedian(finite))
    mad = float(np.nanmedian(np.abs(finite - median)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-9:
        std = float(np.nanstd(finite))
        scale = std if np.isfinite(std) and std >= 1e-9 else fallback
    return float(scale)


def _max_abs_derivative(values: np.ndarray, dt: float) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) < 2:
        return 0.0
    deriv = np.gradient(values, dt)
    return float(np.nanmax(np.abs(deriv))) if np.isfinite(deriv).any() else 0.0


def _rolling_nanmean_max(values: np.ndarray, window: int = 5) -> float:
    finite_any = np.isfinite(values).any()
    if not finite_any:
        return 0.0
    rolled = pd.Series(values).rolling(window=window, center=True, min_periods=1).mean().to_numpy(dtype=float)
    return float(np.nanmax(rolled)) if np.isfinite(rolled).any() else 0.0


def zero_center_frame(frame: pd.DataFrame, pre_event_seconds: float = 3.0) -> pd.DataFrame:
    out = frame.copy()
    ts = _numeric(frame, "TIMESTAMP")
    if len(ts) < 2:
        return out
    pre_mask = ts <= float(np.nanmin(ts) + pre_event_seconds)
    for col in frame.columns:
        if col in {"TIMESTAMP", "Event", "DATA_PRESENT"}:
            continue
        if not pd.api.types.is_numeric_dtype(pd.to_numeric(frame[col], errors="coerce")):
            continue
        values = _numeric(frame, col)
        if col.endswith(ANGLE_SUFFIX):
            values = _unwrap_angle_deg(values)
        base = float(np.nanmean(values[pre_mask])) if np.any(pre_mask) else float(np.nanmean(values))
        out[col] = values - base
    return out


def extract_window_features(frame: pd.DataFrame, pre_event_seconds: float = 3.0) -> dict[str, Any]:
    centered = zero_center_frame(frame, pre_event_seconds=pre_event_seconds)
    ts = _numeric(centered, "TIMESTAMP")
    dt = float(np.nanmedian(np.diff(ts))) if len(ts) > 1 else 1.0 / 30.0
    features: dict[str, Any] = {"n_samples": int(len(centered)), "dt_s": dt}
    if len(ts) > 1 and np.isfinite(dt) and dt > 0:
        gaps = np.diff(ts)
        features["max_timestamp_gap_s"] = float(np.nanmax(gaps)) if np.isfinite(gaps).any() else 0.0
        features["max_timestamp_gap_ratio"] = float(features["max_timestamp_gap_s"] / dt)
    else:
        features["max_timestamp_gap_s"] = 0.0
        features["max_timestamp_gap_ratio"] = 1.0
    if "DATA_PRESENT" in centered:
        present = pd.to_numeric(centered["DATA_PRESENT"], errors="coerce").to_numpy(dtype=float)
        features["data_present_fraction"] = float(np.nanmean(present == 1)) if len(present) else 1.0
        features["data_present_min"] = float(np.nanmin(present)) if np.isfinite(present).any() else 1.0
    else:
        features["data_present_fraction"] = 1.0
        features["data_present_min"] = 1.0
    angle_arrays: list[np.ndarray] = []
    signal_scores: list[float] = []
    signal_nan_fractions: list[float] = []
    pre_mask = ts <= float(np.nanmin(ts) + pre_event_seconds) if len(ts) else np.array([], dtype=bool)

    for col in centered.columns:
        if col in {"TIMESTAMP", "Event", "DATA_PRESENT"}:
            continue
        values = _numeric(centered, col)
        signal_nan_fractions.append(float(np.mean(~np.isfinite(values))) if len(values) else 0.0)
        if not np.isfinite(values).any():
            continue
        features[f"{col}__min"] = float(np.nanmin(values))
        features[f"{col}__max"] = float(np.nanmax(values))
        features[f"{col}__span"] = float(np.nanmax(values) - np.nanmin(values))
        if len(values) > 1:
            max_deriv = _max_abs_derivative(values, dt)
            features[f"{col}__max_abs_derivative"] = max_deriv
            deriv = np.gradient(values, dt)
            pre_deriv = deriv[pre_mask] if len(pre_mask) == len(deriv) and np.any(pre_mask) else deriv[: max(3, min(len(deriv), 90))]
            deriv_scale = _robust_scale(pre_deriv, fallback=max(float(np.nanstd(deriv)), 1.0))
            deriv_score = max_deriv / max(deriv_scale, 1e-9)
            features[f"{col}__robust_derivative_score"] = float(deriv_score)
            signal_scores.append(float(deriv_score))
        pre_values = values[pre_mask] if len(pre_mask) == len(values) and np.any(pre_mask) else values[: max(3, min(len(values), 90))]
        center = _safe_nanmedian(pre_values)
        scale = _robust_scale(pre_values, fallback=max(float(np.nanstd(values[np.isfinite(values)])), 1.0))
        robust_z = np.abs((values - center) / max(scale, 1e-9))
        if np.isfinite(robust_z).any():
            zmax = float(np.nanmax(robust_z))
            features[f"{col}__max_abs_robust_z"] = zmax
            features[f"{col}__sustained_abs_robust_z_5"] = _rolling_nanmean_max(robust_z, window=5)
            features[f"{col}__sustained_abs_robust_z_15"] = _rolling_nanmean_max(robust_z, window=15)
            signal_scores.append(zmax)
        if col.endswith(ANGLE_SUFFIX):
            angle_arrays.append(values)

    features["nan_fraction_max"] = float(max(signal_nan_fractions)) if signal_nan_fractions else 0.0
    features["nan_fraction_mean"] = float(np.mean(signal_nan_fractions)) if signal_nan_fractions else 0.0
    if signal_scores:
        ordered = sorted(signal_scores, reverse=True)
        features["single_signal_score_max"] = float(ordered[0])
        features["single_signal_score_second"] = float(ordered[1]) if len(ordered) > 1 else 0.0
        features["single_signal_score_ratio"] = float(ordered[0] / max(ordered[1], 1e-9)) if len(ordered) > 1 else float(ordered[0])
        features["single_signal_score_count_gt20"] = float(sum(score > 20.0 for score in signal_scores))
    else:
        features["single_signal_score_max"] = 0.0
        features["single_signal_score_second"] = 0.0
        features["single_signal_score_ratio"] = 0.0
        features["single_signal_score_count_gt20"] = 0.0

    for group, suffixes in PHASE_GROUPS.items():
        cols = [col for col in centered.columns if any(col.endswith(suffix) for suffix in suffixes)]
        if len(cols) != 3:
            continue
        arrays = [_numeric(centered, col) for col in cols]
        n = min(len(item) for item in arrays)
        if n == 0:
            continue
        mat = np.vstack([item[:n] for item in arrays])
        valid = np.isfinite(mat).any(axis=0)
        if not valid.any():
            continue
        spread = np.nanmax(mat[:, valid], axis=0) - np.nanmin(mat[:, valid], axis=0)
        features[f"{group}_phase_spread__max"] = float(np.nanmax(spread))
        features[f"{group}_phase_spread__mean"] = float(np.nanmean(spread))
        valid_idx = np.where(valid)[0]
        max_valid_idx = int(valid_idx.max()) if len(valid_idx) else -1
        pre_valid = pre_mask[valid_idx] if len(pre_mask) > max_valid_idx else np.zeros(len(spread), dtype=bool)
        pre_spread = spread[pre_valid] if len(pre_valid) == len(spread) and np.any(pre_valid) else spread[: max(3, min(len(spread), 90))]
        spread_center = _safe_nanmedian(pre_spread)
        spread_scale = _robust_scale(pre_spread, fallback=max(float(np.nanstd(spread[np.isfinite(spread)])), 1.0))
        spread_z = np.abs((spread - spread_center) / max(spread_scale, 1e-9))
        features[f"{group}_phase_spread__max_abs_robust_z"] = float(np.nanmax(spread_z)) if np.isfinite(spread_z).any() else 0.0
        features[f"{group}_phase_spread__sustained_abs_robust_z_5"] = _rolling_nanmean_max(spread_z, window=5)
        features[f"{group}_phase_spread__sustained_abs_robust_z_15"] = _rolling_nanmean_max(spread_z, window=15)

    freq_cols = [col for col in centered.columns if col.endswith("_Freq")]
    for freq_col in freq_cols:
        bus_prefix = freq_col[: -len("_Freq")]
        angle_col = f"{bus_prefix}_VA_ANG"
        if angle_col not in centered:
            continue
        freq = _numeric(centered, freq_col)
        angle = _numeric(centered, angle_col)
        if len(freq) < 3 or len(angle) < 3:
            continue
        n = min(len(freq), len(angle))
        freq = freq[:n]
        angle = angle[:n]
        angle_freq = np.gradient(angle, dt) / 360.0
        local_pre = pre_mask[:n] if len(pre_mask) >= n and np.any(pre_mask[:n]) else np.arange(n) < max(3, min(n, 90))
        freq_delta = freq - _safe_nanmedian(freq[local_pre])
        angle_freq_delta = angle_freq - _safe_nanmedian(angle_freq[local_pre])
        mismatch = freq_delta - angle_freq_delta
        features["freq_angle_mismatch__max_abs"] = float(np.nanmax(np.abs(mismatch))) if np.isfinite(mismatch).any() else 0.0
        pre_mismatch = mismatch[local_pre]
        mismatch_scale = _robust_scale(pre_mismatch, fallback=max(float(np.nanstd(mismatch[np.isfinite(mismatch)])), 1e-4))
        mismatch_z = np.abs((mismatch - _safe_nanmedian(pre_mismatch)) / max(mismatch_scale, 1e-9))
        features["freq_angle_mismatch__max_abs_robust_z"] = float(np.nanmax(mismatch_z)) if np.isfinite(mismatch_z).any() else 0.0
        features["freq_angle_mismatch__sustained_abs_robust_z_5"] = _rolling_nanmean_max(mismatch_z, window=5)
        features["freq_angle_mismatch__sustained_abs_robust_z_15"] = _rolling_nanmean_max(mismatch_z, window=15)

    if angle_arrays:
        n = min(len(values) for values in angle_arrays)
        mat = np.vstack([values[:n] for values in angle_arrays])
        valid_cols = np.isfinite(mat).any(axis=0)
        if valid_cols.any():
            mat_valid = mat[:, valid_cols]
            spread = np.nanmax(mat_valid, axis=0) - np.nanmin(mat_valid, axis=0)
            features["pmu_angle_spread__max"] = float(np.nanmax(spread))
            features["pmu_angle_spread__mean"] = float(np.nanmean(spread))
        else:
            features["pmu_angle_spread__max"] = np.nan
            features["pmu_angle_spread__mean"] = np.nan

    event = pd.to_numeric(centered.get("Event", pd.Series(dtype=float)), errors="coerce")
    features["label_event_mode"] = int(event.mode().iloc[0]) if len(event.dropna()) else 0
    features["label_abnormal"] = int(features["label_event_mode"] != 0)
    return features


def extract_features_from_csvs(input_dir: Path, out_csv: Path, pre_event_seconds: float = 3.0) -> pd.DataFrame:
    rows = []
    for path in sorted(input_dir.glob("*.csv")):
        frame = pd.read_csv(path)
        row = {"source_file": str(path.resolve())}
        row.update(extract_window_features(frame, pre_event_seconds=pre_event_seconds))
        rows.append(row)
    features = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_csv, index=False)
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract zero-centered window features from PMU CSV files.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--pre-event-seconds", type=float, default=3.0)
    args = parser.parse_args()
    features = extract_features_from_csvs(args.input_dir, args.out_csv, pre_event_seconds=args.pre_event_seconds)
    print(json.dumps({"rows": int(len(features)), "out_csv": str(args.out_csv.resolve())}, indent=2))


if __name__ == "__main__":
    main()
