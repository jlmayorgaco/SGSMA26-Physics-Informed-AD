from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ANGLE_SUFFIX = "_ANG"
PHASES = ("A", "B", "C")
PHASE_GROUPS = {
    "V_MAG": ("VA_MAG", "VB_MAG", "VC_MAG"),
    "I_MAG": ("IA_MAG", "IB_MAG", "IC_MAG"),
    "V_ANG": ("VA_ANG", "VB_ANG", "VC_ANG"),
    "I_ANG": ("IA_ANG", "IB_ANG", "IC_ANG"),
}
META_COLUMNS = {"TIMESTAMP", "Event", "DATA_PRESENT"}


_ALGEBRAIC_LOCALIZER: Any | None | bool = None


def _algebraic_localizer() -> Any | None:
    global _ALGEBRAIC_LOCALIZER
    if _ALGEBRAIC_LOCALIZER is None:
        try:
            from src.data_factory.algebraic_residual_localizer_v4 import (
                DEFAULT_BRANCHES,
                DEFAULT_YBUS,
                DEFAULT_ZBUS,
                AlgebraicResidualLocalizerV4,
            )
        except ModuleNotFoundError:
            _ALGEBRAIC_LOCALIZER = False
        else:
            _ALGEBRAIC_LOCALIZER = AlgebraicResidualLocalizerV4(DEFAULT_YBUS, DEFAULT_ZBUS, DEFAULT_BRANCHES)
    return None if _ALGEBRAIC_LOCALIZER is False else _ALGEBRAIC_LOCALIZER


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def _unwrap_angle_deg(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    if finite.sum() < 2:
        return values.astype(float)
    filled = pd.Series(values).interpolate(limit_direction="both").to_numpy(dtype=float)
    out = np.rad2deg(np.unwrap(np.deg2rad(filled)))
    out[~finite] = np.nan
    return out


def _safe_finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def _nanmedian(values: np.ndarray, fallback: float = 0.0) -> float:
    finite = _safe_finite(values)
    if len(finite) == 0:
        return fallback
    value = float(np.nanmedian(finite))
    return value if np.isfinite(value) else fallback


def _nanmean(values: np.ndarray, fallback: float = 0.0) -> float:
    finite = _safe_finite(values)
    if len(finite) == 0:
        return fallback
    value = float(np.nanmean(finite))
    return value if np.isfinite(value) else fallback


def _nanstd(values: np.ndarray, fallback: float = 0.0) -> float:
    finite = _safe_finite(values)
    if len(finite) < 2:
        return fallback
    value = float(np.nanstd(finite))
    return value if np.isfinite(value) else fallback


def _robust_scale(values: np.ndarray, fallback: float = 1.0) -> float:
    finite = _safe_finite(values)
    if len(finite) < 3:
        return fallback
    median = float(np.nanmedian(finite))
    mad = float(np.nanmedian(np.abs(finite - median)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-9:
        scale = _nanstd(finite, fallback=fallback)
    return float(scale if np.isfinite(scale) and scale >= 1e-9 else fallback)


def _rolling_nanmean_max(values: np.ndarray, window: int) -> float:
    if not np.isfinite(values).any():
        return 0.0
    rolled = pd.Series(values).rolling(window=window, center=True, min_periods=1).mean()
    out = rolled.to_numpy(dtype=float)
    return float(np.nanmax(out)) if np.isfinite(out).any() else 0.0


def _max_abs_derivative(values: np.ndarray, dt: float) -> float:
    if len(values) < 2 or not np.isfinite(values).any() or not np.isfinite(dt) or dt <= 0:
        return 0.0
    filled = pd.Series(values).interpolate(limit_direction="both").to_numpy(dtype=float)
    deriv = np.gradient(filled, dt)
    return float(np.nanmax(np.abs(deriv))) if np.isfinite(deriv).any() else 0.0


def _rms_derivative(values: np.ndarray, dt: float) -> float:
    if len(values) < 2 or not np.isfinite(values).any() or not np.isfinite(dt) or dt <= 0:
        return 0.0
    filled = pd.Series(values).interpolate(limit_direction="both").to_numpy(dtype=float)
    deriv = np.gradient(filled, dt)
    return float(np.sqrt(np.nanmean(deriv * deriv))) if np.isfinite(deriv).any() else 0.0


def _longest_false_run(mask: np.ndarray) -> int:
    if len(mask) == 0:
        return 0
    best = current = 0
    for value in mask:
        if not bool(value):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def _segment_masks(ts: np.ndarray, pre_event_seconds: float) -> dict[str, np.ndarray]:
    if len(ts) == 0:
        empty = np.array([], dtype=bool)
        return {key: empty for key in ("pre", "early", "mid", "late", "full")}
    t0 = float(np.nanmin(ts))
    t1 = float(np.nanmax(ts))
    pre_end = t0 + float(pre_event_seconds)
    late_start = max(pre_end, t1 - 5.0)
    return {
        "pre": ts <= pre_end,
        "early": (ts > pre_end) & (ts <= pre_end + 5.0),
        "mid": (ts > pre_end + 5.0) & (ts <= pre_end + 15.0),
        "late": ts >= late_start,
        "full": np.ones(len(ts), dtype=bool),
    }


def _window_stats(prefix: str, values: np.ndarray, mask: np.ndarray, features: dict[str, Any]) -> None:
    segment = values[mask] if len(mask) == len(values) else values
    finite = _safe_finite(segment)
    if len(finite) == 0:
        features[f"{prefix}__mean"] = 0.0
        features[f"{prefix}__std"] = 0.0
        features[f"{prefix}__max_abs"] = 0.0
        features[f"{prefix}__span"] = 0.0
        return
    features[f"{prefix}__mean"] = float(np.nanmean(finite))
    features[f"{prefix}__std"] = float(np.nanstd(finite)) if len(finite) > 1 else 0.0
    features[f"{prefix}__min"] = float(np.nanmin(finite))
    features[f"{prefix}__max"] = float(np.nanmax(finite))
    features[f"{prefix}__span"] = float(np.nanmax(finite) - np.nanmin(finite))
    features[f"{prefix}__median"] = float(np.nanmedian(finite))
    features[f"{prefix}__max_abs"] = float(np.nanmax(np.abs(finite)))
    features[f"{prefix}__mean_abs"] = float(np.nanmean(np.abs(finite)))
    features[f"{prefix}__p05"] = float(np.nanpercentile(finite, 5))
    features[f"{prefix}__p95"] = float(np.nanpercentile(finite, 95))


def _prepare_signal(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = _numeric(frame, column)
    if column.endswith(ANGLE_SUFFIX):
        return _unwrap_angle_deg(values)
    return values


def zero_reference_frame(frame: pd.DataFrame, pre_event_seconds: float = 3.0) -> pd.DataFrame:
    out = frame.copy()
    ts = _numeric(frame, "TIMESTAMP")
    masks = _segment_masks(ts, pre_event_seconds)
    pre_mask = masks["pre"]
    for col in frame.columns:
        if col in META_COLUMNS:
            continue
        values = _prepare_signal(frame, col)
        base = _nanmedian(values[pre_mask], fallback=_nanmedian(values))
        out[col] = values - base
    return out


def _add_signal_features(
    features: dict[str, Any],
    column: str,
    raw_values: np.ndarray,
    centered_values: np.ndarray,
    masks: dict[str, np.ndarray],
    dt: float,
) -> None:
    pre_values = raw_values[masks["pre"]] if len(masks["pre"]) == len(raw_values) and masks["pre"].any() else raw_values
    center = _nanmedian(pre_values)
    scale = _robust_scale(pre_values, fallback=max(_nanstd(raw_values), 1.0))
    robust_z = np.abs((raw_values - center) / max(scale, 1e-9))

    _window_stats(f"{column}__full", centered_values, masks["full"], features)
    for name in ("pre", "early", "mid", "late"):
        _window_stats(f"{column}__{name}", centered_values, masks[name], features)
    features[f"{column}__early_pre_delta"] = features.get(f"{column}__early__median", 0.0) - features.get(f"{column}__pre__median", 0.0)
    features[f"{column}__mid_pre_delta"] = features.get(f"{column}__mid__median", 0.0) - features.get(f"{column}__pre__median", 0.0)
    features[f"{column}__late_pre_delta"] = features.get(f"{column}__late__median", 0.0) - features.get(f"{column}__pre__median", 0.0)
    features[f"{column}__max_abs_derivative"] = _max_abs_derivative(centered_values, dt)
    features[f"{column}__rms_derivative"] = _rms_derivative(centered_values, dt)
    if np.isfinite(robust_z).any():
        features[f"{column}__max_abs_robust_z"] = float(np.nanmax(robust_z))
        features[f"{column}__sustained_abs_robust_z_5"] = _rolling_nanmean_max(robust_z, 5)
        features[f"{column}__sustained_abs_robust_z_15"] = _rolling_nanmean_max(robust_z, 15)
    else:
        features[f"{column}__max_abs_robust_z"] = 0.0
        features[f"{column}__sustained_abs_robust_z_5"] = 0.0
        features[f"{column}__sustained_abs_robust_z_15"] = 0.0


def _phase_columns(frame: pd.DataFrame, kind: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for phase in PHASES:
        suffix = f"V{phase}_{kind}" if kind in {"MAG", "ANG"} else ""
        if kind.startswith("I"):
            suffix = f"I{phase}_{kind.split('_', 1)[1]}"
        matches = [col for col in frame.columns if col.endswith(suffix)]
        if matches:
            out[phase] = matches[0]
    return out


def _add_phase_spread_features(
    features: dict[str, Any],
    group: str,
    centered: pd.DataFrame,
    masks: dict[str, np.ndarray],
) -> None:
    suffixes = PHASE_GROUPS[group]
    cols = [col for col in centered.columns if any(col.endswith(suffix) for suffix in suffixes)]
    if len(cols) < 3:
        return
    arrays = [_numeric(centered, col) for col in cols[:3]]
    n = min(len(item) for item in arrays)
    if n == 0:
        return
    mat = np.vstack([item[:n] for item in arrays])
    valid = np.isfinite(mat).any(axis=0)
    if not valid.any():
        return
    spread = np.nanmax(mat[:, valid], axis=0) - np.nanmin(mat[:, valid], axis=0)
    valid_idx = np.where(valid)[0]
    for name, mask in masks.items():
        local_mask = mask[valid_idx] if len(mask) > int(valid_idx.max()) else np.ones(len(spread), dtype=bool)
        _window_stats(f"{group}_phase_spread__{name}", spread, local_mask, features)
    pre_mask = masks["pre"][valid_idx] if len(masks["pre"]) > int(valid_idx.max()) else np.ones(len(spread), dtype=bool)
    pre_spread = spread[pre_mask] if pre_mask.any() else spread
    center = _nanmedian(pre_spread)
    scale = _robust_scale(pre_spread, fallback=max(_nanstd(spread), 1.0))
    z = np.abs((spread - center) / max(scale, 1e-9))
    features[f"{group}_phase_spread__max_abs_robust_z"] = float(np.nanmax(z)) if np.isfinite(z).any() else 0.0
    features[f"{group}_phase_spread__sustained_abs_robust_z_5"] = _rolling_nanmean_max(z, 5)
    features[f"{group}_phase_spread__sustained_abs_robust_z_15"] = _rolling_nanmean_max(z, 15)


def _add_power_proxy_features(
    features: dict[str, Any],
    frame: pd.DataFrame,
    masks: dict[str, np.ndarray],
    dt: float,
) -> None:
    for phase in PHASES:
        vmag = [col for col in frame.columns if col.endswith(f"V{phase}_MAG")]
        imag = [col for col in frame.columns if col.endswith(f"I{phase}_MAG")]
        vang = [col for col in frame.columns if col.endswith(f"V{phase}_ANG")]
        iang = [col for col in frame.columns if col.endswith(f"I{phase}_ANG")]
        if not (vmag and imag and vang and iang):
            continue
        vmag_raw = _numeric(frame, vmag[0])
        imag_raw = _numeric(frame, imag[0])
        vang_raw = _unwrap_angle_deg(_numeric(frame, vang[0]))
        iang_raw = _unwrap_angle_deg(_numeric(frame, iang[0]))
        n = min(len(vmag_raw), len(imag_raw), len(vang_raw), len(iang_raw))
        if n == 0:
            continue
        vmag_raw, imag_raw, vang_raw, iang_raw = vmag_raw[:n], imag_raw[:n], vang_raw[:n], iang_raw[:n]
        pre = masks["pre"][:n] if len(masks["pre"]) >= n and masks["pre"].any() else np.ones(n, dtype=bool)
        vpu = vmag_raw / max(abs(_nanmedian(vmag_raw[pre], fallback=1.0)), 1e-9)
        ipu = imag_raw / max(abs(_nanmedian(imag_raw[pre], fallback=1.0)), 1e-9)
        phi = np.deg2rad(vang_raw - iang_raw)
        p = vpu * ipu * np.cos(phi)
        q = vpu * ipu * np.sin(phi)
        pf_angle = np.rad2deg(np.unwrap(np.angle(np.exp(1j * phi))))
        for name, values in ((f"P_PROXY_{phase}", p), (f"Q_PROXY_{phase}", q), (f"PF_ANG_{phase}", pf_angle)):
            base = _nanmedian(values[pre], fallback=_nanmedian(values))
            centered = values - base
            for segment in ("early", "mid", "late", "full"):
                mask = masks[segment][:n] if len(masks[segment]) >= n else np.ones(n, dtype=bool)
                _window_stats(f"{name}__{segment}", centered, mask, features)
            features[f"{name}__max_abs_derivative"] = _max_abs_derivative(centered, dt)


def extract_window_features_v2(frame: pd.DataFrame, pre_event_seconds: float = 3.0) -> dict[str, Any]:
    ts = _numeric(frame, "TIMESTAMP")
    if len(ts) > 1:
        gaps = np.diff(ts)
        dt = float(np.nanmedian(gaps)) if np.isfinite(gaps).any() else 1.0 / 30.0
    else:
        gaps = np.array([], dtype=float)
        dt = 1.0 / 30.0
    centered = zero_reference_frame(frame, pre_event_seconds=pre_event_seconds)
    masks = _segment_masks(ts, pre_event_seconds)
    features: dict[str, Any] = {
        "n_samples": int(len(frame)),
        "dt_s": float(dt),
        "duration_s": float(np.nanmax(ts) - np.nanmin(ts)) if len(ts) and np.isfinite(ts).any() else 0.0,
        "max_timestamp_gap_s": float(np.nanmax(gaps)) if len(gaps) and np.isfinite(gaps).any() else 0.0,
    }
    features["max_timestamp_gap_ratio"] = float(features["max_timestamp_gap_s"] / max(dt, 1e-9))
    if "DATA_PRESENT" in frame:
        present = pd.to_numeric(frame["DATA_PRESENT"], errors="coerce").fillna(0).to_numpy(dtype=float) == 1.0
    else:
        present = np.ones(len(frame), dtype=bool)
    features["data_present_fraction"] = float(np.mean(present)) if len(present) else 1.0
    features["data_present_min"] = float(np.min(present.astype(float))) if len(present) else 1.0
    features["missing_run_max_samples"] = _longest_false_run(present)
    features["missing_run_max_s"] = float(features["missing_run_max_samples"] * max(dt, 0.0))
    for segment, mask in masks.items():
        local = present[mask] if len(mask) == len(present) and mask.any() else present
        features[f"data_present_fraction__{segment}"] = float(np.mean(local)) if len(local) else 1.0

    nan_fractions: list[float] = []
    score_values: list[float] = []
    for col in frame.columns:
        if col in META_COLUMNS:
            continue
        raw_values = _prepare_signal(frame, col)
        centered_values = _numeric(centered, col)
        nan_fractions.append(float(np.mean(~np.isfinite(raw_values))) if len(raw_values) else 0.0)
        if not np.isfinite(raw_values).any():
            continue
        _add_signal_features(features, col, raw_values, centered_values, masks, dt)
        score_values.append(float(features.get(f"{col}__max_abs_robust_z", 0.0)))
        score_values.append(float(features.get(f"{col}__max_abs_derivative", 0.0)))

    features["nan_fraction_max"] = float(max(nan_fractions)) if nan_fractions else 0.0
    features["nan_fraction_mean"] = float(np.mean(nan_fractions)) if nan_fractions else 0.0
    if score_values:
        ordered = sorted(score_values, reverse=True)
        features["single_signal_score_max"] = float(ordered[0])
        features["single_signal_score_second"] = float(ordered[1]) if len(ordered) > 1 else 0.0
        features["single_signal_score_ratio"] = float(ordered[0] / max(ordered[1], 1e-9)) if len(ordered) > 1 else float(ordered[0])
        features["single_signal_score_count_gt20"] = float(sum(score > 20.0 for score in score_values))
    else:
        features["single_signal_score_max"] = 0.0
        features["single_signal_score_second"] = 0.0
        features["single_signal_score_ratio"] = 0.0
        features["single_signal_score_count_gt20"] = 0.0

    for group in PHASE_GROUPS:
        _add_phase_spread_features(features, group, centered, masks)
    _add_power_proxy_features(features, frame, masks, dt)

    angle_cols = [col for col in centered.columns if col.endswith(ANGLE_SUFFIX)]
    if angle_cols:
        arrays = [_numeric(centered, col) for col in angle_cols]
        n = min(len(values) for values in arrays)
        if n:
            mat = np.vstack([values[:n] for values in arrays])
            valid = np.isfinite(mat).any(axis=0)
            if valid.any():
                spread = np.nanmax(mat[:, valid], axis=0) - np.nanmin(mat[:, valid], axis=0)
                features["pmu_angle_spread__max"] = float(np.nanmax(spread))
                features["pmu_angle_spread__mean"] = float(np.nanmean(spread))

    algebraic_localizer = _algebraic_localizer()
    if algebraic_localizer is not None:
        features.update(algebraic_localizer.extract_from_wide_frame(frame, pre_event_seconds=pre_event_seconds))

    event = pd.to_numeric(frame.get("Event", pd.Series(dtype=float)), errors="coerce")
    features["label_event_mode"] = int(event.mode().iloc[0]) if len(event.dropna()) else 0
    features["label_abnormal"] = int(features["label_event_mode"] != 0)
    return features


def extract_features_from_csvs(input_dir: Path, out_csv: Path, pre_event_seconds: float = 3.0) -> pd.DataFrame:
    rows = []
    for path in sorted(input_dir.glob("*.csv")):
        frame = pd.read_csv(path)
        row = {"source_file": str(path.resolve())}
        row.update(extract_window_features_v2(frame, pre_event_seconds=pre_event_seconds))
        rows.append(row)
    features = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_csv, index=False)
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract stable SGSMA v2 PMU features.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--pre-event-seconds", type=float, default=3.0)
    args = parser.parse_args()
    features = extract_features_from_csvs(args.input_dir, args.out_csv, args.pre_event_seconds)
    print(json.dumps({"rows": int(len(features)), "out_csv": str(args.out_csv.resolve())}, indent=2))


if __name__ == "__main__":
    main()
