from __future__ import annotations

import math
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import hilbert

from src.features.pmu_discovery import discover_bus_csvs, infer_pmu_buses_from_frames

LEGACY_PMU_BUSES = (2, 5, 6, 10, 19, 22, 29, 39)
PMU_BUSES = LEGACY_PMU_BUSES
SIGNALS = (
    "VA_MAG",
    "VB_MAG",
    "VC_MAG",
    "IA_MAG",
    "IB_MAG",
    "IC_MAG",
    "VA_ANG",
    "VB_ANG",
    "VC_ANG",
    "IA_ANG",
    "IB_ANG",
    "IC_ANG",
    "Freq",
    "ROCOF",
)
ANGLE_SIGNALS = tuple(signal for signal in SIGNALS if signal.endswith("_ANG"))
GRAPH_SIGNALS = ("VA_MAG", "IA_MAG", "VA_ANG", "IA_ANG", "Freq", "ROCOF")
RESIDUAL_TOPOLOGY_SIGNALS = ("VA_MAG", "IA_MAG", "VA_ANG", "IA_ANG", "Freq", "ROCOF")
ROLLING_WINDOWS = (5, 15, 30, 60, 120)
ROLLING_SIGNALS = SIGNALS
RLS_SIGNALS = ("VA_MAG", "IA_MAG", "VA_ANG", "IA_ANG", "Freq", "ROCOF")
HILBERT_SIGNALS = ("VA_MAG", "IA_MAG", "VA_ANG", "IA_ANG", "Freq", "ROCOF")
RLS_FORGETTINGS = (0.92, 0.985)
EPS = 1e-9


def _finite(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def _safe_stat(values: np.ndarray, fn: str, fallback: float = 0.0) -> float:
    finite = _finite(values)
    if finite.size == 0:
        return float(fallback)
    if fn == "mean":
        out = np.nanmean(finite)
    elif fn == "std":
        out = np.nanstd(finite)
    elif fn == "median":
        out = np.nanmedian(finite)
    elif fn == "max":
        out = np.nanmax(finite)
    elif fn == "min":
        out = np.nanmin(finite)
    elif fn == "p95":
        out = np.nanpercentile(finite, 95)
    elif fn == "p05":
        out = np.nanpercentile(finite, 5)
    elif fn == "max_abs":
        out = np.nanmax(np.abs(finite))
    else:
        raise ValueError(fn)
    return float(out) if np.isfinite(out) else float(fallback)


def _robust_scale(values: np.ndarray, fallback: float = 1.0) -> float:
    finite = _finite(values)
    if finite.size < 3:
        return float(fallback)
    med = float(np.nanmedian(finite))
    mad = float(np.nanmedian(np.abs(finite - med)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < EPS:
        scale = float(np.nanstd(finite))
    return float(scale) if np.isfinite(scale) and scale >= EPS else float(fallback)


def _fill(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    if np.isfinite(arr).sum() == 0:
        return np.zeros_like(arr, dtype=float)
    return pd.Series(arr).interpolate(limit_direction="both").fillna(0.0).to_numpy(dtype=float)


def _unwrap_if_needed(values: np.ndarray, signal: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if signal not in ANGLE_SIGNALS:
        return arr
    finite = np.isfinite(arr)
    if finite.sum() < 2:
        return arr
    filled = _fill(arr)
    out = np.rad2deg(np.unwrap(np.deg2rad(filled)))
    out[~finite] = np.nan
    return out


def _column_for_bus_signal(frame: pd.DataFrame, bus: int, signal: str) -> str | None:
    direct = f"BUS{bus}_{signal}"
    if direct in frame.columns:
        return direct
    matches = [col for col in frame.columns if col.upper().endswith(f"_{signal}")]
    return matches[0] if matches else None


def _dt(timeline: np.ndarray) -> float:
    finite = _finite(np.diff(np.asarray(timeline, dtype=float)))
    finite = finite[finite > 0]
    if finite.size == 0:
        return 1.0 / 30.0
    return float(np.nanmedian(finite))


def _segment_masks(timeline: np.ndarray, pre_event_seconds: float = 3.0) -> dict[str, np.ndarray]:
    t = np.asarray(timeline, dtype=float)
    if t.size == 0:
        empty = np.array([], dtype=bool)
        return {"pre": empty, "early": empty, "mid": empty, "late": empty, "full": empty}
    t0 = float(np.nanmin(t))
    t1 = float(np.nanmax(t))
    pre_end = t0 + float(pre_event_seconds)
    return {
        "pre": t <= pre_end,
        "early": (t > pre_end) & (t <= pre_end + 5.0),
        "mid": (t > pre_end + 5.0) & (t <= pre_end + 15.0),
        "late": t >= max(pre_end, t1 - 5.0),
        "full": np.ones(t.size, dtype=bool),
    }


def _zero_center(values: np.ndarray, signal: str, masks: dict[str, np.ndarray]) -> np.ndarray:
    prepared = _unwrap_if_needed(values, signal)
    pre = prepared[masks["pre"]] if masks["pre"].size == prepared.size and masks["pre"].any() else prepared
    base = _safe_stat(pre, "median", fallback=_safe_stat(prepared, "median"))
    return prepared - base


def _time_to_peak(values: np.ndarray, timeline: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).any():
        return 0.0
    idx = int(np.nanargmax(np.abs(np.nan_to_num(arr, nan=0.0))))
    t0 = float(np.nanmin(timeline)) if timeline.size else 0.0
    t1 = float(np.nanmax(timeline)) if timeline.size else 1.0
    return float((float(timeline[idx]) - t0) / max(t1 - t0, EPS))


def _slope(values: np.ndarray, dt: float) -> np.ndarray:
    arr = _fill(values)
    if arr.size < 2:
        return np.zeros_like(arr)
    return np.gradient(arr, max(float(dt), EPS))


def rolling_multiscale_features(
    signal_name: str,
    values: np.ndarray,
    timeline: np.ndarray,
    dt: float,
    windows: tuple[int, ...] = ROLLING_WINDOWS,
) -> dict[str, float]:
    out: dict[str, float] = {}
    filled = _fill(values)
    abs_values = np.abs(filled)
    for window in windows:
        roll = pd.Series(filled).rolling(window=window, center=True, min_periods=max(2, min(window // 3, window))).mean()
        roll_std = pd.Series(filled).rolling(window=window, center=True, min_periods=max(2, min(window // 3, window))).std()
        roll_values = roll.to_numpy(dtype=float)
        roll_std_values = roll_std.to_numpy(dtype=float)
        roll_slope = _slope(roll_values, dt)
        prefix = f"ROLL__{signal_name}__w{window}"
        out[f"{prefix}__mean_abs_max"] = _safe_stat(np.abs(roll_values), "max")
        out[f"{prefix}__std_max"] = _safe_stat(roll_std_values, "max")
        out[f"{prefix}__std_mean"] = _safe_stat(roll_std_values, "mean")
        out[f"{prefix}__slope_max_abs"] = _safe_stat(roll_slope, "max_abs")
        out[f"{prefix}__energy"] = float(np.nanmean(abs_values * abs_values)) if np.isfinite(abs_values).any() else 0.0
        out[f"{prefix}__time_to_abs_peak"] = _time_to_peak(roll_values, timeline)
    return out


def _rls_constant(values: np.ndarray, forgetting: float) -> tuple[np.ndarray, np.ndarray]:
    filled = _fill(values)
    theta = 0.0
    covariance = 100.0
    residual = np.zeros(filled.size, dtype=float)
    theta_delta = np.zeros(filled.size, dtype=float)
    for idx, y_t in enumerate(filled):
        gain = covariance / max(float(forgetting) + covariance, EPS)
        new_theta = theta + gain * (float(y_t) - theta)
        residual[idx] = abs(float(y_t) - theta)
        theta_delta[idx] = abs(new_theta - theta)
        covariance = (covariance - gain * covariance) / max(float(forgetting), 0.5)
        theta = new_theta
    return residual, theta_delta


def _rls_linear(values: np.ndarray, timeline: np.ndarray, forgetting: float) -> np.ndarray:
    filled = _fill(values)
    t0 = float(timeline[0]) if timeline.size else 0.0
    theta = np.zeros(2, dtype=float)
    covariance = np.eye(2, dtype=float) * 100.0
    residual = np.zeros(filled.size, dtype=float)
    for idx, y_t in enumerate(filled):
        phi = np.array([1.0, float(timeline[idx]) - t0], dtype=float)
        pred = float(phi @ theta)
        denom = float(forgetting + phi @ covariance @ phi)
        gain = (covariance @ phi) / max(denom, EPS)
        theta = theta + gain * (float(y_t) - pred)
        covariance = (covariance - np.outer(gain, phi) @ covariance) / max(float(forgetting), 0.5)
        residual[idx] = abs(float(y_t) - pred)
    return residual


def _kalman_level_innovation(values: np.ndarray, q_scale: float = 0.005, r_scale: float = 0.08) -> np.ndarray:
    filled = _fill(values)
    scale = _robust_scale(filled, fallback=max(_safe_stat(filled, "std"), 1.0))
    q = max((q_scale * scale) ** 2, EPS)
    r = max((r_scale * scale) ** 2, EPS)
    x = float(filled[0]) if filled.size else 0.0
    p = scale * scale
    innov = np.zeros(filled.size, dtype=float)
    for idx, y_t in enumerate(filled):
        p = p + q
        innovation = float(y_t) - x
        k = p / max(p + r, EPS)
        x = x + k * innovation
        p = (1.0 - k) * p
        innov[idx] = abs(innovation)
    return innov


def rls_kalman_features(
    signal_name: str,
    values: np.ndarray,
    timeline: np.ndarray,
    masks: dict[str, np.ndarray],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for forgetting in RLS_FORGETTINGS:
        tag = f"f{int(round(forgetting * 1000)):03d}"
        resid, theta_delta = _rls_constant(values, forgetting)
        linear_resid = _rls_linear(values, timeline, forgetting)
        for name, arr in (("level_resid", resid), ("theta_delta", theta_delta), ("linear_resid", linear_resid)):
            prefix = f"RLSK__{signal_name}__{tag}__{name}"
            out[f"{prefix}__max"] = _safe_stat(arr, "max")
            out[f"{prefix}__mean"] = _safe_stat(arr, "mean")
            out[f"{prefix}__p95"] = _safe_stat(arr, "p95")
            late = arr[masks["late"]] if masks["late"].size == arr.size and masks["late"].any() else arr
            pre = arr[masks["pre"]] if masks["pre"].size == arr.size and masks["pre"].any() else arr
            out[f"{prefix}__late_pre_delta"] = _safe_stat(late, "median") - _safe_stat(pre, "median")
    innov = _kalman_level_innovation(values)
    prefix = f"RLSK__{signal_name}__kalman_innov"
    out[f"{prefix}__max"] = _safe_stat(innov, "max")
    out[f"{prefix}__mean"] = _safe_stat(innov, "mean")
    out[f"{prefix}__p95"] = _safe_stat(innov, "p95")
    out[f"{prefix}__time_to_peak"] = _time_to_peak(innov, timeline)
    return out


def hilbert_features(signal_name: str, values: np.ndarray, timeline: np.ndarray, dt: float, masks: dict[str, np.ndarray]) -> dict[str, float]:
    out: dict[str, float] = {}
    filled = _fill(values)
    if filled.size < 4 or np.nanstd(filled) < EPS:
        analytic = filled.astype(complex)
    else:
        analytic = hilbert(filled)
    envelope = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic))
    inst_freq = _slope(phase, dt) / (2.0 * np.pi)
    for name, arr in (("envelope", envelope), ("inst_freq", inst_freq)):
        prefix = f"HILB__{signal_name}__{name}"
        out[f"{prefix}__max"] = _safe_stat(arr, "max")
        out[f"{prefix}__mean"] = _safe_stat(arr, "mean")
        out[f"{prefix}__std"] = _safe_stat(arr, "std")
        out[f"{prefix}__p95"] = _safe_stat(arr, "p95")
        out[f"{prefix}__max_abs"] = _safe_stat(arr, "max_abs")
        late = arr[masks["late"]] if masks["late"].size == arr.size and masks["late"].any() else arr
        pre = arr[masks["pre"]] if masks["pre"].size == arr.size and masks["pre"].any() else arr
        out[f"{prefix}__late_pre_delta"] = _safe_stat(late, "median") - _safe_stat(pre, "median")
    out[f"HILB__{signal_name}__phase_span"] = _safe_stat(phase, "max") - _safe_stat(phase, "min")
    out[f"HILB__{signal_name}__envelope_time_to_peak"] = _time_to_peak(envelope, timeline)
    return out


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    x = _fill(a)
    y = _fill(b)
    if x.size < 3 or y.size < 3:
        return 0.0
    if np.nanstd(x) < EPS or np.nanstd(y) < EPS:
        return 0.0
    value = float(np.corrcoef(x, y)[0, 1])
    return value if np.isfinite(value) else 0.0


def _lag_at_max_corr(a: np.ndarray, b: np.ndarray, max_lag: int = 60) -> float:
    x = _fill(a) - np.nanmean(_fill(a))
    y = _fill(b) - np.nanmean(_fill(b))
    if x.size < 3 or y.size < 3:
        return 0.0
    max_lag = int(min(max_lag, max(1, x.size // 3)))
    best_lag = 0
    best_score = -np.inf
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            xx, yy = x[-lag:], y[: lag or None]
        elif lag > 0:
            xx, yy = x[:-lag], y[lag:]
        else:
            xx, yy = x, y
        if xx.size < 3 or yy.size < 3 or np.nanstd(xx) < EPS or np.nanstd(yy) < EPS:
            continue
        score = abs(float(np.corrcoef(xx, yy)[0, 1]))
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_lag = lag
    return float(best_lag)


def graph_temporal_features(
    centered_by_bus: dict[int, dict[str, np.ndarray]],
    timeline: np.ndarray,
    topology_distances: pd.DataFrame | None = None,
    observed_pmus: tuple[int, ...] | None = None,
) -> dict[str, float]:
    out: dict[str, float] = {}
    pmu_buses = tuple(sorted(int(bus) for bus in (observed_pmus or centered_by_bus.keys())))
    if topology_distances is not None and not topology_distances.empty:
        dist_lookup = {
            (int(row.from_bus), int(row.to_bus)): float(row.z_eff_abs)
            for row in topology_distances.itertuples(index=False)
            if int(row.from_bus) in pmu_buses and int(row.to_bus) in pmu_buses
        }
    else:
        dist_lookup = {}
    for signal in GRAPH_SIGNALS:
        series = {bus: centered_by_bus[bus][signal] for bus in pmu_buses if bus in centered_by_bus and signal in centered_by_bus[bus]}
        if len(series) < 2:
            continue
        buses = [bus for bus in pmu_buses if bus in series]
        min_len = min(int(np.asarray(series[bus]).size) for bus in buses)
        if min_len < 2:
            continue
        aligned_timeline = np.asarray(timeline, dtype=float)[:min_len]
        series = {bus: np.asarray(series[bus], dtype=float)[:min_len] for bus in buses}
        mat = np.column_stack([_fill(series[bus]) for bus in buses])
        if mat.size == 0:
            continue
        spread = np.nanmax(mat, axis=1) - np.nanmin(mat, axis=1)
        prefix = f"GRAPH__{signal}"
        out[f"{prefix}__pmu_spread_max"] = _safe_stat(spread, "max_abs")
        out[f"{prefix}__pmu_spread_mean"] = _safe_stat(np.abs(spread), "mean")
        out[f"{prefix}__pmu_spread_p95"] = _safe_stat(np.abs(spread), "p95")
        peak_times = np.asarray([_time_to_peak(series[bus], aligned_timeline) for bus in buses], dtype=float)
        out[f"{prefix}__time_to_peak_range"] = _safe_stat(peak_times, "max") - _safe_stat(peak_times, "min")
        out[f"{prefix}__time_to_peak_std"] = _safe_stat(peak_times, "std")
        graph_energy = []
        for a, b in combinations(buses, 2):
            za = dist_lookup.get((a, b), dist_lookup.get((b, a), 1.0))
            weight = 1.0 / max(float(za), 1e-5)
            diff = series[a] - series[b]
            graph_energy.append(weight * _safe_stat(diff, "max_abs"))
            pair = f"{prefix}__pair_{a}_{b}"
            out[f"{pair}__diff_max_abs"] = _safe_stat(diff, "max_abs")
            out[f"{pair}__diff_p95"] = _safe_stat(np.abs(diff), "p95")
            out[f"{pair}__corr"] = _corr(series[a], series[b])
            out[f"{pair}__time_to_peak_diff"] = abs(
                _time_to_peak(series[a], aligned_timeline) - _time_to_peak(series[b], aligned_timeline)
            )
        out[f"{prefix}__weighted_graph_diff_max"] = _safe_stat(np.asarray(graph_energy), "max")
        out[f"{prefix}__weighted_graph_diff_mean"] = _safe_stat(np.asarray(graph_energy), "mean")
    return out


def residual_topology_features(
    centered_by_bus: dict[int, dict[str, np.ndarray]],
    topology_distances: pd.DataFrame | None = None,
    observed_pmus: tuple[int, ...] | None = None,
) -> dict[str, float]:
    out: dict[str, float] = {}
    if topology_distances is None or topology_distances.empty or not centered_by_bus:
        return out
    pmu_buses = tuple(sorted(int(bus) for bus in (observed_pmus or centered_by_bus.keys())))
    buses = sorted({int(bus) for bus in topology_distances["to_bus"].dropna().unique().tolist()})
    dist_lookup = {
        (int(row.from_bus), int(row.to_bus)): max(float(row.z_eff_abs), EPS)
        for row in topology_distances.itertuples(index=False)
    }
    observed: dict[int, float] = {}
    for pmu in pmu_buses:
        signals = centered_by_bus.get(pmu, {})
        values = [
            _safe_stat(np.asarray(signals[signal], dtype=float), "max_abs")
            for signal in RESIDUAL_TOPOLOGY_SIGNALS
            if signal in signals
        ]
        observed[pmu] = _safe_stat(np.asarray(values, dtype=float), "max") if values else 0.0
    obs_vec = np.asarray([observed.get(pmu, 0.0) for pmu in pmu_buses], dtype=float)
    obs_norm = float(np.linalg.norm(obs_vec))
    best_bus = 0
    best_score = -1.0
    scores = []
    for bus in buses:
        expected = np.asarray(
            [1.0 / max(dist_lookup.get((pmu, bus), dist_lookup.get((bus, pmu), 1.0)), EPS) for pmu in pmu_buses],
            dtype=float,
        )
        exp_norm = float(np.linalg.norm(expected))
        score = float(np.dot(obs_vec, expected) / (obs_norm * exp_norm)) if obs_norm > EPS and exp_norm > EPS else 0.0
        score = score if math.isfinite(score) else 0.0
        out[f"RESID__BUS{bus}__influence_cosine"] = score
        out[f"RESID__BUS{bus}__weighted_observed_energy"] = float(np.dot(obs_vec, expected) / max(float(np.sum(expected)), EPS))
        scores.append(score)
        if score > best_score:
            best_score = score
            best_bus = int(bus)
    score_arr = np.asarray(scores, dtype=float)
    out["RESID__grid__candidate_score_max"] = _safe_stat(score_arr, "max")
    out["RESID__grid__candidate_score_mean"] = _safe_stat(score_arr, "mean")
    out["RESID__grid__candidate_score_std"] = _safe_stat(score_arr, "std")
    out["RESID__grid__best_bus"] = float(best_bus)
    return out


def data_quality_features(frames_by_bus: dict[int, pd.DataFrame], observed_pmus: tuple[int, ...] | None = None) -> dict[str, float]:
    out: dict[str, float] = {}
    pmu_buses = tuple(sorted(int(bus) for bus in (observed_pmus or infer_pmu_buses_from_frames(frames_by_bus) or LEGACY_PMU_BUSES)))
    present_values: list[float] = []
    nan_values: list[float] = []
    gap_values: list[float] = []
    for bus in pmu_buses:
        frame = frames_by_bus.get(bus)
        if frame is None or frame.empty:
            out[f"META__BUS{bus}__data_present_fraction"] = 0.0
            out[f"META__BUS{bus}__nan_fraction_max"] = 1.0
            out[f"META__BUS{bus}__max_timestamp_gap_ratio"] = 999.0
            continue
        if "DATA_PRESENT" in frame.columns:
            present = pd.to_numeric(frame["DATA_PRESENT"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            present_fraction = float(np.nanmean(present > 0.5)) if present.size else 0.0
        else:
            present_fraction = 1.0
        timeline = pd.to_numeric(frame["TIMESTAMP"], errors="coerce").to_numpy(dtype=float)
        diffs = _finite(np.diff(timeline))
        diffs = diffs[diffs > 0]
        if diffs.size:
            gap_ratio = float(np.nanmax(diffs) / max(float(np.nanmedian(diffs)), EPS))
        else:
            gap_ratio = 999.0
        signal_nan_fractions = []
        for signal in SIGNALS:
            col = _column_for_bus_signal(frame, bus, signal)
            if col is None:
                continue
            values = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
            signal_nan_fractions.append(float(np.isnan(values).mean()) if values.size else 1.0)
        nan_fraction_max = max(signal_nan_fractions) if signal_nan_fractions else 1.0
        out[f"META__BUS{bus}__data_present_fraction"] = present_fraction
        out[f"META__BUS{bus}__nan_fraction_max"] = nan_fraction_max
        out[f"META__BUS{bus}__max_timestamp_gap_ratio"] = gap_ratio
        present_values.append(present_fraction)
        nan_values.append(nan_fraction_max)
        gap_values.append(gap_ratio)
    out["META__GLOBAL__data_present_fraction_min"] = _safe_stat(np.asarray(present_values), "min", fallback=0.0)
    out["META__GLOBAL__data_present_fraction_mean"] = _safe_stat(np.asarray(present_values), "mean", fallback=0.0)
    out["META__GLOBAL__nan_fraction_max"] = _safe_stat(np.asarray(nan_values), "max", fallback=1.0)
    out["META__GLOBAL__max_timestamp_gap_ratio"] = _safe_stat(np.asarray(gap_values), "max", fallback=999.0)
    return out


def extract_dynamic_features_from_frames(
    frames_by_bus: dict[int, pd.DataFrame],
    topology_distances: pd.DataFrame | None = None,
    include_blocks: tuple[str, ...] = ("rolling", "rls_kalman", "hilbert", "graph_temporal"),
    pre_event_seconds: float = 3.0,
    observed_pmus: tuple[int, ...] | None = None,
) -> dict[str, float]:
    pmu_buses = tuple(sorted(int(bus) for bus in (observed_pmus or infer_pmu_buses_from_frames(frames_by_bus) or LEGACY_PMU_BUSES)))
    features: dict[str, float] = data_quality_features(frames_by_bus, pmu_buses)
    centered_by_bus: dict[int, dict[str, np.ndarray]] = {}
    for bus in pmu_buses:
        frame = frames_by_bus.get(bus)
        if frame is None or frame.empty:
            continue
        timeline = pd.to_numeric(frame["TIMESTAMP"], errors="coerce").to_numpy(dtype=float)
        step = _dt(timeline)
        masks = _segment_masks(timeline, pre_event_seconds=pre_event_seconds)
        centered_by_bus[bus] = {}
        for signal in SIGNALS:
            col = _column_for_bus_signal(frame, bus, signal)
            if col is None:
                continue
            raw = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
            centered = _zero_center(raw, signal, masks)
            centered_by_bus[bus][signal] = centered
            signal_key = f"BUS{bus}__{signal}"
            if "rolling" in include_blocks and signal in ROLLING_SIGNALS:
                for key, value in rolling_multiscale_features(signal_key, centered, timeline, step).items():
                    features[key] = value
            if "rls_kalman" in include_blocks and signal in RLS_SIGNALS:
                for key, value in rls_kalman_features(signal_key, centered, timeline, masks).items():
                    features[key] = value
            if "hilbert" in include_blocks and signal in HILBERT_SIGNALS:
                for key, value in hilbert_features(signal_key, centered, timeline, step, masks).items():
                    features[key] = value
    if "graph_temporal" in include_blocks and centered_by_bus:
        features.update(
            graph_temporal_features(
                centered_by_bus,
                next(iter(frames_by_bus.values()))["TIMESTAMP"].to_numpy(dtype=float),
                topology_distances,
                pmu_buses,
            )
        )
    if "residual_topology" in include_blocks and centered_by_bus:
        features.update(residual_topology_features(centered_by_bus, topology_distances, pmu_buses))
    clean: dict[str, float] = {}
    for key, value in features.items():
        value = float(value)
        clean[key] = value if math.isfinite(value) else 0.0
    return clean


def load_pmu_frames(pmu_dir: Path) -> dict[int, pd.DataFrame]:
    frames: dict[int, pd.DataFrame] = {}
    for bus, path in discover_bus_csvs(pmu_dir).items():
        frames[bus] = pd.read_csv(path)
    return frames
