from __future__ import annotations

import math
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PHASES = ("A", "B", "C")
VOLTAGE_MAG = tuple(f"V{p}_MAG" for p in PHASES)
CURRENT_MAG = tuple(f"I{p}_MAG" for p in PHASES)
VOLTAGE_ANG = tuple(f"V{p}_ANG" for p in PHASES)
CURRENT_ANG = tuple(f"I{p}_ANG" for p in PHASES)
SIGNALS = VOLTAGE_MAG + CURRENT_MAG + VOLTAGE_ANG + CURRENT_ANG + ("Freq", "ROCOF")
ANGLE_SIGNALS = VOLTAGE_ANG + CURRENT_ANG
EPS = 1e-9


def finite(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def safe_stat(values: np.ndarray, name: str, fallback: float = 0.0) -> float:
    arr = finite(values)
    if arr.size == 0:
        return float(fallback)
    if name == "mean":
        value = np.nanmean(arr)
    elif name == "std":
        value = np.nanstd(arr)
    elif name == "median":
        value = np.nanmedian(arr)
    elif name == "min":
        value = np.nanmin(arr)
    elif name == "max":
        value = np.nanmax(arr)
    elif name == "max_abs":
        value = np.nanmax(np.abs(arr))
    elif name == "mean_abs":
        value = np.nanmean(np.abs(arr))
    elif name == "p05":
        value = np.nanpercentile(arr, 5)
    elif name == "p95":
        value = np.nanpercentile(arr, 95)
    else:
        raise ValueError(name)
    return float(value) if np.isfinite(value) else float(fallback)


def robust_center_scale(values: np.ndarray) -> tuple[float, float]:
    arr = finite(values)
    if arr.size == 0:
        return 0.0, 1.0
    center = float(np.nanmedian(arr))
    mad = float(np.nanmedian(np.abs(arr - center)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < EPS:
        scale = float(np.nanstd(arr)) if arr.size > 1 else 1.0
    if not np.isfinite(scale) or scale < EPS:
        scale = max(abs(center) * 0.002, 1.0)
    return center, scale


def fill_signal(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    if np.isfinite(arr).sum() == 0:
        return np.zeros(arr.size, dtype=float)
    return pd.Series(arr).interpolate(limit_direction="both").bfill().ffill().to_numpy(dtype=float)


def unwrap_degrees(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    mask = np.isfinite(arr)
    if mask.sum() < 2:
        return arr.copy()
    filled = fill_signal(arr)
    out = np.rad2deg(np.unwrap(np.deg2rad(filled)))
    out[~mask] = np.nan
    return out


def dt_from_timestamps(timestamps: np.ndarray) -> float:
    diffs = finite(np.diff(np.asarray(timestamps, dtype=float)))
    diffs = diffs[diffs > 0]
    return float(np.nanmedian(diffs)) if diffs.size else 1.0 / 30.0


def segment_masks(timestamps: np.ndarray, pre_seconds: float = 3.0) -> dict[str, np.ndarray]:
    t = np.asarray(timestamps, dtype=float)
    if t.size == 0:
        empty = np.array([], dtype=bool)
        return {name: empty for name in ("pre", "early", "mid", "late", "full")}
    start = float(np.nanmin(t))
    end = float(np.nanmax(t))
    pre_end = start + float(pre_seconds)
    return {
        "pre": t <= pre_end,
        "early": (t > pre_end) & (t <= pre_end + 5.0),
        "mid": (t > pre_end + 5.0) & (t <= pre_end + 15.0),
        "late": t >= max(pre_end, end - 5.0),
        "full": np.ones(t.size, dtype=bool),
    }


def signal_column(frame: pd.DataFrame, bus: int, signal: str) -> str | None:
    direct = f"BUS{bus}_{signal}"
    if direct in frame:
        return direct
    suffix = f"_{signal}".upper()
    matches = [col for col in frame.columns if col.upper().endswith(suffix)]
    if matches:
        return matches[0]
    return signal if signal in frame else None


def numeric_signal(frame: pd.DataFrame, bus: int, signal: str) -> np.ndarray:
    col = signal_column(frame, bus, signal)
    if col is None:
        return np.full(len(frame), np.nan, dtype=float)
    values = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
    return unwrap_degrees(values) if signal in ANGLE_SIGNALS else values


def baseline_relative(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    base_segment = arr[mask] if mask.size == arr.size and mask.any() else arr
    center = safe_stat(base_segment, "median", fallback=safe_stat(arr, "median"))
    return arr - center


def robust_z(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    base_segment = arr[mask] if mask.size == arr.size and mask.any() else arr
    center, scale = robust_center_scale(base_segment)
    return (arr - center) / max(scale, EPS)


def derivative(values: np.ndarray, dt: float) -> np.ndarray:
    arr = fill_signal(values)
    if arr.size < 2:
        return np.zeros(arr.size, dtype=float)
    return np.gradient(arr, max(float(dt), EPS))


def rolling_max_mean_abs(values: np.ndarray, window: int) -> float:
    arr = np.abs(fill_signal(values))
    if arr.size == 0:
        return 0.0
    rolled = pd.Series(arr).rolling(window=window, center=True, min_periods=1).mean()
    return safe_stat(rolled.to_numpy(dtype=float), "max")


def signal_features(name: str, values: np.ndarray, timestamps: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, float]:
    out: dict[str, float] = {}
    centered = baseline_relative(values, masks["pre"])
    z = np.abs(robust_z(values, masks["pre"]))
    dx = derivative(centered, dt_from_timestamps(timestamps))
    for segment, mask in masks.items():
        local = centered[mask] if mask.size == centered.size else centered
        prefix = f"{name}__{segment}"
        out[f"{prefix}__mean"] = safe_stat(local, "mean")
        out[f"{prefix}__std"] = safe_stat(local, "std")
        out[f"{prefix}__span"] = safe_stat(local, "max") - safe_stat(local, "min")
        out[f"{prefix}__max_abs"] = safe_stat(local, "max_abs")
        out[f"{prefix}__p95"] = safe_stat(local, "p95")
    out[f"{name}__early_pre_delta"] = out[f"{name}__early__mean"] - out[f"{name}__pre__mean"]
    out[f"{name}__mid_pre_delta"] = out[f"{name}__mid__mean"] - out[f"{name}__pre__mean"]
    out[f"{name}__late_pre_delta"] = out[f"{name}__late__mean"] - out[f"{name}__pre__mean"]
    out[f"{name}__max_abs_derivative"] = safe_stat(dx, "max_abs")
    out[f"{name}__rms_derivative"] = float(np.sqrt(np.nanmean(dx * dx))) if dx.size else 0.0
    out[f"{name}__max_abs_robust_z"] = safe_stat(z, "max")
    out[f"{name}__sustained_abs_robust_z_5"] = rolling_max_mean_abs(z, 5)
    out[f"{name}__sustained_abs_robust_z_15"] = rolling_max_mean_abs(z, 15)
    out[f"{name}__energy"] = float(np.nanmean(fill_signal(centered) ** 2)) if centered.size else 0.0
    return {key: float(value) if math.isfinite(float(value)) else 0.0 for key, value in out.items()}


def sequence_components(phasors: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.exp(1j * 2.0 * np.pi / 3.0)
    xa = phasors["A"]
    xb = phasors["B"]
    xc = phasors["C"]
    x0 = (xa + xb + xc) / 3.0
    x1 = (xa + a * xb + (a**2) * xc) / 3.0
    x2 = (xa + (a**2) * xb + a * xc) / 3.0
    return x0, x1, x2


def phase_phasors(frame: pd.DataFrame, bus: int, kind: str) -> dict[str, np.ndarray] | None:
    out: dict[str, np.ndarray] = {}
    for phase in PHASES:
        mag = numeric_signal(frame, bus, f"{kind}{phase}_MAG")
        ang = numeric_signal(frame, bus, f"{kind}{phase}_ANG")
        if np.isfinite(mag).sum() == 0 or np.isfinite(ang).sum() == 0:
            return None
        out[phase] = fill_signal(mag) * np.exp(1j * np.deg2rad(fill_signal(ang)))
    return out


def three_phase_features(frame: pd.DataFrame, bus: int, timestamps: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, float]:
    out: dict[str, float] = {}
    for prefix, group in (("V_MAG_SPREAD", VOLTAGE_MAG), ("I_MAG_SPREAD", CURRENT_MAG), ("V_ANG_SPREAD", VOLTAGE_ANG), ("I_ANG_SPREAD", CURRENT_ANG)):
        values = np.vstack([numeric_signal(frame, bus, signal) for signal in group])
        if not np.isfinite(values).any():
            spread = np.zeros(values.shape[1], dtype=float)
        else:
            with np.errstate(all="ignore"):
                spread = np.nanmax(values, axis=0) - np.nanmin(values, axis=0)
            spread[~np.isfinite(spread)] = 0.0
        out.update(signal_features(prefix, spread, timestamps, masks))
    for kind in ("V", "I"):
        phasors = phase_phasors(frame, bus, kind)
        if phasors is None:
            continue
        x0, x1, x2 = sequence_components(phasors)
        pos = np.abs(x1)
        neg_ratio = np.abs(x2) / (np.abs(x1) + EPS)
        zero_ratio = np.abs(x0) / (np.abs(x1) + EPS)
        out.update(signal_features(f"{kind}_POS_SEQ_MAG", pos, timestamps, masks))
        out.update(signal_features(f"{kind}_NEG_SEQ_RATIO", neg_ratio, timestamps, masks))
        out.update(signal_features(f"{kind}_ZERO_SEQ_RATIO", zero_ratio, timestamps, masks))
    return out


def power_proxy_features(frame: pd.DataFrame, bus: int, timestamps: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, float]:
    p_total = np.zeros(len(frame), dtype=float)
    q_total = np.zeros(len(frame), dtype=float)
    valid = False
    for phase in PHASES:
        vmag = numeric_signal(frame, bus, f"V{phase}_MAG")
        imag = numeric_signal(frame, bus, f"I{phase}_MAG")
        vang = numeric_signal(frame, bus, f"V{phase}_ANG")
        iang = numeric_signal(frame, bus, f"I{phase}_ANG")
        if np.isfinite(vmag).sum() == 0 or np.isfinite(imag).sum() == 0:
            continue
        delta = np.deg2rad(fill_signal(vang) - fill_signal(iang))
        p_total += fill_signal(vmag) * fill_signal(imag) * np.cos(delta)
        q_total += fill_signal(vmag) * fill_signal(imag) * np.sin(delta)
        valid = True
    if not valid:
        return {}
    out = signal_features("P_PROXY", p_total, timestamps, masks)
    out.update(signal_features("Q_PROXY", q_total, timestamps, masks))
    out.update(signal_features("PF_ANGLE", np.rad2deg(np.arctan2(q_total, p_total + EPS)), timestamps, masks))
    return out


def data_quality_features(frame: pd.DataFrame) -> dict[str, float]:
    if "DATA_PRESENT" in frame:
        present = pd.to_numeric(frame["DATA_PRESENT"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    else:
        present = np.ones(len(frame), dtype=float)
    measurement_cols = [col for col in frame.columns if col not in {"TIMESTAMP", "DATA_PRESENT", "Event"}]
    nan_fraction = frame[measurement_cols].isna().mean(axis=1).to_numpy(dtype=float) if measurement_cols else np.ones(len(frame))
    missing = present < 0.5
    longest = current = 0
    for value in missing:
        current = current + 1 if bool(value) else 0
        longest = max(longest, current)
    timestamps = pd.to_numeric(frame["TIMESTAMP"], errors="coerce").to_numpy(dtype=float) if "TIMESTAMP" in frame else np.arange(len(frame))
    dt = dt_from_timestamps(timestamps)
    diffs = finite(np.diff(timestamps))
    gap_ratio = float(np.nanmax(diffs) / max(dt, EPS)) if diffs.size else 1.0
    return {
        "DQ__data_present_fraction": float(np.nanmean(present > 0.5)) if present.size else 0.0,
        "DQ__nan_fraction_max": safe_stat(nan_fraction, "max", fallback=1.0),
        "DQ__nan_fraction_mean": safe_stat(nan_fraction, "mean", fallback=1.0),
        "DQ__missing_run_max_samples": float(longest),
        "DQ__max_timestamp_gap_ratio": gap_ratio,
    }


def pmu_window_features(frame: pd.DataFrame, bus: int, pre_seconds: float = 3.0) -> dict[str, float]:
    timestamps = pd.to_numeric(frame["TIMESTAMP"], errors="coerce").to_numpy(dtype=float)
    masks = segment_masks(timestamps, pre_seconds)
    out = {"bus_id": float(bus), "n_samples": float(len(frame)), "duration_s": safe_stat(timestamps, "max") - safe_stat(timestamps, "min")}
    out.update(data_quality_features(frame))
    for signal in SIGNALS:
        out.update(signal_features(signal, numeric_signal(frame, bus, signal), timestamps, masks))
    out.update(three_phase_features(frame, bus, timestamps, masks))
    out.update(power_proxy_features(frame, bus, timestamps, masks))
    return {key: float(value) if math.isfinite(float(value)) else 0.0 for key, value in out.items()}


def aggregate_pmu_features(rows: dict[int, dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows.values() for key in row if key != "bus_id"})
    out: dict[str, float] = {"n_pmus": float(len(rows))}
    for key in keys:
        values = np.asarray([row.get(key, 0.0) for row in rows.values()], dtype=float)
        out[f"AGG__{key}__mean"] = safe_stat(values, "mean")
        out[f"AGG__{key}__max"] = safe_stat(values, "max")
        out[f"AGG__{key}__min"] = safe_stat(values, "min")
        out[f"AGG__{key}__std"] = safe_stat(values, "std")
    severity = pmu_severity_vector(rows)
    if severity:
        buses = np.asarray(list(severity.keys()), dtype=float)
        vals = np.asarray(list(severity.values()), dtype=float)
        idx = int(np.nanargmax(vals))
        out["GLOB__top_pmu_bus"] = float(buses[idx])
        out["GLOB__severity_max"] = safe_stat(vals, "max")
        out["GLOB__severity_entropy"] = entropy(vals)
        sorted_vals = np.sort(vals)[::-1]
        out["GLOB__severity_margin"] = float(sorted_vals[0] - sorted_vals[1]) if sorted_vals.size > 1 else float(sorted_vals[0])
    return out


def pmu_severity_vector(rows: dict[int, dict[str, float]]) -> dict[int, float]:
    out: dict[int, float] = {}
    for bus, row in rows.items():
        values = [
            row.get("VA_MAG__max_abs_robust_z", 0.0),
            row.get("VB_MAG__max_abs_robust_z", 0.0),
            row.get("VC_MAG__max_abs_robust_z", 0.0),
            row.get("IA_MAG__max_abs_robust_z", 0.0),
            row.get("IB_MAG__max_abs_robust_z", 0.0),
            row.get("IC_MAG__max_abs_robust_z", 0.0),
            row.get("Freq__max_abs_robust_z", 0.0),
            min(row.get("ROCOF__full__max_abs", 0.0) / 0.02, 50.0),
            row.get("DQ__nan_fraction_max", 0.0) * 30.0,
            (1.0 - row.get("DQ__data_present_fraction", 1.0)) * 40.0,
        ]
        out[int(bus)] = float(np.nanmax(np.asarray(values, dtype=float)))
    return out


def entropy(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = np.maximum(arr, 0.0)
    total = float(np.sum(arr))
    if total <= EPS:
        return 0.0
    p = arr / total
    return float(-np.sum(p * np.log(p + EPS)))


def load_zbus_distances(topology_dir: Path) -> pd.DataFrame:
    path = Path(topology_dir) / "zbus_effective_distance_full.csv"
    if not path.exists():
        return pd.DataFrame(columns=["from_bus", "to_bus", "z_eff_abs"])
    return pd.read_csv(path)


def distance_lookup(distances: pd.DataFrame) -> dict[tuple[int, int], float]:
    lookup: dict[tuple[int, int], float] = {}
    if distances.empty:
        return lookup
    for row in distances.itertuples(index=False):
        a = int(getattr(row, "from_bus"))
        b = int(getattr(row, "to_bus"))
        value = max(float(getattr(row, "z_eff_abs")), EPS)
        lookup[(a, b)] = value
        lookup[(b, a)] = value
    return lookup


def graph_total_variation(severity: dict[int, float], distances: pd.DataFrame) -> float:
    lookup = distance_lookup(distances)
    buses = sorted(severity)
    if len(buses) < 2:
        return 0.0
    value = 0.0
    denom = 0.0
    for a, b in combinations(buses, 2):
        weight = 1.0 / max(lookup.get((a, b), 1.0), EPS)
        value += weight * (float(severity[a]) - float(severity[b])) ** 2
        denom += weight
    return float(value / max(denom, EPS))


def zbus_diffusion_scores(
    severity: dict[int, float],
    distances: pd.DataFrame,
    candidate_buses: list[int],
    taus: tuple[float, ...] = (0.05, 0.10, 0.20, 0.40, 0.80),
) -> dict[str, float]:
    if not severity or distances.empty:
        return {}
    lookup = distance_lookup(distances)
    observed_buses = sorted(severity)
    x = np.asarray([severity[bus] for bus in observed_buses], dtype=float)
    x = x / max(float(np.linalg.norm(x)), EPS)
    out: dict[str, float] = {}
    for tau in taus:
        scores = []
        for candidate in candidate_buses:
            k = np.asarray([math.exp(-lookup.get((bus, int(candidate)), 1.0) / max(float(tau), EPS)) for bus in observed_buses], dtype=float)
            k = k / max(float(np.linalg.norm(k)), EPS)
            score = float(np.dot(x, k))
            out[f"GSP__BUS{int(candidate)}__tau{tau:g}__cosine"] = score
            out[f"GSP__BUS{int(candidate)}__tau{tau:g}__residual"] = float(np.linalg.norm(x - k))
            scores.append(score)
        arr = np.asarray(scores, dtype=float)
        ordered = np.sort(arr)[::-1]
        out[f"GSP__grid__tau{tau:g}__top1_score"] = safe_stat(arr, "max")
        out[f"GSP__grid__tau{tau:g}__top2_margin"] = float(ordered[0] - ordered[1]) if ordered.size > 1 else safe_stat(arr, "max")
        out[f"GSP__grid__tau{tau:g}__score_entropy"] = entropy(arr - safe_stat(arr, "min"))
        out[f"GSP__grid__tau{tau:g}__best_bus"] = float(candidate_buses[int(np.nanargmax(arr))]) if arr.size else 0.0
    return out


def line_diffusion_scores(
    severity: dict[int, float],
    distances: pd.DataFrame,
    lines: list[tuple[int, int]],
    tau: float = 0.20,
) -> dict[str, float]:
    if not severity or distances.empty:
        return {}
    lookup = distance_lookup(distances)
    observed_buses = sorted(severity)
    x = np.asarray([severity[bus] for bus in observed_buses], dtype=float)
    x = x / max(float(np.linalg.norm(x)), EPS)
    out: dict[str, float] = {}
    scores = []
    for a, b in lines:
        ka = np.asarray([math.exp(-lookup.get((bus, int(a)), 1.0) / max(float(tau), EPS)) for bus in observed_buses], dtype=float)
        kb = np.asarray([math.exp(-lookup.get((bus, int(b)), 1.0) / max(float(tau), EPS)) for bus in observed_buses], dtype=float)
        k = np.maximum(ka, kb)
        k = k / max(float(np.linalg.norm(k)), EPS)
        score = float(np.dot(x, k))
        aa, bb = min(int(a), int(b)), max(int(a), int(b))
        out[f"GSP__LINE{aa}-{bb}__cosine"] = score
        scores.append(score)
    arr = np.asarray(scores, dtype=float)
    ordered = np.sort(arr)[::-1]
    out["GSP__line_grid__top1_score"] = safe_stat(arr, "max")
    out["GSP__line_grid__top2_margin"] = float(ordered[0] - ordered[1]) if ordered.size > 1 else safe_stat(arr, "max")
    return out


def graph_feature_block(
    rows: dict[int, dict[str, float]],
    distances: pd.DataFrame,
    candidate_buses: list[int] | None = None,
    lines: list[tuple[int, int]] | None = None,
) -> dict[str, float]:
    severity = pmu_severity_vector(rows)
    candidate_buses = candidate_buses or list(range(1, 40))
    lines = lines or []
    out = {"GSP__severity_graph_total_variation": graph_total_variation(severity, distances)}
    out.update(zbus_diffusion_scores(severity, distances, candidate_buses))
    out.update(line_diffusion_scores(severity, distances, lines))
    return out
