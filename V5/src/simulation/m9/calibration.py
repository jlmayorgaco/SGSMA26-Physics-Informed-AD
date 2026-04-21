"""RAW0001 reference-statistics extraction and RAW-vs-SIM comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import math
import re

import numpy as np
import pandas as pd

from src.simulation.m9.constants import PMU_MEASUREMENT_SUFFIXES


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def bus_token_from_path(path: Path) -> str:
    m = re.search(r"Bus(\d+)", path.name, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot identify bus token from {path}")
    return m.group(1)


def measurement_columns_for_bus(bus_token: str) -> list[str]:
    return [f"BUS{int(bus_token)}_{suffix}" for suffix in PMU_MEASUREMENT_SUFFIXES]


def _series_stats(s: pd.Series) -> dict[str, Any]:
    x = pd.to_numeric(s, errors="coerce")
    clean = x.dropna()
    if clean.empty:
        return {"count": 0, "mean": None, "std": None, "median": None, "iqr": None, "min": None, "max": None, "increment_std": None, "nan_fraction": float(x.isna().mean())}
    diff = clean.diff().dropna()
    q75 = float(clean.quantile(0.75))
    q25 = float(clean.quantile(0.25))
    return {
        "count": int(clean.size),
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=0)),
        "median": float(clean.median()),
        "iqr": float(q75 - q25),
        "min": float(clean.min()),
        "max": float(clean.max()),
        "increment_std": float(diff.std(ddof=0)) if not diff.empty else 0.0,
        "nan_fraction": float(x.isna().mean()),
    }


def _nan_bursts(mask: pd.Series) -> dict[str, Any]:
    vals = mask.fillna(False).astype(bool).to_numpy()
    bursts: list[int] = []
    current = 0
    for flag in vals:
        if flag:
            current += 1
        elif current:
            bursts.append(current)
            current = 0
    if current:
        bursts.append(current)
    return {
        "burst_count": len(bursts),
        "max_burst_frames": int(max(bursts) if bursts else 0),
        "mean_burst_frames": float(np.mean(bursts)) if bursts else 0.0,
    }


def _read_csv_sample(path: Path, max_rows: int | None = 20000) -> pd.DataFrame:
    try:
        if max_rows and max_rows > 0:
            return pd.read_csv(path, nrows=int(max_rows))
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def extract_reference_statistics(reference_pmu_dir: str | Path, max_rows: int | None = 20000) -> dict[str, Any]:
    root = Path(reference_pmu_dir)
    files = sorted(root.glob("Bus*_Competition_Data*.csv"))
    if not files:
        return _fallback_reference_statistics(root)
    buses: dict[str, Any] = {}
    global_numeric: list[pd.Series] = []
    for path in files:
        token = bus_token_from_path(path)
        bus = f"BUS{int(token)}"
        df = _read_csv_sample(path, max_rows=max_rows)
        cols = [c for c in measurement_columns_for_bus(token) if c in df.columns]
        channel_stats = {c: _series_stats(df[c]) for c in cols}
        numeric_cols = [pd.to_numeric(df[c], errors="coerce") for c in cols]
        global_numeric.extend(numeric_cols)
        data_present = pd.to_numeric(df.get("DATA_PRESENT", pd.Series([1] * len(df))), errors="coerce").fillna(1)
        any_nan = df[cols].isna().any(axis=1) if cols else pd.Series(False, index=df.index)
        corr_cols = [c for c in cols if any(s in c for s in ["VA_MAG", "IA_MAG", "Freq", "ROCOF"])]
        corr = df[corr_cols].corr(numeric_only=True).fillna(0.0).to_dict() if len(corr_cols) >= 2 else {}
        buses[bus] = {
            "file": str(path),
            "rows_sampled": int(len(df)),
            "channels": channel_stats,
            "data_present_fraction": float((data_present == 1).mean()),
            "nan_burst_statistics": _nan_bursts(any_nan),
            "cross_channel_correlations": corr,
        }
    all_values = pd.concat(global_numeric, ignore_index=True) if global_numeric else pd.Series(dtype=float)
    clean = pd.to_numeric(all_values, errors="coerce").dropna()
    return {
        "reference_dir": str(root),
        "source": "RAW0001",
        "max_rows_per_bus": max_rows,
        "bus_count": len(buses),
        "buses": buses,
        "global": {
            "numeric_count": int(clean.size),
            "numeric_mean": float(clean.mean()) if not clean.empty else None,
            "numeric_std": float(clean.std(ddof=0)) if not clean.empty else None,
        },
        "fallback_used": False,
    }


def _fallback_reference_statistics(root: Path) -> dict[str, Any]:
    buses: dict[str, Any] = {}
    for bus_num in [39, 29, 10, 22, 19, 2, 5, 6]:
        bus = f"BUS{bus_num}"
        channels = {}
        for suffix in PMU_MEASUREMENT_SUFFIXES:
            col = f"{bus}_{suffix}"
            if suffix.endswith("MAG") and suffix.startswith("V"):
                mean, std = 200000.0, 900.0
            elif suffix.endswith("MAG") and suffix.startswith("I"):
                mean, std = 250.0, 8.0
            elif suffix == "Freq":
                mean, std = 60.0, 0.015
            elif suffix == "ROCOF":
                mean, std = 0.0, 0.08
            else:
                mean, std = -10.0, 0.6
            channels[col] = {"count": 0, "mean": mean, "std": std, "median": mean, "iqr": 1.35 * std, "min": mean - 4 * std, "max": mean + 4 * std, "increment_std": std * 0.5, "nan_fraction": 0.0}
        buses[bus] = {"file": None, "rows_sampled": 0, "channels": channels, "data_present_fraction": 1.0, "nan_burst_statistics": {"burst_count": 0, "max_burst_frames": 0, "mean_burst_frames": 0.0}, "cross_channel_correlations": {}}
    return {"reference_dir": str(root), "source": "fallback", "max_rows_per_bus": 0, "bus_count": len(buses), "buses": buses, "global": {}, "fallback_used": True}


def channel_stat(reference_stats: dict[str, Any], bus: str, suffix: str, key: str, default: float) -> float:
    bus_stats = reference_stats.get("buses", {}).get(bus, {})
    col = f"{bus}_{suffix}"
    value = bus_stats.get("channels", {}).get(col, {}).get(key, None)
    try:
        f = float(value)
        return f if math.isfinite(f) else float(default)
    except Exception:
        return float(default)


def compute_pmu_statistics(pmu_dir: str | Path, max_rows: int | None = None) -> dict[str, Any]:
    root = Path(pmu_dir)
    files = sorted(root.glob("Bus*_Competition_Data*.csv"))
    buses: dict[str, Any] = {}
    for path in files:
        token = bus_token_from_path(path)
        bus = f"BUS{int(token)}"
        df = _read_csv_sample(path, max_rows=max_rows)
        cols = [c for c in measurement_columns_for_bus(token) if c in df.columns]
        buses[bus] = {
            "file": str(path),
            "rows_sampled": int(len(df)),
            "channels": {c: _series_stats(df[c]) for c in cols},
            "data_present_fraction": float((pd.to_numeric(df.get("DATA_PRESENT", 1), errors="coerce") == 1).mean()),
            "event_counts": {str(k): int(v) for k, v in df.get("Event", pd.Series(dtype=int)).value_counts(dropna=False).sort_index().items()},
        }
    return {"pmu_dir": str(root), "bus_count": len(buses), "buses": buses}


def compare_raw_vs_sim(reference_stats: dict[str, Any], sim_pmu_dir: str | Path, max_rows: int | None = None) -> dict[str, Any]:
    sim_stats = compute_pmu_statistics(sim_pmu_dir, max_rows=max_rows)
    per_bus: dict[str, Any] = {}
    all_mean_scores: list[float] = []
    all_std_scores: list[float] = []
    for bus, ref_bus in reference_stats.get("buses", {}).items():
        sim_bus = sim_stats.get("buses", {}).get(bus, {})
        channel_deltas: dict[str, Any] = {}
        for col, ref_ch in ref_bus.get("channels", {}).items():
            sim_ch = sim_bus.get("channels", {}).get(col, {})
            ref_mean = ref_ch.get("mean")
            ref_std = ref_ch.get("std") or ref_ch.get("iqr") or 1.0
            sim_mean = sim_ch.get("mean")
            sim_std = sim_ch.get("std")
            if ref_mean is None or sim_mean is None:
                continue
            scale = max(abs(float(ref_mean)), abs(float(ref_std)), 1.0)
            mean_delta = abs(float(sim_mean) - float(ref_mean)) / scale
            std_delta = abs(float(sim_std or 0.0) - float(ref_std or 0.0)) / max(abs(float(ref_std or 1.0)), 1.0)
            channel_deltas[col] = {
                "reference_mean": ref_mean,
                "sim_mean": sim_mean,
                "normalized_mean_delta": mean_delta,
                "reference_std": ref_std,
                "sim_std": sim_std,
                "normalized_std_delta": std_delta,
            }
            all_mean_scores.append(float(mean_delta))
            all_std_scores.append(float(std_delta))
        mean_score = float(np.nanmean(list(all_mean_scores))) if all_mean_scores else 0.0
        std_score = float(np.nanmean(list(all_std_scores))) if all_std_scores else 0.0
        per_bus[bus] = {
            "channel_deltas": channel_deltas,
            "data_present_fraction_ref": ref_bus.get("data_present_fraction"),
            "data_present_fraction_sim": sim_bus.get("data_present_fraction"),
            "pass_mean_scale": bool(np.nanmean([v["normalized_mean_delta"] for v in channel_deltas.values()] or [0.0]) < 0.35),
            "pass_std_scale": bool(np.nanmean([v["normalized_std_delta"] for v in channel_deltas.values()] or [0.0]) < 2.50),
            "aggregate_seen_so_far": {"mean_distance": mean_score, "std_distance": std_score},
        }
    mean_distance = float(np.nanmean(all_mean_scores)) if all_mean_scores else 0.0
    std_distance = float(np.nanmean(all_std_scores)) if all_std_scores else 0.0
    pass_realism = bool(mean_distance < 0.35 and std_distance < 2.50 and sim_stats.get("bus_count", 0) >= 8)
    return {
        "reference_source": reference_stats.get("source", "unknown"),
        "sim_stats": sim_stats,
        "per_bus": per_bus,
        "distance_summary": {
            "mean_normalized_mean_delta": mean_distance,
            "mean_normalized_std_delta": std_distance,
            "heuristic_thresholds": {"mean_delta_max": 0.35, "std_delta_max": 2.50},
        },
        "pass_realism": pass_realism,
        "notes": [
            "Statistics compare channel means/stds and are intended as a guardrail, not a formal two-sample test.",
            "Synthetic noise is calibrated to RAW0001 channel scale where reference files are available.",
        ],
    }
