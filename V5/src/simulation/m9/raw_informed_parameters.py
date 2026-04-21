"""RAW-informed parameter extraction for Event 0/5/7 cyber modeling."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from src.simulation.m9.calibration import write_json
from src.simulation.m9.constants import PMU_MEASUREMENT_SUFFIXES


TARGET_EVENTS = {0, 5, 7}
EVENT5_STATES = ("NORMAL", "PARTIAL_DROPOUT", "FULL_DROPOUT")
ANGLE_SUFFIXES = [suffix for suffix in PMU_MEASUREMENT_SUFFIXES if suffix.endswith("ANG")]


@dataclass(slots=True)
class ChunkBusFrame:
    chunk_name: str
    chunk_index: int
    event_folder: int
    bus: str
    frame: pd.DataFrame


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        cast = float(value)
        return cast if math.isfinite(cast) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _safe_float(value: float | int | np.number | None, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def _safe_mean(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return 0.0
    return float(np.mean(array))


def _safe_std(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return 0.0
    return float(np.std(array))


def _safe_quantile(values: np.ndarray, q: float) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return 0.0
    return float(np.quantile(array, q))


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    flags = np.asarray(mask, dtype=bool)
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, flag in enumerate(flags):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(flags) - 1))
    return runs


def _run_lengths(mask: np.ndarray) -> list[int]:
    return [end - start + 1 for start, end in _contiguous_runs(mask)]


def _interburst_lengths(mask: np.ndarray) -> list[int]:
    runs = _contiguous_runs(mask)
    if len(runs) <= 1:
        return []
    out: list[int] = []
    for left, right in zip(runs[:-1], runs[1:]):
        out.append(max(0, right[0] - left[1] - 1))
    return out


def _distribution_summary(values: list[int] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "count": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }
    return {
        "count": float(array.size),
        "mean": _safe_mean(array),
        "std": _safe_std(array),
        "p50": _safe_quantile(array, 0.50),
        "p90": _safe_quantile(array, 0.90),
        "p95": _safe_quantile(array, 0.95),
        "max": _safe_float(np.max(array)),
    }


def _wrapped_deg(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return ((array + 180.0) % 360.0) - 180.0


def _wrapped_diff(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size < 2:
        return np.zeros((0,), dtype=float)
    return _wrapped_deg(np.diff(array))


def _channel_family(channel: str) -> str:
    name = str(channel).upper()
    if name == "FREQ":
        return "frequency"
    if name == "ROCOF":
        return "rocof"
    if name.startswith("V"):
        return "voltage"
    if name.startswith("I"):
        return "current"
    return "other"


def _canonicalize_bus_frame(frame: pd.DataFrame, *, bus: str) -> pd.DataFrame:
    out = frame.copy()
    prefix = f"{bus.upper()}_"
    renamed: dict[str, str] = {}
    for column in out.columns:
        upper = str(column).upper()
        if upper.startswith(prefix):
            renamed[column] = upper[len(prefix) :]
        else:
            renamed[column] = upper
    out = out.rename(columns=renamed)
    keep = ["TIMESTAMP", *[suffix.upper() for suffix in PMU_MEASUREMENT_SUFFIXES], "DATA_PRESENT", "EVENT"]
    for column in keep:
        if column not in out.columns:
            out[column] = np.nan
    out = out[keep].copy()
    out["TIMESTAMP"] = pd.to_numeric(out["TIMESTAMP"], errors="coerce")
    out = out.dropna(subset=["TIMESTAMP"]).sort_values("TIMESTAMP").reset_index(drop=True)
    for suffix in [suffix.upper() for suffix in PMU_MEASUREMENT_SUFFIXES]:
        out[suffix] = pd.to_numeric(out[suffix], errors="coerce")
    out["DATA_PRESENT"] = pd.to_numeric(out["DATA_PRESENT"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    out["EVENT"] = pd.to_numeric(out["EVENT"], errors="coerce").fillna(0).astype(int)
    return out


def _parse_chunk_name(chunk_name: str) -> tuple[int, int]:
    match = re.match(r"^chunk(\d+)_event_(\d+)_", chunk_name, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Unable to parse chunk folder name: {chunk_name}")
    return int(match.group(1)), int(match.group(2))


def load_raw_chunks(chunks_root: Path) -> list[ChunkBusFrame]:
    root = chunks_root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Chunks root does not exist: {root}")
    records: list[ChunkBusFrame] = []
    for chunk_dir in sorted([path for path in root.iterdir() if path.is_dir()]):
        chunk_index, event_folder = _parse_chunk_name(chunk_dir.name)
        for csv_path in sorted(chunk_dir.glob("Bus*.csv")):
            bus = csv_path.stem.upper()
            raw = pd.read_csv(csv_path)
            frame = _canonicalize_bus_frame(raw, bus=bus)
            records.append(
                ChunkBusFrame(
                    chunk_name=chunk_dir.name,
                    chunk_index=chunk_index,
                    event_folder=event_folder,
                    bus=bus,
                    frame=frame,
                )
            )
    if not records:
        raise RuntimeError(f"No Bus*.csv files found under {root}")
    return records


def records_to_table(records: list[ChunkBusFrame]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for item in records:
        frame = item.frame.copy()
        frame["BUS"] = item.bus
        frame["CHUNK_NAME"] = item.chunk_name
        frame["CHUNK_INDEX"] = item.chunk_index
        frame["EVENT_FOLDER"] = item.event_folder
        rows.append(frame)
    merged = pd.concat(rows, ignore_index=True)
    merged = merged.loc[merged["EVENT"].isin(TARGET_EVENTS)].copy()
    return merged.reset_index(drop=True)


def _state_from_row(row_values: np.ndarray, data_present: np.ndarray) -> np.ndarray:
    row_nan_fraction = np.mean(~np.isfinite(row_values), axis=1)
    full = (row_nan_fraction >= 0.95) | (data_present < 0.5)
    partial = (row_nan_fraction > 0.0) & (~full)
    states = np.full((len(row_nan_fraction),), "NORMAL", dtype=object)
    states[partial] = "PARTIAL_DROPOUT"
    states[full] = "FULL_DROPOUT"
    return states


def _transition_counts(states: np.ndarray) -> dict[str, dict[str, int]]:
    out = {
        "NORMAL": {"NORMAL": 0, "PARTIAL_DROPOUT": 0, "FULL_DROPOUT": 0},
        "PARTIAL_DROPOUT": {"NORMAL": 0, "PARTIAL_DROPOUT": 0, "FULL_DROPOUT": 0},
        "FULL_DROPOUT": {"NORMAL": 0, "PARTIAL_DROPOUT": 0, "FULL_DROPOUT": 0},
    }
    if len(states) < 2:
        return out
    for prev, curr in zip(states[:-1], states[1:]):
        out[str(prev)][str(curr)] += 1
    return out


def _normalize_transition_counts(counts: dict[str, dict[str, int]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for source, targets in counts.items():
        total = float(sum(targets.values()) + len(targets))
        out[source] = {}
        for target, value in targets.items():
            out[source][target] = float((value + 1.0) / total)
    return out


def _merge_transition_counts(
    base: dict[str, dict[str, int]],
    add: dict[str, dict[str, int]],
) -> dict[str, dict[str, int]]:
    out = {
        "NORMAL": {"NORMAL": base["NORMAL"]["NORMAL"], "PARTIAL_DROPOUT": base["NORMAL"]["PARTIAL_DROPOUT"], "FULL_DROPOUT": base["NORMAL"]["FULL_DROPOUT"]},
        "PARTIAL_DROPOUT": {
            "NORMAL": base["PARTIAL_DROPOUT"]["NORMAL"],
            "PARTIAL_DROPOUT": base["PARTIAL_DROPOUT"]["PARTIAL_DROPOUT"],
            "FULL_DROPOUT": base["PARTIAL_DROPOUT"]["FULL_DROPOUT"],
        },
        "FULL_DROPOUT": {"NORMAL": base["FULL_DROPOUT"]["NORMAL"], "PARTIAL_DROPOUT": base["FULL_DROPOUT"]["PARTIAL_DROPOUT"], "FULL_DROPOUT": base["FULL_DROPOUT"]["FULL_DROPOUT"]},
    }
    for source in out:
        for target in out[source]:
            out[source][target] += int(add[source][target])
    return out


def extract_event5_parameters(data: pd.DataFrame) -> dict[str, Any]:
    event5 = data.loc[data["EVENT"] == 5].copy()
    if event5.empty:
        return {
            "state_transition_matrix": _normalize_transition_counts(_transition_counts(np.asarray(["NORMAL"]))),
            "dropout_burst_length_distribution": _distribution_summary([]),
            "inter_burst_interval_distribution": _distribution_summary([]),
            "full_vs_partial_dropout_fraction": {"full": 0.0, "partial": 0.0},
            "pmu_specific_dropout_tendencies": {},
            "channel_family_dropout_tendencies": {},
            "contiguous_missing_run_statistics": {},
            "onset_recovery_timing": {},
        }

    global_counts = _transition_counts(np.asarray(["NORMAL"]))
    burst_lengths: list[int] = []
    interburst_lengths: list[int] = []
    full_frames = 0
    partial_frames = 0
    pmu_tendencies: dict[str, dict[str, float]] = {}
    channel_family_rates: dict[str, list[float]] = {}
    run_stats: dict[str, dict[str, Any]] = {}
    onset_frames: list[int] = []
    recovery_frames: list[int] = []
    down_transitions = 0
    up_transitions = 0

    channels_upper = [suffix.upper() for suffix in PMU_MEASUREMENT_SUFFIXES]
    for (chunk_name, bus), group in event5.groupby(["CHUNK_NAME", "BUS"], sort=True):
        ordered = group.sort_values("TIMESTAMP").reset_index(drop=True)
        values = ordered[channels_upper].to_numpy(dtype=float)
        data_present = ordered["DATA_PRESENT"].to_numpy(dtype=float)
        states = _state_from_row(values, data_present)
        full_mask = states == "FULL_DROPOUT"
        partial_mask = states == "PARTIAL_DROPOUT"
        dropout_mask = states != "NORMAL"
        full_frames += int(np.sum(full_mask))
        partial_frames += int(np.sum(partial_mask))
        burst_lengths.extend(_run_lengths(dropout_mask))
        interburst_lengths.extend(_interburst_lengths(dropout_mask))

        transitions = _transition_counts(states)
        global_counts = _merge_transition_counts(global_counts, transitions)

        prev = np.roll(states, 1)
        prev[0] = states[0]
        onset = (prev == "NORMAL") & (states != "NORMAL")
        recovery = (prev != "NORMAL") & (states == "NORMAL")
        down_transitions += int(np.sum(onset))
        up_transitions += int(np.sum(recovery))
        onset_frames.extend([int(idx) for idx in np.where(onset)[0]])
        recovery_frames.extend([int(idx) for idx in np.where(recovery)[0]])

        row_nan_fraction = np.mean(~np.isfinite(values), axis=1)
        pmu_tendencies[bus] = {
            "rows": float(len(ordered)),
            "data_present_zero_rate": _safe_float(np.mean(data_present < 0.5)),
            "nan_fraction_mean": _safe_float(np.mean(row_nan_fraction)),
            "partial_dropout_rate": _safe_float(np.mean(partial_mask)),
            "full_dropout_rate": _safe_float(np.mean(full_mask)),
            "dropout_burst_mean_frames": _distribution_summary(_run_lengths(dropout_mask))["mean"],
            "interburst_mean_frames": _distribution_summary(_interburst_lengths(dropout_mask))["mean"],
        }

        for channel in channels_upper:
            channel_values = ordered[channel].to_numpy(dtype=float)
            family = _channel_family(channel)
            channel_family_rates.setdefault(family, []).append(float(np.mean(~np.isfinite(channel_values))))
            key = f"{bus}::{channel}"
            channel_runs = _run_lengths(~np.isfinite(channel_values))
            run_stats[key] = {
                "family": family,
                "run_count": float(len(channel_runs)),
                "run_length_mean": _distribution_summary(channel_runs)["mean"],
                "run_length_p95": _distribution_summary(channel_runs)["p95"],
                "run_length_max": _distribution_summary(channel_runs)["max"],
            }

    total_dropout = float(full_frames + partial_frames)
    family_out: dict[str, dict[str, float]] = {}
    for family, values in channel_family_rates.items():
        family_out[family] = {
            "nan_fraction_mean": _safe_mean(np.asarray(values, dtype=float)),
            "nan_fraction_std": _safe_std(np.asarray(values, dtype=float)),
        }

    return {
        "state_transition_matrix": _normalize_transition_counts(global_counts),
        "dropout_burst_length_distribution": _distribution_summary(burst_lengths),
        "inter_burst_interval_distribution": _distribution_summary(interburst_lengths),
        "full_vs_partial_dropout_fraction": {
            "full": float(full_frames / max(total_dropout, 1.0)),
            "partial": float(partial_frames / max(total_dropout, 1.0)),
        },
        "pmu_specific_dropout_tendencies": pmu_tendencies,
        "channel_family_dropout_tendencies": family_out,
        "contiguous_missing_run_statistics": run_stats,
        "onset_recovery_timing": {
            "down_transitions": int(down_transitions),
            "up_transitions": int(up_transitions),
            "onset_frame_distribution": _distribution_summary(onset_frames),
            "recovery_frame_distribution": _distribution_summary(recovery_frames),
        },
    }


def _robust_sigma(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return 1.0
    median = np.median(array)
    mad = np.median(np.abs(array - median))
    sigma = 1.4826 * mad
    return float(max(sigma, 1e-6))


def _series_jump_rate(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size < 3:
        return 0.0
    diff = np.abs(np.diff(array))
    threshold = float(np.quantile(diff, 0.95))
    if threshold <= 1e-12:
        return 0.0
    return float(np.mean(diff >= threshold))


def _series_outlier_rate(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size < 3:
        return 0.0
    sigma = _robust_sigma(array)
    median = float(np.median(array))
    z = np.abs((array - median) / max(sigma, 1e-6))
    return float(np.mean(z > 3.5))


def _series_stuck_rate(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size < 3:
        return 0.0
    diff = np.abs(np.diff(array))
    tol = max(1e-9, 0.001 * max(float(np.quantile(np.abs(array - np.median(array)), 0.75)), 1e-6))
    return float(np.mean(diff <= tol))


def _psd_lowfreq_ratio(values: np.ndarray, dt: float) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size < 8 or dt <= 0.0 or not math.isfinite(dt):
        return 0.0
    centered = array - np.mean(array)
    spec = np.fft.rfft(centered)
    power = np.abs(spec) ** 2
    freqs = np.fft.rfftfreq(array.size, d=dt)
    total = float(np.sum(power))
    if total <= 1e-12:
        return 0.0
    low = freqs <= min(0.5, 0.5 / dt)
    if not np.any(low):
        return 0.0
    return float(np.sum(power[low]) / total)


def _angle_circular_variance(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size < 3:
        return 0.0
    radians = np.deg2rad(_wrapped_deg(array))
    sin_mean = float(np.mean(np.sin(radians)))
    cos_mean = float(np.mean(np.cos(radians)))
    resultant = math.sqrt(sin_mean * sin_mean + cos_mean * cos_mean)
    return float(1.0 - resultant)


def _baseline_channel_stats(event0: pd.DataFrame) -> dict[str, dict[str, dict[str, float]]]:
    channels = [suffix.upper() for suffix in PMU_MEASUREMENT_SUFFIXES]
    out: dict[str, dict[str, dict[str, float]]] = {}
    for bus, group in event0.groupby("BUS", sort=True):
        out[bus] = {}
        for channel in channels:
            values = group[channel].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                out[bus][channel] = {"median": 0.0, "sigma": 1.0, "iqr": 1.0}
                continue
            out[bus][channel] = {
                "median": float(np.median(values)),
                "sigma": _robust_sigma(values),
                "iqr": _safe_float(np.quantile(values, 0.75) - np.quantile(values, 0.25), 1.0),
            }
    return out


def extract_event7_parameters(data: pd.DataFrame) -> dict[str, Any]:
    event7 = data.loc[data["EVENT"] == 7].copy()
    event0 = data.loc[data["EVENT"] == 0].copy()
    if event7.empty:
        return {
            "mode_prior": {"SPIKE": 0.25, "BIAS_DRIFT": 0.25, "STUCK": 0.25, "REPLAY_LIKE": 0.25},
            "spike_amplitude_distribution": _distribution_summary([]),
            "spike_duration_distribution": _distribution_summary([]),
            "drift_slope_distribution": _distribution_summary([]),
            "stuck_run_length_distribution": _distribution_summary([]),
            "replay_segment_length_distribution": _distribution_summary([]),
            "bad_data_persistence_distribution": _distribution_summary([]),
            "pmu_specific_corruption_tendencies": {},
            "channel_family_corruption_tendencies": {},
            "feature_shift_vs_event0": {},
        }

    baseline = _baseline_channel_stats(event0)
    channels = [suffix.upper() for suffix in PMU_MEASUREMENT_SUFFIXES]
    spike_amplitudes: list[float] = []
    spike_widths: list[int] = []
    drift_slopes: list[float] = []
    stuck_lengths: list[int] = []
    replay_lengths: list[int] = []
    persistence_lengths: list[int] = []
    pmu_rows: dict[str, dict[str, float]] = {}
    family_metrics: dict[str, dict[str, list[float]]] = {}
    mode_counts = {"SPIKE": 0.0, "BIAS_DRIFT": 0.0, "STUCK": 0.0, "REPLAY_LIKE": 0.0}

    for (chunk_name, bus), group in event7.groupby(["CHUNK_NAME", "BUS"], sort=True):
        ordered = group.sort_values("TIMESTAMP").reset_index(drop=True)
        ts = ordered["TIMESTAMP"].to_numpy(dtype=float)
        persistence_lengths.extend(_run_lengths(np.ones((len(ordered),), dtype=bool)))

        jump_list: list[float] = []
        outlier_list: list[float] = []
        stuck_list: list[float] = []
        psd_list: list[float] = []
        angle_var_list: list[float] = []
        dt = np.diff(ts)
        dt = dt[np.isfinite(dt) & (dt > 0.0)]
        dt_median = float(np.median(dt)) if dt.size else 0.033

        for channel in channels:
            values = ordered[channel].to_numpy(dtype=float)
            finite = np.isfinite(values)
            if int(np.sum(finite)) < 4:
                continue
            valid_values = values[finite]
            channel_base = baseline.get(bus, {}).get(channel, {"median": 0.0, "sigma": 1.0, "iqr": 1.0})
            sigma = max(float(channel_base["sigma"]), 1e-6)
            median0 = float(channel_base["median"])
            z = np.abs((valid_values - median0) / sigma)
            spike_mask = z >= 4.0
            spike_amps = z[spike_mask]
            spike_amplitudes.extend(spike_amps.tolist())
            widths = _run_lengths(spike_mask)
            spike_widths.extend(widths)
            if spike_amps.size:
                mode_counts["SPIKE"] += float(spike_amps.size)

            if valid_values.size >= 5 and np.unique(ts[finite]).size >= 5:
                try:
                    slope = float(np.polyfit(ts[finite], valid_values, 1)[0])
                except Exception:
                    slope = 0.0
                drift_slopes.append(slope)
                if abs(slope) > 1e-6:
                    mode_counts["BIAS_DRIFT"] += 1.0

            diffs = np.diff(valid_values)
            tol = max(1e-9, 0.001 * max(float(channel_base["iqr"]), 1e-6))
            stuck_mask = np.abs(diffs) <= tol if diffs.size else np.zeros((0,), dtype=bool)
            stuck_runs = _run_lengths(stuck_mask)
            stuck_lengths.extend(stuck_runs)
            if stuck_runs:
                mode_counts["STUCK"] += float(len(stuck_runs))

            best_replay = 0
            for lag in range(2, min(9, valid_values.size - 1)):
                diff_lag = np.abs(valid_values[lag:] - valid_values[:-lag])
                replay_mask = diff_lag <= (2.0 * tol)
                replay_runs = _run_lengths(replay_mask)
                if replay_runs:
                    best_replay = max(best_replay, int(max(replay_runs)))
            if best_replay > 0:
                replay_lengths.append(best_replay)
                mode_counts["REPLAY_LIKE"] += 1.0

            family = _channel_family(channel)
            family_metrics.setdefault(family, {"jump_rate": [], "outlier_rate": [], "stuck_rate": [], "psd_lowfreq_ratio": []})
            jump = _series_jump_rate(valid_values)
            outlier = _series_outlier_rate(valid_values)
            stuck_rate = _series_stuck_rate(valid_values)
            psd_low = _psd_lowfreq_ratio(valid_values, dt_median)
            family_metrics[family]["jump_rate"].append(jump)
            family_metrics[family]["outlier_rate"].append(outlier)
            family_metrics[family]["stuck_rate"].append(stuck_rate)
            family_metrics[family]["psd_lowfreq_ratio"].append(psd_low)

            jump_list.append(jump)
            outlier_list.append(outlier)
            stuck_list.append(stuck_rate)
            psd_list.append(psd_low)
            if channel in ANGLE_SUFFIXES:
                angle_var_list.append(_angle_circular_variance(valid_values))

        freq_std = _safe_std(ordered["FREQ"].to_numpy(dtype=float))
        rocof_std = _safe_std(ordered["ROCOF"].to_numpy(dtype=float))
        pmu_rows[bus] = {
            "rows": float(len(ordered)),
            "jump_rate_mean": _safe_mean(np.asarray(jump_list, dtype=float)),
            "outlier_rate_mean": _safe_mean(np.asarray(outlier_list, dtype=float)),
            "stuck_rate_mean": _safe_mean(np.asarray(stuck_list, dtype=float)),
            "psd_lowfreq_ratio_mean": _safe_mean(np.asarray(psd_list, dtype=float)),
            "angle_circular_var_mean": _safe_mean(np.asarray(angle_var_list, dtype=float)),
            "freq_std": freq_std,
            "rocof_std": rocof_std,
        }

    family_out: dict[str, dict[str, float]] = {}
    for family, metrics in family_metrics.items():
        family_out[family] = {
            "jump_rate_mean": _safe_mean(np.asarray(metrics["jump_rate"], dtype=float)),
            "outlier_rate_mean": _safe_mean(np.asarray(metrics["outlier_rate"], dtype=float)),
            "stuck_rate_mean": _safe_mean(np.asarray(metrics["stuck_rate"], dtype=float)),
            "psd_lowfreq_ratio_mean": _safe_mean(np.asarray(metrics["psd_lowfreq_ratio"], dtype=float)),
        }

    # Event7 vs Event0 aggregate shifts for the requested diagnostics.
    def _event_feature_snapshot(source: pd.DataFrame) -> dict[str, float]:
        if source.empty:
            return {
                "jump_rate_mean": 0.0,
                "outlier_rate_mean": 0.0,
                "stuck_rate_mean": 0.0,
                "psd_lowfreq_ratio_mean": 0.0,
                "freq_std": 0.0,
                "rocof_std": 0.0,
                "angle_circular_var_mean": 0.0,
            }
        jumps: list[float] = []
        outliers: list[float] = []
        stucks: list[float] = []
        psd_low: list[float] = []
        freq_stds: list[float] = []
        rocof_stds: list[float] = []
        angle_vars: list[float] = []
        for (chunk_name, bus), group in source.groupby(["CHUNK_NAME", "BUS"], sort=True):
            ordered = group.sort_values("TIMESTAMP").reset_index(drop=True)
            ts = ordered["TIMESTAMP"].to_numpy(dtype=float)
            dt = np.diff(ts)
            dt = dt[np.isfinite(dt) & (dt > 0.0)]
            dt_median = float(np.median(dt)) if dt.size else 0.033
            for channel in channels:
                values = ordered[channel].to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                if values.size < 4:
                    continue
                jumps.append(_series_jump_rate(values))
                outliers.append(_series_outlier_rate(values))
                stucks.append(_series_stuck_rate(values))
                psd_low.append(_psd_lowfreq_ratio(values, dt_median))
                if channel in ANGLE_SUFFIXES:
                    angle_vars.append(_angle_circular_variance(values))
            freq_stds.append(_safe_std(ordered["FREQ"].to_numpy(dtype=float)))
            rocof_stds.append(_safe_std(ordered["ROCOF"].to_numpy(dtype=float)))
        return {
            "jump_rate_mean": _safe_mean(np.asarray(jumps, dtype=float)),
            "outlier_rate_mean": _safe_mean(np.asarray(outliers, dtype=float)),
            "stuck_rate_mean": _safe_mean(np.asarray(stucks, dtype=float)),
            "psd_lowfreq_ratio_mean": _safe_mean(np.asarray(psd_low, dtype=float)),
            "freq_std": _safe_mean(np.asarray(freq_stds, dtype=float)),
            "rocof_std": _safe_mean(np.asarray(rocof_stds, dtype=float)),
            "angle_circular_var_mean": _safe_mean(np.asarray(angle_vars, dtype=float)),
        }

    event7_snapshot = _event_feature_snapshot(event7)
    event0_snapshot = _event_feature_snapshot(event0)
    shifts: dict[str, float] = {}
    for key in event7_snapshot:
        shifts[f"{key}_delta"] = float(event7_snapshot[key] - event0_snapshot[key])

    total_modes = sum(mode_counts.values())
    if total_modes <= 0.0:
        mode_prior = {"SPIKE": 0.25, "BIAS_DRIFT": 0.25, "STUCK": 0.25, "REPLAY_LIKE": 0.25}
    else:
        mode_prior = {name: float(value / total_modes) for name, value in mode_counts.items()}

    return {
        "mode_prior": mode_prior,
        "spike_amplitude_distribution": _distribution_summary(spike_amplitudes),
        "spike_duration_distribution": _distribution_summary(spike_widths),
        "drift_slope_distribution": _distribution_summary(drift_slopes),
        "stuck_run_length_distribution": _distribution_summary(stuck_lengths),
        "replay_segment_length_distribution": _distribution_summary(replay_lengths),
        "bad_data_persistence_distribution": _distribution_summary(persistence_lengths),
        "pmu_specific_corruption_tendencies": pmu_rows,
        "channel_family_corruption_tendencies": family_out,
        "feature_shift_vs_event0": shifts,
        "event7_feature_snapshot": event7_snapshot,
        "event0_feature_snapshot": event0_snapshot,
    }


def extract_event0_noise_baseline(data: pd.DataFrame) -> dict[str, Any]:
    event0 = data.loc[data["EVENT"] == 0].copy()
    channels = [suffix.upper() for suffix in PMU_MEASUREMENT_SUFFIXES]
    if event0.empty:
        return {"per_pmu_channel": {}, "channel_family_summary": {}}

    by_channel: dict[str, dict[str, float]] = {}
    family_bucket: dict[str, dict[str, list[float]]] = {}
    for bus, bus_frame in event0.groupby("BUS", sort=True):
        ts = bus_frame["TIMESTAMP"].to_numpy(dtype=float)
        dt = np.diff(ts)
        dt = dt[np.isfinite(dt) & (dt > 0.0)]
        dt_median = float(np.median(dt)) if dt.size else 0.033
        for channel in channels:
            values = bus_frame[channel].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size < 8:
                continue
            median = float(np.median(values))
            sigma = _robust_sigma(values)
            std = _safe_std(values)
            demeaned = values - median
            kurt = 0.0
            if std > 1e-12:
                kurt = float(np.mean((demeaned / std) ** 4) - 3.0)
            student_df = 30.0 if kurt <= 1e-6 else float(np.clip((6.0 / max(kurt, 1e-6)) + 4.0, 3.5, 50.0))
            lag1 = 0.0
            if values.size > 2:
                left = values[:-1]
                right = values[1:]
                if np.std(left) > 1e-12 and np.std(right) > 1e-12:
                    lag1 = float(np.corrcoef(left, right)[0, 1])
            low_ratio = _psd_lowfreq_ratio(values, dt_median)
            family = _channel_family(channel)
            key = f"{bus}::{channel}"
            by_channel[key] = {
                "bus": bus,
                "channel": channel,
                "family": family,
                "median": median,
                "robust_sigma": sigma,
                "std": std,
                "student_t_df": student_df,
                "autocorr_lag1": lag1,
                "psd_lowfreq_ratio": low_ratio,
            }
            family_bucket.setdefault(family, {"robust_sigma": [], "std": [], "student_t_df": [], "autocorr_lag1": [], "psd_lowfreq_ratio": []})
            family_bucket[family]["robust_sigma"].append(sigma)
            family_bucket[family]["std"].append(std)
            family_bucket[family]["student_t_df"].append(student_df)
            family_bucket[family]["autocorr_lag1"].append(lag1)
            family_bucket[family]["psd_lowfreq_ratio"].append(low_ratio)

    family_summary: dict[str, dict[str, float]] = {}
    for family, metrics in family_bucket.items():
        family_summary[family] = {
            "robust_sigma_mean": _safe_mean(np.asarray(metrics["robust_sigma"], dtype=float)),
            "std_mean": _safe_mean(np.asarray(metrics["std"], dtype=float)),
            "student_t_df_mean": _safe_mean(np.asarray(metrics["student_t_df"], dtype=float)),
            "autocorr_lag1_mean": _safe_mean(np.asarray(metrics["autocorr_lag1"], dtype=float)),
            "psd_lowfreq_ratio_mean": _safe_mean(np.asarray(metrics["psd_lowfreq_ratio"], dtype=float)),
        }

    return {
        "per_pmu_channel": by_channel,
        "channel_family_summary": family_summary,
    }


def extract_raw_informed_parameters(
    *,
    chunks_root: Path,
    output_metadata_dir: Path,
) -> dict[str, Any]:
    records = load_raw_chunks(chunks_root)
    table = records_to_table(records)
    event5_params = extract_event5_parameters(table)
    event7_params = extract_event7_parameters(table)
    event0_noise = extract_event0_noise_baseline(table)

    output_metadata_dir.mkdir(parents=True, exist_ok=True)
    event5_path = output_metadata_dir / "event5_params.json"
    event7_path = output_metadata_dir / "event7_params.json"
    event0_path = output_metadata_dir / "event0_noise_baseline.json"
    write_json(event5_path, _json_safe(event5_params))
    write_json(event7_path, _json_safe(event7_params))
    write_json(event0_path, _json_safe(event0_noise))
    return {
        "event5_params_path": str(event5_path),
        "event7_params_path": str(event7_path),
        "event0_noise_baseline_path": str(event0_path),
        "event5_params": event5_params,
        "event7_params": event7_params,
        "event0_noise_baseline": event0_noise,
    }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
