"""Feature extraction for 8-PMU-only IEEE 39 event models.

This module is deliberately strict about the input contract: model features may
come only from the guideline PMU buses. Scenario JSON and non-PMU CSVs can be
used as labels or validation truth, but never as predictor columns.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

PMU_BUSES = [2, 5, 6, 10, 19, 22, 29, 39]
MEASUREMENT_SUFFIXES = [
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
    "Freq",
    "ROCOF",
]


def bus_columns(bus: int) -> list[str]:
    prefix = f"BUS{bus}"
    return ["TIMESTAMP"] + [f"{prefix}_{suffix}" for suffix in MEASUREMENT_SUFFIXES] + [
        "DATA_PRESENT",
        "Event",
    ]


def measurement_columns(bus: int) -> list[str]:
    prefix = f"BUS{bus}"
    return [f"{prefix}_{suffix}" for suffix in MEASUREMENT_SUFFIXES]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_scenario_jsons(synthetic_dir: Path, max_scenarios: int | None = None) -> list[Path]:
    """Return scenario JSONs in manifest order when possible."""

    manifest_path = synthetic_dir / "synthetic_dataset_manifest.json"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        paths = [synthetic_dir / item["json"] for item in manifest.get("scenarios", [])]
    else:
        paths = sorted(synthetic_dir.glob("SIM_*/SIM_*.json"))
    paths = [path for path in paths if path.exists()]
    if max_scenarios is not None:
        paths = paths[: int(max_scenarios)]
    return paths


def find_bus_csv(input_dir: Path, bus: int) -> Path:
    """Find either raw-style or synthetic PMU-prefixed CSV for a bus."""

    candidates = [
        input_dir / "csv" / f"PMU_Bus{bus}_Competition_Data_nanmask.csv",
        input_dir / "csv" / f"Bus{bus}_Competition_Data_nanmask.csv",
        input_dir / f"PMU_Bus{bus}_Competition_Data_nanmask.csv",
        input_dir / f"Bus{bus}_Competition_Data_nanmask.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find PMU CSV for bus {bus} under {input_dir}")


def read_pmu_frames_from_dir(input_dir: Path, pmu_buses: Iterable[int] = PMU_BUSES) -> dict[int, pd.DataFrame]:
    frames: dict[int, pd.DataFrame] = {}
    for bus in pmu_buses:
        path = find_bus_csv(input_dir, int(bus))
        frame = pd.read_csv(path)
        expected = bus_columns(int(bus))
        if list(frame.columns) != expected:
            raise ValueError(f"{path} does not match the expected PMU schema for bus {bus}")
        frames[int(bus)] = frame
    return frames


def read_scenario_pmu_frames(
    scenario_json: Path,
    pmu_buses: Iterable[int] = PMU_BUSES,
) -> tuple[dict[str, Any], dict[int, pd.DataFrame]]:
    scenario = load_json(scenario_json)
    scenario_dir = scenario_json.parent
    frames: dict[int, pd.DataFrame] = {}
    for bus in pmu_buses:
        bus = int(bus)
        rel_path = scenario.get("csv_files", {}).get(str(bus))
        path = scenario_dir / rel_path if rel_path else find_bus_csv(scenario_dir, bus)
        if not path.exists():
            path = find_bus_csv(scenario_dir, bus)
        frame = pd.read_csv(path)
        expected = bus_columns(bus)
        if list(frame.columns) != expected:
            raise ValueError(f"{path} does not match the expected PMU schema for bus {bus}")
        frames[bus] = frame
    return scenario, frames


def frame_time_bounds(frames: dict[int, pd.DataFrame]) -> tuple[float, float]:
    first = next(iter(frames.values()))
    timestamps = pd.to_numeric(first["TIMESTAMP"], errors="coerce").dropna()
    return float(timestamps.iloc[0]), float(timestamps.iloc[-1])


def _numeric(values: pd.Series) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)


def _window(frame: pd.DataFrame, center_sec: float, window_sec: float) -> pd.DataFrame:
    timestamps = _numeric(frame["TIMESTAMP"])
    half = float(window_sec) / 2.0
    mask = (timestamps >= center_sec - half) & (timestamps <= center_sec + half)
    if not np.any(mask):
        nearest = int(np.nanargmin(np.abs(timestamps - center_sec)))
        mask = np.zeros(len(frame), dtype=bool)
        mask[nearest] = True
    return frame.loc[mask]


def _stats(
    prefix: str,
    values: pd.Series,
    *,
    include_level: bool = False,
    nominal: float | None = None,
) -> dict[str, float]:
    arr = _numeric(values)
    finite = arr[np.isfinite(arr)]
    nan_frac = 1.0 - (float(finite.size) / float(arr.size or 1))
    if finite.size == 0:
        result = {
            f"{prefix}_std": np.nan,
            f"{prefix}_range": np.nan,
            f"{prefix}_last_minus_first": np.nan,
            f"{prefix}_max_abs_delta": np.nan,
            f"{prefix}_relative_range": np.nan,
            f"{prefix}_relative_last_minus_first": np.nan,
            f"{prefix}_relative_max_abs_delta": np.nan,
            f"{prefix}_nan_frac": nan_frac,
        }
        if include_level:
            result.update({f"{prefix}_mean": np.nan, f"{prefix}_min": np.nan, f"{prefix}_max": np.nan})
        if nominal is not None:
            result.update({f"{prefix}_mean_minus_nominal": np.nan, f"{prefix}_max_abs_nominal_dev": np.nan})
        return result
    first = float(finite[0])
    median_level = float(np.median(np.abs(finite))) + 1e-9
    value_range = float(np.max(finite) - np.min(finite))
    last_minus_first = float(finite[-1] - first)
    max_abs_delta = float(np.max(np.abs(finite - first)))
    result = {
        f"{prefix}_std": float(np.std(finite)),
        f"{prefix}_range": value_range,
        f"{prefix}_last_minus_first": last_minus_first,
        f"{prefix}_max_abs_delta": max_abs_delta,
        f"{prefix}_relative_range": value_range / median_level,
        f"{prefix}_relative_last_minus_first": last_minus_first / median_level,
        f"{prefix}_relative_max_abs_delta": max_abs_delta / median_level,
        f"{prefix}_nan_frac": nan_frac,
    }
    if include_level:
        result.update(
            {
                f"{prefix}_mean": float(np.mean(finite)),
                f"{prefix}_min": float(np.min(finite)),
                f"{prefix}_max": float(np.max(finite)),
            }
        )
    if nominal is not None:
        result.update(
            {
                f"{prefix}_mean_minus_nominal": float(np.mean(finite) - nominal),
                f"{prefix}_max_abs_nominal_dev": float(np.max(np.abs(finite - nominal))),
            }
        )
    return result


def _phase_balance_features(window: pd.DataFrame, bus: int, prefix: str) -> dict[str, float]:
    features: dict[str, float] = {}
    for channel, cols in {
        "v_mag": [f"BUS{bus}_VA_MAG", f"BUS{bus}_VB_MAG", f"BUS{bus}_VC_MAG"],
        "i_mag": [f"BUS{bus}_IA_MAG", f"BUS{bus}_IB_MAG", f"BUS{bus}_IC_MAG"],
    }.items():
        phase_means: list[float] = []
        for col in cols:
            finite = _numeric(window[col])
            finite = finite[np.isfinite(finite)]
            phase_means.append(float(np.mean(finite)) if finite.size else np.nan)
        means = np.array(phase_means, dtype=float)
        if np.all(~np.isfinite(means)):
            features[f"{prefix}_{channel}_phase_spread"] = np.nan
            features[f"{prefix}_{channel}_phase_spread_ratio"] = np.nan
            continue
        spread = float(np.nanmax(means) - np.nanmin(means))
        level = float(abs(np.nanmean(means))) + 1e-9
        features[f"{prefix}_{channel}_phase_spread"] = spread
        features[f"{prefix}_{channel}_phase_spread_ratio"] = spread / level
    return features


def extract_window_features(
    frames: dict[int, pd.DataFrame],
    center_sec: float,
    window_sec: float,
    nominal_frequency_hz: float = 60.0,
) -> dict[str, float]:
    """Extract transparent PMU-only window features for one timestamp."""

    features: dict[str, float] = {}
    missing_fractions: list[float] = []
    voltage_sag_ratios: list[float] = []
    current_jump_ratios: list[float] = []
    frequency_deviations: list[float] = []
    rocof_peaks: list[float] = []

    for bus in sorted(frames):
        window = _window(frames[bus], center_sec, window_sec)
        bus_prefix = f"BUS{bus}"
        present = _numeric(window["DATA_PRESENT"])
        missing_fraction = float(np.mean(present == 0)) if present.size else 1.0
        missing_fractions.append(missing_fraction)
        features[f"{bus_prefix}_missing_fraction"] = missing_fraction
        features[f"{bus_prefix}_data_present_mean"] = float(np.nanmean(present)) if present.size else np.nan

        for suffix in MEASUREMENT_SUFFIXES:
            col = f"{bus_prefix}_{suffix}"
            if suffix == "Freq":
                features.update(_stats(col, window[col], include_level=True, nominal=nominal_frequency_hz))
            elif suffix == "ROCOF":
                features.update(_stats(col, window[col], include_level=True, nominal=0.0))
            else:
                features.update(_stats(col, window[col]))

        v_cols = [f"{bus_prefix}_VA_MAG", f"{bus_prefix}_VB_MAG", f"{bus_prefix}_VC_MAG"]
        i_cols = [f"{bus_prefix}_IA_MAG", f"{bus_prefix}_IB_MAG", f"{bus_prefix}_IC_MAG"]
        v_stack = np.concatenate([_numeric(window[col]) for col in v_cols])
        i_stack = np.concatenate([_numeric(window[col]) for col in i_cols])
        if np.isfinite(v_stack).any():
            v_med = float(np.nanmedian(v_stack)) + 1e-9
            voltage_sag_ratios.append(float((v_med - np.nanmin(v_stack)) / abs(v_med)))
        if np.isfinite(i_stack).any():
            i_med = float(np.nanmedian(i_stack)) + 1e-9
            current_jump_ratios.append(float((np.nanmax(i_stack) - i_med) / abs(i_med)))

        freq = _numeric(window[f"{bus_prefix}_Freq"])
        if np.isfinite(freq).any():
            frequency_deviations.append(float(np.nanmax(np.abs(freq - nominal_frequency_hz))))
        rocof = _numeric(window[f"{bus_prefix}_ROCOF"])
        if np.isfinite(rocof).any():
            rocof_peaks.append(float(np.nanmax(np.abs(rocof))))

        features.update(_phase_balance_features(window, bus, bus_prefix))

    features["grid_missing_fraction_max"] = float(np.max(missing_fractions)) if missing_fractions else np.nan
    features["grid_missing_pmu_count"] = float(np.sum(np.array(missing_fractions) > 0.5))
    features["grid_voltage_sag_ratio_max"] = float(np.max(voltage_sag_ratios)) if voltage_sag_ratios else np.nan
    features["grid_current_jump_ratio_max"] = float(np.max(current_jump_ratios)) if current_jump_ratios else np.nan
    features["grid_frequency_deviation_max"] = float(np.max(frequency_deviations)) if frequency_deviations else np.nan
    features["grid_rocof_abs_max"] = float(np.max(rocof_peaks)) if rocof_peaks else np.nan
    return features


def format_line_location(line: Iterable[int]) -> str:
    left, right = sorted(int(bus) for bus in line)
    return f"line_{left}_{right}"


def format_bus_location(bus: int) -> str:
    return f"bus_{int(bus)}"


def event_primary_location(event: dict[str, Any]) -> str:
    """Return the physical target when known, otherwise the PMU telemetry target."""

    line = event.get("line")
    if line:
        return format_line_location(line)
    nodes = event.get("nodes") or []
    if len(nodes) == 1:
        return format_bus_location(int(nodes[0]))
    if len(nodes) > 1:
        return "buses_" + "_".join(str(int(node)) for node in sorted(nodes))
    pmu_bus = event.get("pmu_bus")
    if pmu_bus is not None:
        return format_bus_location(int(pmu_bus))
    return "unknown"


def event_telemetry_location(event: dict[str, Any]) -> str | None:
    pmu_bus = event.get("pmu_bus")
    if pmu_bus is None:
        return None
    return format_bus_location(int(pmu_bus))


def event_sample_centers(event: dict[str, Any], samples_per_event: int) -> list[float]:
    start = float(event["start_sec"])
    end = float(event["end_sec"])
    if samples_per_event <= 1 or end <= start:
        return [round((start + end) / 2.0, 4)]
    span = end - start
    left = start + 0.2 * span
    right = end - 0.2 * span
    centers = np.linspace(left, right, int(samples_per_event))
    return sorted({round(float(center), 4) for center in centers})


def normal_sample_centers(
    duration_sec: float,
    events: list[dict[str, Any]],
    count: int,
    margin_sec: float,
) -> list[float]:
    if count <= 0:
        return []
    intervals = [
        (max(0.0, float(event["start_sec"]) - margin_sec), min(duration_sec, float(event["end_sec"]) + margin_sec))
        for event in events
    ]
    candidates = np.linspace(margin_sec, max(margin_sec, duration_sec - margin_sec), max(count * 12, count))
    selected: list[float] = []
    for candidate in candidates:
        inside_event = any(start <= candidate <= end for start, end in intervals)
        if not inside_event:
            selected.append(round(float(candidate), 4))
        if len(selected) >= count:
            break
    return selected


def feature_frame(rows: list[dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows).sort_index(axis=1)
