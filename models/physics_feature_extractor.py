from __future__ import annotations

import math
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from enevt0_d3tector import Event0RangeDetector, _bus_csv_paths


BUSES = ("BUS2", "BUS5", "BUS6", "BUS10", "BUS19", "BUS22", "BUS29", "BUS39")
PHASES = ("A", "B", "C")


def _json_float(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def event_label_from_name(name: str) -> int | None:
    match = re.search(r"_event(\d+)$", name)
    return int(match.group(1)) if match else None


def load_bus_frames(chunk_dir: Path | str) -> dict[str, pd.DataFrame]:
    chunk_path = Path(chunk_dir)
    frames: dict[str, pd.DataFrame] = {}
    for csv_path in _bus_csv_paths(chunk_path):
        bus = csv_path.name.split("_", 1)[0].upper()
        frames[bus] = pd.read_csv(csv_path)
    if not frames:
        raise FileNotFoundError(f"No Bus*_Competition_Data*.csv files found in {chunk_path}")
    return frames


def missing_data_summary(bus_frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    missing_buses: list[str] = []
    total_rows = 0
    missing_rows = 0
    per_bus: dict[str, dict[str, Any]] = {}
    for bus, frame in bus_frames.items():
        bus_missing = pd.Series(False, index=frame.index)
        if "DATA_PRESENT" in frame.columns:
            data_present = pd.to_numeric(frame["DATA_PRESENT"], errors="coerce")
            bus_missing = bus_missing | (data_present == 0)
        measurement_cols = [col for col in frame.columns if col not in {"TIMESTAMP", "DATA_PRESENT", "Event"}]
        if measurement_cols:
            bus_missing = bus_missing | frame[measurement_cols].isna().any(axis=1)
        count = int(bus_missing.sum())
        total_rows += int(len(frame))
        missing_rows += count
        if count:
            missing_buses.append(bus)
        per_bus[bus] = {
            "missing_rows": count,
            "n_rows": int(len(frame)),
            "missing_fraction": _json_float(count / len(frame) if len(frame) else 0.0),
        }
    return {
        "missing_data": bool(missing_buses),
        "missing_buses": sorted(missing_buses),
        "missing_fraction": _json_float(missing_rows / total_rows if total_rows else 0.0),
        "per_bus": per_bus,
    }


def _nan_op(values: np.ndarray, op: str) -> float | None:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.isnan(arr).all():
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if op == "min":
            return _json_float(np.nanmin(arr))
        if op == "max":
            return _json_float(np.nanmax(arr))
        if op == "span":
            return _json_float(np.nanmax(arr) - np.nanmin(arr))
        if op == "absmax":
            return _json_float(np.nanmax(np.abs(arr)))
        if op == "mean":
            return _json_float(np.nanmean(arr))
        if op == "median":
            return _json_float(np.nanmedian(arr))
    raise ValueError(op)


class PhysicsFeatureExtractor:
    """Build bus-agnostic physics features from PMU CSV chunks/windows."""

    def __init__(self) -> None:
        self.event0_detector = Event0RangeDetector()

    def build_features(self, bus_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
        filtered = self.event0_detector.build_features(bus_frames, include_derivatives=True)
        original_window = self.event0_detector.rolling_window
        try:
            self.event0_detector.rolling_window = 1
            unfiltered = self.event0_detector.build_features(bus_frames, include_derivatives=False)
        finally:
            self.event0_detector.rolling_window = original_window
        for column in unfiltered.columns:
            if "_MAG_PU_FILTERED_" in column:
                filtered[column] = unfiltered[column]
        return filtered

    def summarize_frames(self, bus_frames: dict[str, pd.DataFrame], sample_id: str = "") -> dict[str, Any]:
        x = self.build_features(bus_frames)
        missing = missing_data_summary(bus_frames)
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "n_rows": int(len(x)),
            "duration_seconds_est": _json_float(len(x) / 30.0),
            "missing_data": bool(missing["missing_data"]),
            "missing_fraction": missing["missing_fraction"],
            "missing_bus_count": int(len(missing["missing_buses"])),
        }
        for bus in BUSES:
            prefix = f"{bus}_"
            v_cols = [f"V{p}_MAG_PU_FILTERED_{bus}" for p in PHASES]
            i_cols = [f"I{p}_MAG_PU_FILTERED_{bus}" for p in PHASES]
            vang_cols = [f"V{p}_ANG_RAD_WRAPPED_FILTERED_{bus}" for p in PHASES]
            iang_cols = [f"I{p}_ANG_RAD_WRAPPED_FILTERED_{bus}" for p in PHASES]
            dv_cols = [f"{col}_D1_PER_S" for col in v_cols]
            di_cols = [f"{col}_D1_PER_S" for col in i_cols]
            freq_col = f"Freq_HZ_DEV_FILTERED_{bus}"
            rocof_col = f"ROCOF_HZ_PER_S_FILTERED_{bus}"

            vals_v = x[v_cols].to_numpy(dtype=float)
            vals_i = x[i_cols].to_numpy(dtype=float)
            vals_dv = x[dv_cols].to_numpy(dtype=float)
            vals_di = x[di_cols].to_numpy(dtype=float)
            vals_va = x[vang_cols].to_numpy(dtype=float)
            vals_ia = x[iang_cols].to_numpy(dtype=float)
            vals_freq = x[freq_col].to_numpy(dtype=float)
            vals_rocof = x[rocof_col].to_numpy(dtype=float)

            row[prefix + "v_min"] = _nan_op(vals_v, "min")
            row[prefix + "v_max"] = _nan_op(vals_v, "max")
            row[prefix + "v_span"] = _nan_op(vals_v, "span")
            v_phase_min = np.asarray([_nan_op(vals_v[:, idx], "min") for idx in range(vals_v.shape[1])], dtype=float)
            row[prefix + "v_phase_spread_min"] = _nan_op(v_phase_min, "span")
            row[prefix + "i_min"] = _nan_op(vals_i, "min")
            row[prefix + "i_max"] = _nan_op(vals_i, "max")
            row[prefix + "i_span"] = _nan_op(vals_i, "span")
            i_phase_max = np.asarray([_nan_op(vals_i[:, idx], "max") for idx in range(vals_i.shape[1])], dtype=float)
            row[prefix + "i_phase_spread_max"] = _nan_op(i_phase_max, "span")
            row[prefix + "dv_abs_max"] = _nan_op(vals_dv, "absmax")
            row[prefix + "di_abs_max"] = _nan_op(vals_di, "absmax")
            row[prefix + "freq_span"] = _nan_op(vals_freq, "span")
            head = _nan_op(vals_freq[: max(1, len(vals_freq) // 5)], "median")
            tail = _nan_op(vals_freq[-max(1, len(vals_freq) // 5) :], "median")
            row[prefix + "freq_abs_step"] = _json_float(abs(tail - head)) if head is not None and tail is not None else None
            row[prefix + "rocof_abs_max"] = _nan_op(vals_rocof, "absmax")
            row[prefix + "v_angle_span"] = _nan_op(vals_va, "span")
            row[prefix + "i_angle_span"] = _nan_op(vals_ia, "span")

            # Approximate apparent impedance magnitude per phase. It is not a
            # substitute for phasor-domain line impedance, but the V/I collapse
            # is physically meaningful for faults and outages.
            z = vals_v / np.where(np.abs(vals_i) > 1e-6, vals_i, np.nan)
            row[prefix + "z_min"] = _nan_op(z, "min")
            row[prefix + "z_span"] = _nan_op(z, "span")

        metric_names = [
            "v_min",
            "v_max",
            "v_span",
            "i_max",
            "i_span",
            "dv_abs_max",
            "di_abs_max",
            "freq_span",
            "freq_abs_step",
            "rocof_abs_max",
            "z_min",
            "z_span",
        ]
        for metric in metric_names:
            values = [row.get(f"{bus}_{metric}") for bus in BUSES]
            values = np.asarray([value for value in values if value is not None], dtype=float)
            if values.size:
                row[f"GLOBAL_{metric}_max"] = _json_float(np.nanmax(values))
                row[f"GLOBAL_{metric}_min"] = _json_float(np.nanmin(values))
                row[f"GLOBAL_{metric}_mean"] = _json_float(np.nanmean(values))
        return row

    def summarize_chunk(self, chunk_dir: Path | str) -> dict[str, Any]:
        return self.summarize_frames(load_bus_frames(chunk_dir), sample_id=Path(chunk_dir).name)
