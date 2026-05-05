from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features.pmu_discovery import discover_bus_csvs
from src.helpers.paths import DEFAULT_TOPOLOGY_DIR


SIGNALS = (
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
)
VOLTAGE_MAG_SIGNALS = ("VA_MAG", "VB_MAG", "VC_MAG")
CURRENT_MAG_SIGNALS = ("IA_MAG", "IB_MAG", "IC_MAG")
ANGLE_SIGNALS = ("VA_ANG", "VB_ANG", "VC_ANG", "IA_ANG", "IB_ANG", "IC_ANG")
META_COLUMNS = {"TIMESTAMP", "DATA_PRESENT", "Event"}
PMU_FEATURE_COLUMNS = (
    "bus_id",
    "data_present",
    "nan_fraction",
    "voltage_dev",
    "current_dev",
    "angle_dev",
    "freq_dev",
    "rocof_abs",
    "single_signal_score",
    "severity",
    "time_to_peak",
    "degree",
)
GLOBAL_FEATURE_COLUMNS = (
    "n_pmus",
    "present_fraction_min",
    "present_fraction_mean",
    "nan_fraction_max",
    "severity_max",
    "severity_mean",
    "severity_std",
    "voltage_dev_max",
    "current_dev_max",
    "freq_dev_max",
    "rocof_abs_max",
    "single_signal_score_max",
    "top_bus_id",
    "top_bus_degree",
)


@dataclass(frozen=True)
class PredictionDiagnostics:
    input_dir: str
    output_csv: str
    buses_detected: list[int]
    model: str
    warnings: list[str]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_dir": self.input_dir,
            "output_csv": self.output_csv,
            "buses_detected": self.buses_detected,
            "model": self.model,
            "warnings": self.warnings,
            "summary": self.summary,
        }


def discover_pmu_csvs(input_dir: Path) -> dict[int, Path]:
    return discover_bus_csvs(input_dir)


def _signal_column(frame: pd.DataFrame, bus: int, signal: str) -> str | None:
    direct = f"BUS{bus}_{signal}"
    if direct in frame.columns:
        return direct
    suffix = f"_{signal}".upper()
    matches = [col for col in frame.columns if col.upper().endswith(suffix)]
    if matches:
        return matches[0]
    if signal in frame.columns:
        return signal
    return None


def _numeric(frame: pd.DataFrame, column: str | None, fallback: float = np.nan) -> np.ndarray:
    if column is None or column not in frame:
        return np.full(len(frame), fallback, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def _unwrap_degrees(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    if finite.sum() < 2:
        return values.astype(float)
    filled = pd.Series(values).interpolate(limit_direction="both").bfill().ffill().to_numpy(dtype=float)
    out = np.rad2deg(np.unwrap(np.deg2rad(filled)))
    out[~finite] = np.nan
    return out


def _baseline(values: np.ndarray, fraction: float = 0.08) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    n = max(10, int(math.ceil(len(arr) * fraction)))
    head = arr[: min(len(arr), n)]
    head = head[np.isfinite(head)]
    if head.size < 3:
        head = finite
    center = float(np.nanmedian(head))
    mad = float(np.nanmedian(np.abs(head - center))) if head.size else 0.0
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-9:
        scale = float(np.nanstd(head)) if head.size > 1 else 1.0
    if not np.isfinite(scale) or scale < 1e-9:
        scale = max(abs(center) * 0.002, 1.0)
    return center, scale


def _robust_z(values: np.ndarray) -> np.ndarray:
    center, scale = _baseline(values)
    return np.abs((values - center) / max(scale, 1e-9))


def _safe_max(arrays: list[np.ndarray], length: int) -> np.ndarray:
    valid = [np.asarray(arr, dtype=float) for arr in arrays if len(arr) == length]
    if not valid:
        return np.zeros(length, dtype=float)
    stacked = np.vstack(valid)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        out = np.nanmax(stacked, axis=0)
    out[~np.isfinite(out)] = 0.0
    return out


def _safe_mean(arrays: list[np.ndarray], length: int) -> np.ndarray:
    valid = [np.asarray(arr, dtype=float) for arr in arrays if len(arr) == length]
    if not valid:
        return np.zeros(length, dtype=float)
    stacked = np.vstack(valid)
    with np.errstate(all="ignore"):
        out = np.nanmean(stacked, axis=0)
    out[~np.isfinite(out)] = 0.0
    return out


def _load_degrees(topology_dir: Path = DEFAULT_TOPOLOGY_DIR) -> dict[int, int]:
    path = topology_dir / "branches_physical.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    degrees: dict[int, int] = {}
    from_col = "from_bus" if "from_bus" in frame.columns else frame.columns[0]
    to_col = "to_bus" if "to_bus" in frame.columns else frame.columns[1]
    for row in frame.itertuples(index=False):
        a = int(getattr(row, from_col))
        b = int(getattr(row, to_col))
        degrees[a] = degrees.get(a, 0) + 1
        degrees[b] = degrees.get(b, 0) + 1
    return degrees


def _load_lines(topology_dir: Path = DEFAULT_TOPOLOGY_DIR) -> list[tuple[int, int]]:
    path = topology_dir / "branches_physical.csv"
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    from_col = "from_bus" if "from_bus" in frame.columns else frame.columns[0]
    to_col = "to_bus" if "to_bus" in frame.columns else frame.columns[1]
    lines: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for row in frame.itertuples(index=False):
        a = int(getattr(row, from_col))
        b = int(getattr(row, to_col))
        key = tuple(sorted((a, b)))
        if key not in seen:
            seen.add(key)
            lines.append((a, b))
    return lines


def load_pmu_frames(input_dir: Path) -> dict[int, pd.DataFrame]:
    csvs = discover_pmu_csvs(input_dir)
    if not csvs:
        existing = []
        if Path(input_dir).exists():
            existing = [path.name for path in sorted(Path(input_dir).iterdir())[:20]]
        raise ValueError(
            "No Bus*.csv PMU files were found. "
            f"Checked input directory: {Path(input_dir).resolve()}. "
            "Pass the folder that contains files such as Bus2_Competition_Data_nanmask.csv "
            "or a parent folder containing RAW0001/RAW001. "
            f"First entries found: {existing}"
        )
    frames: dict[int, pd.DataFrame] = {}
    for bus, path in csvs.items():
        frame = pd.read_csv(path)
        if "TIMESTAMP" not in frame.columns:
            raise ValueError(f"{path} does not contain TIMESTAMP")
        frames[bus] = frame.sort_values("TIMESTAMP").reset_index(drop=True)
    return frames


def extract_bus_agnostic_features(
    frames: dict[int, pd.DataFrame],
    topology_dir: Path = DEFAULT_TOPOLOGY_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract order-invariant PMU and global features for arbitrary Bus*.csv files."""
    degrees = _load_degrees(topology_dir)
    pmu_rows: list[pd.DataFrame] = []
    for bus, frame in sorted(frames.items()):
        length = len(frame)
        signal_z: dict[str, np.ndarray] = {}
        signal_raw: dict[str, np.ndarray] = {}
        for signal in SIGNALS:
            values = _numeric(frame, _signal_column(frame, bus, signal))
            if signal in ANGLE_SIGNALS:
                values = _unwrap_degrees(values)
            signal_raw[signal] = values
            signal_z[signal] = _robust_z(values)

        data_present = _numeric(frame, "DATA_PRESENT", fallback=1.0)
        data_present = np.where(np.isfinite(data_present), data_present, 1.0)
        measurement_cols = [_signal_column(frame, bus, signal) for signal in SIGNALS]
        available_measurement_cols = [col for col in measurement_cols if col is not None and col in frame]
        if available_measurement_cols:
            nan_fraction = frame[available_measurement_cols].isna().mean(axis=1).to_numpy(dtype=float)
        else:
            nan_fraction = np.ones(length, dtype=float)

        voltage_dev = _safe_max([signal_z[s] for s in VOLTAGE_MAG_SIGNALS], length)
        current_dev = _safe_max([signal_z[s] for s in CURRENT_MAG_SIGNALS], length)
        angle_dev = _safe_max([signal_z[s] for s in ANGLE_SIGNALS], length)
        freq_dev = signal_z.get("Freq", np.zeros(length, dtype=float))
        rocof_abs = np.abs(signal_raw.get("ROCOF", np.zeros(length, dtype=float)))
        single_signal_score = _safe_max(list(signal_z.values()), length)
        severity = np.maximum.reduce(
            [
                voltage_dev,
                current_dev,
                angle_dev * 0.7,
                freq_dev * 1.2,
                np.minimum(rocof_abs / 0.02, 50.0),
                nan_fraction * 30.0,
                (1.0 - data_present) * 40.0,
            ]
        )
        time_to_peak = np.zeros(length, dtype=float)
        if length > 1:
            peak_idx = int(np.nanargmax(np.nan_to_num(severity, nan=0.0)))
            t = _numeric(frame, "TIMESTAMP", fallback=0.0)
            denom = max(float(np.nanmax(t) - np.nanmin(t)), 1e-9)
            time_to_peak[:] = (float(t[peak_idx] - np.nanmin(t))) / denom

        pmu_rows.append(
            pd.DataFrame(
                {
                    "TIMESTAMP": _numeric(frame, "TIMESTAMP", fallback=0.0),
                    "Bus": int(bus),
                    "bus_id": float(bus),
                    "data_present": data_present,
                    "nan_fraction": nan_fraction,
                    "voltage_dev": voltage_dev,
                    "current_dev": current_dev,
                    "angle_dev": angle_dev,
                    "freq_dev": freq_dev,
                    "rocof_abs": rocof_abs,
                    "single_signal_score": single_signal_score,
                    "severity": severity,
                    "time_to_peak": time_to_peak,
                    "degree": float(degrees.get(int(bus), 0)),
                }
            )
        )

    if not pmu_rows:
        raise ValueError("No Bus*.csv PMU files were found.")

    pmu_features = pd.concat(pmu_rows, ignore_index=True)
    grouped = pmu_features.groupby("TIMESTAMP", sort=True)
    global_features = grouped.agg(
        n_pmus=("Bus", "nunique"),
        present_fraction_min=("data_present", "min"),
        present_fraction_mean=("data_present", "mean"),
        nan_fraction_max=("nan_fraction", "max"),
        severity_max=("severity", "max"),
        severity_mean=("severity", "mean"),
        severity_std=("severity", "std"),
        voltage_dev_max=("voltage_dev", "max"),
        current_dev_max=("current_dev", "max"),
        freq_dev_max=("freq_dev", "max"),
        rocof_abs_max=("rocof_abs", "max"),
        single_signal_score_max=("single_signal_score", "max"),
    ).reset_index()
    global_features["severity_std"] = global_features["severity_std"].fillna(0.0)
    top_idx = grouped["severity"].idxmax()
    top_rows = pmu_features.loc[top_idx, ["TIMESTAMP", "Bus", "degree"]].rename(
        columns={"Bus": "top_bus_id", "degree": "top_bus_degree"}
    )
    global_features = global_features.merge(top_rows, on="TIMESTAMP", how="left")
    global_features["top_bus_id"] = global_features["top_bus_id"].fillna(0).astype(int)
    global_features["top_bus_degree"] = global_features["top_bus_degree"].fillna(0.0)
    return pmu_features, global_features


class BusAgnosticPhysicsModel:
    """Bus-agnostic inference model for arbitrary IEEE-39 PMU placements.

    This model intentionally does not depend on fixed BUS-prefixed training columns.
    It uses robust per-PMU baselines, order-invariant global features, and topology
    metadata to produce a valid SGSMA submission CSV for arbitrary Bus*.csv inputs.
    """

    model_name = "bus_agnostic_physics_v1"

    def __init__(self, topology_dir: Path = DEFAULT_TOPOLOGY_DIR, config: dict[str, Any] | None = None) -> None:
        self.topology_dir = Path(topology_dir)
        self.lines = _load_lines(self.topology_dir)
        self.config = {
            "physical_severity_threshold": 10.0,
            "physical_voltage_threshold": 8.0,
            "physical_current_threshold": 8.0,
            "physical_frequency_threshold": 7.0,
            "physical_rocof_threshold": 0.03,
            "bad_data_single_signal_threshold": 25.0,
            "fault_voltage_threshold": 12.0,
            "fault_current_threshold": 6.0,
            "line_current_threshold": 10.0,
            "line_voltage_max": 10.0,
            "generation_frequency_threshold": 8.0,
            "generation_rocof_threshold": 0.05,
        }
        if config:
            self.config.update(config)

    def _line_location(self, top_buses: list[int]) -> str:
        if not self.lines or not top_buses:
            return "UNKNOWN"
        top = [int(bus) for bus in top_buses]
        best_line = self.lines[0]
        best_score = float("inf")
        for a, b in self.lines:
            score = min(abs(a - bus) for bus in top) + min(abs(b - bus) for bus in top)
            if len(top) >= 2:
                score += min(abs(a - top[0]) + abs(b - top[1]), abs(a - top[1]) + abs(b - top[0]))
            if score < best_score:
                best_score = float(score)
                best_line = (a, b)
        return f"LINE{min(best_line)}-{max(best_line)}"

    def predict(self, frames: dict[int, pd.DataFrame]) -> tuple[pd.DataFrame, PredictionDiagnostics]:
        pmu_features, global_features = extract_bus_agnostic_features(frames, self.topology_dir)
        merged = pmu_features.merge(global_features, on="TIMESTAMP", how="left", suffixes=("", "_global"))
        missing_local = (merged["data_present"].to_numpy(dtype=float) < 0.5) | (merged["nan_fraction"].to_numpy(dtype=float) > 0.90)
        missing_global = (merged["present_fraction_min"].to_numpy(dtype=float) < 0.5) | (
            merged["nan_fraction_max"].to_numpy(dtype=float) > 0.90
        )
        physical = (
            (merged["severity_max"].to_numpy(dtype=float) >= float(self.config["physical_severity_threshold"]))
            | (merged["voltage_dev_max"].to_numpy(dtype=float) >= float(self.config["physical_voltage_threshold"]))
            | (merged["current_dev_max"].to_numpy(dtype=float) >= float(self.config["physical_current_threshold"]))
            | (merged["freq_dev_max"].to_numpy(dtype=float) >= float(self.config["physical_frequency_threshold"]))
            | (merged["rocof_abs_max"].to_numpy(dtype=float) >= float(self.config["physical_rocof_threshold"]))
        )
        single_signal = merged["single_signal_score"].to_numpy(dtype=float)
        severity_max = merged["severity_max"].to_numpy(dtype=float)
        bad_data = (
            (single_signal >= float(self.config["bad_data_single_signal_threshold"]))
            & (severity_max < np.maximum(single_signal * 1.15, 35.0))
            & ~missing_local
        )

        fault = (merged["voltage_dev_max"].to_numpy(dtype=float) >= float(self.config["fault_voltage_threshold"])) & (
            merged["current_dev_max"].to_numpy(dtype=float) >= float(self.config["fault_current_threshold"])
        )
        line = (merged["current_dev_max"].to_numpy(dtype=float) >= float(self.config["line_current_threshold"])) & (
            merged["voltage_dev_max"].to_numpy(dtype=float) < float(self.config["line_voltage_max"])
        )
        generation = (merged["freq_dev_max"].to_numpy(dtype=float) >= float(self.config["generation_frequency_threshold"])) | (
            merged["rocof_abs_max"].to_numpy(dtype=float) >= float(self.config["generation_rocof_threshold"])
        )
        load = merged["severity_max"].to_numpy(dtype=float) >= float(self.config["physical_severity_threshold"])

        pred_event_arr = np.select(
            [
                missing_local & physical,
                missing_local | (missing_global & ~physical),
                bad_data,
                ~physical,
                fault,
                line,
                generation,
                load,
            ],
            [6, 5, 7, 0, 1, 2, 3, 4],
            default=8,
        ).astype(int)

        top_bus = merged["top_bus_id"].astype(int).to_numpy()
        own_bus = merged["Bus"].astype(int).to_numpy()
        bus_location_arr = np.char.add("BUS", top_bus.astype(str))
        pmu_location_arr = np.char.add("PMU", own_bus.astype(str))
        line_cache = {int(bus): self._line_location([int(bus)]) for bus in np.unique(top_bus)}
        line_location_arr = np.array([line_cache[int(bus)] for bus in top_bus], dtype=object)
        pred_location_arr = np.where(
            pred_event_arr == 0,
            "none",
            np.where(
                np.isin(pred_event_arr, [5, 7]),
                pmu_location_arr,
                np.where(pred_event_arr == 2, line_location_arr, bus_location_arr),
            ),
        )
        top1 = bus_location_arr.astype(str)
        top2 = line_location_arr.astype(str)
        top3 = pmu_location_arr.astype(str)

        out = pd.DataFrame(
            {
                "TIMESTAMP": merged["TIMESTAMP"].to_numpy(dtype=float),
                "Bus": merged["Bus"].astype(int).to_numpy(),
                "Predicted_Event": pred_event_arr,
                "Predicted_Location": pred_location_arr,
                "Top1_Location": top1,
                "Top2_Location": top2,
                "Top3_Location": top3,
                "Severity": merged["severity"].to_numpy(dtype=float),
                "Global_Severity": merged["severity_max"].to_numpy(dtype=float),
            }
        ).sort_values(["TIMESTAMP", "Bus"], kind="mergesort")
        summary = {
            "n_rows": int(len(out)),
            "n_timestamps": int(out["TIMESTAMP"].nunique()),
            "event_counts": {str(k): int(v) for k, v in out["Predicted_Event"].value_counts().sort_index().items()},
            "location_counts_top10": {str(k): int(v) for k, v in out["Predicted_Location"].value_counts().head(10).items()},
        }
        diagnostics = PredictionDiagnostics(
            input_dir="",
            output_csv="",
            buses_detected=sorted(int(bus) for bus in frames),
            model=self.model_name,
            warnings=[
                "Bus-agnostic physics/ranking route was used; fixed BUS2/BUS5/BUS6/BUS10/BUS19/BUS22/BUS29/BUS39 columns are not required.",
                "Event labels are inferred from PMU measurements only; any Event column in input files is ignored.",
            ],
            summary=summary,
        )
        return out, diagnostics


def run_bus_agnostic_prediction(
    input_dir: Path,
    output_csv: Path | None = None,
    topology_dir: Path = DEFAULT_TOPOLOGY_DIR,
    model_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    input_dir = Path(input_dir)
    if output_csv is None:
        output_csv = input_dir / "predictions.csv"
    config: dict[str, Any] | None = None
    if model_dir is not None:
        config_path = Path(model_dir) / "model_config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
    frames = load_pmu_frames(input_dir)
    model = BusAgnosticPhysicsModel(topology_dir=topology_dir, config=config)
    predictions, diagnostics = model.predict(frames)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    predictions[["TIMESTAMP", "Bus", "Predicted_Event", "Predicted_Location"]].to_csv(output_csv, index=False)
    diagnostics_dict = diagnostics.to_dict()
    diagnostics_dict["input_dir"] = str(input_dir.resolve())
    diagnostics_dict["output_csv"] = str(output_csv.resolve())
    diagnostics_dict["schema"] = {
        "pmu_feature_columns": list(PMU_FEATURE_COLUMNS),
        "global_feature_columns": list(GLOBAL_FEATURE_COLUMNS),
        "output_columns": ["TIMESTAMP", "Bus", "Predicted_Event", "Predicted_Location"],
    }
    diagnostics_path = output_csv.parent / "prediction_diagnostics.json"
    diagnostics_path.write_text(json.dumps(diagnostics_dict, indent=2), encoding="utf-8")
    return predictions, diagnostics_dict
