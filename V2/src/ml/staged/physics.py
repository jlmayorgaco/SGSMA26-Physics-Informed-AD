"""Physics-aware feature preprocessing for staged V2 models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.ml.models.ieee39_physics import (
    ALL_BUSES,
    IEEE39_BRANCHES,
    PMU_BUSES,
    bus_index,
    electrical_distance_matrix,
    finite_or_zero,
    phase_delta_deg,
    phasor,
    safe_log1p,
    ybus_matrix,
)


def _as_frame(values: pd.DataFrame | np.ndarray, feature_names: list[str] | None = None) -> pd.DataFrame:
    if isinstance(values, pd.DataFrame):
        return values.copy()
    array = np.asarray(values)
    names = feature_names or [f"f_{index}" for index in range(array.shape[1])]
    return pd.DataFrame(array, columns=names)


def _col(frame: pd.DataFrame, name: str, default: float = 0.0) -> np.ndarray:
    if name not in frame:
        return np.full(len(frame), default, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)


def _mean_cols(frame: pd.DataFrame, names: Iterable[str], default: float = 0.0) -> np.ndarray:
    cols = [_col(frame, name, default) for name in names if name in frame]
    if not cols:
        return np.full(len(frame), default, dtype=float)
    matrix = np.column_stack(cols)
    valid = np.isfinite(matrix)
    counts = np.sum(valid, axis=1)
    totals = np.sum(np.where(valid, matrix, 0.0), axis=1)
    return np.where(counts > 0, totals / np.maximum(counts, 1), default)


@dataclass(frozen=True)
class GridStateEstimate:
    voltage: np.ndarray
    pmu_residuals: np.ndarray
    bus_residuals: np.ndarray


class YbusStateEstimator:
    """Reconstruct hidden-bus voltage phasors from observable PMU buses."""

    def estimate(self, frame: pd.DataFrame) -> GridStateEstimate:
        pmu_voltage = self._pmu_voltage_phasors(frame)
        voltage = self._reconstruct_all_bus_voltage(pmu_voltage)
        pmu_residuals = self._pmu_kcl_residuals(frame, voltage)
        bus_residuals = self._spread_residuals(pmu_residuals)
        return GridStateEstimate(voltage=voltage, pmu_residuals=pmu_residuals, bus_residuals=bus_residuals)

    def _pmu_voltage_phasors(self, frame: pd.DataFrame) -> np.ndarray:
        values = np.zeros((len(frame), len(PMU_BUSES)), dtype=complex)
        for col, bus in enumerate(PMU_BUSES):
            prefix = f"BUS{bus}"
            magnitude = _mean_cols(
                frame,
                [f"{prefix}_VA_MAG_mean", f"{prefix}_VB_MAG_mean", f"{prefix}_VC_MAG_mean"],
                default=1.0,
            )
            angle_a = _col(frame, f"{prefix}_VA_ANG_mean")
            angle_b = _col(frame, f"{prefix}_VB_ANG_mean") + 120.0
            angle_c = _col(frame, f"{prefix}_VC_ANG_mean") - 120.0
            values[:, col] = phasor(magnitude, self._mean_angle_deg(np.column_stack([angle_a, angle_b, angle_c])))
        return values

    def _mean_angle_deg(self, angles_deg: np.ndarray) -> np.ndarray:
        radians = np.deg2rad(np.asarray(angles_deg, dtype=float))
        valid = np.isfinite(radians)
        counts = np.sum(valid, axis=1)
        cos_v = np.sum(np.where(valid, np.cos(radians), 0.0), axis=1) / np.maximum(counts, 1)
        sin_v = np.sum(np.where(valid, np.sin(radians), 0.0), axis=1) / np.maximum(counts, 1)
        return np.rad2deg(np.arctan2(sin_v, cos_v))

    def _reconstruct_all_bus_voltage(self, pmu_voltage: np.ndarray) -> np.ndarray:
        known_idx = np.array([bus_index(bus) for bus in PMU_BUSES], dtype=int)
        hidden_idx = np.array([bus_index(bus) for bus in ALL_BUSES if bus not in PMU_BUSES], dtype=int)
        ybus = ybus_matrix()
        y_hh = ybus[np.ix_(hidden_idx, hidden_idx)]
        y_hk = ybus[np.ix_(hidden_idx, known_idx)]
        regularized = y_hh + np.eye(len(hidden_idx), dtype=complex) * 1e-6
        voltage = np.zeros((pmu_voltage.shape[0], len(ALL_BUSES)), dtype=complex)
        voltage[:, known_idx] = pmu_voltage
        rhs = -(y_hk @ pmu_voltage.T)
        try:
            hidden = np.linalg.solve(regularized, rhs).T
        except np.linalg.LinAlgError:
            hidden = np.linalg.lstsq(regularized, rhs, rcond=None)[0].T
        voltage[:, hidden_idx] = hidden
        return voltage

    def _pmu_kcl_residuals(self, frame: pd.DataFrame, voltage: np.ndarray) -> np.ndarray:
        estimated_current = voltage @ ybus_matrix().T
        residuals = np.zeros((len(frame), len(PMU_BUSES)), dtype=float)
        for col, bus in enumerate(PMU_BUSES):
            prefix = f"BUS{bus}"
            measured_i_mag = _mean_cols(
                frame,
                [f"{prefix}_IA_MAG_mean", f"{prefix}_IB_MAG_mean", f"{prefix}_IC_MAG_mean"],
            )
            measured_i_ang = _mean_cols(
                frame,
                [f"{prefix}_IA_ANG_mean", f"{prefix}_IB_ANG_mean", f"{prefix}_IC_ANG_mean"],
            )
            measured_i = phasor(measured_i_mag, measured_i_ang)
            estimate = estimated_current[:, bus_index(bus)]
            scale = np.nanmedian(np.abs(measured_i[np.isfinite(measured_i)]))
            scale = float(scale) if np.isfinite(scale) and scale > 1e-9 else 1.0
            residuals[:, col] = np.abs(estimate - measured_i) / scale
        return safe_log1p(finite_or_zero(residuals))

    def _spread_residuals(self, pmu_residuals: np.ndarray) -> np.ndarray:
        distances = electrical_distance_matrix()[:, [bus_index(bus) for bus in PMU_BUSES]]
        scale = max(float(np.nanmedian(distances)), 1e-6)
        weights = np.exp(-distances / scale)
        weights = weights / np.maximum(np.sum(weights, axis=1, keepdims=True), 1e-9)
        return finite_or_zero(pmu_residuals @ weights.T)


class PhysicsFeatureEngineer(BaseEstimator, TransformerMixin):
    """Append named Ybus, Zbus, branch, and swing features to PMU features."""

    def __init__(self, include_original: bool = True) -> None:
        self.include_original = bool(include_original)
        self.input_features_: list[str] = []

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray | None = None) -> "PhysicsFeatureEngineer":
        frame = _as_frame(X)
        self.input_features_ = list(frame.columns)
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        frame = _as_frame(X, self.input_features_).reindex(columns=self.input_features_, fill_value=np.nan)
        estimate = YbusStateEstimator().estimate(frame)
        physics = self._estimate_features(estimate, frame.index)
        swing = self._swing_features(frame)
        pieces = [physics, swing]
        if self.include_original:
            pieces.insert(0, frame)
        return pd.concat(pieces, axis=1)

    def _estimate_features(self, estimate: GridStateEstimate, index: pd.Index) -> pd.DataFrame:
        data: dict[str, np.ndarray] = {}
        mag = np.abs(estimate.voltage)
        angle = np.rad2deg(np.angle(estimate.voltage))
        for bus in ALL_BUSES:
            idx = bus_index(bus)
            data[f"phys_bus_{bus:02d}_v_mag_est"] = safe_log1p(mag[:, idx])
            data[f"phys_bus_{bus:02d}_v_angle_est"] = angle[:, idx] / 180.0
            data[f"phys_bus_{bus:02d}_ybus_residual"] = estimate.bus_residuals[:, idx]
        for col, bus in enumerate(PMU_BUSES):
            data[f"phys_pmu_{bus:02d}_kcl_residual"] = estimate.pmu_residuals[:, col]
        for left, right in IEEE39_BRANCHES:
            li = bus_index(left)
            ri = bus_index(right)
            key = f"phys_line_{left:02d}_{right:02d}"
            data[f"{key}_endpoint_residual"] = np.maximum(estimate.bus_residuals[:, li], estimate.bus_residuals[:, ri])
            data[f"{key}_angle_delta"] = np.abs(phase_delta_deg(angle[:, li], angle[:, ri])) / 180.0
        data["phys_grid_ybus_residual_max"] = np.nanmax(estimate.bus_residuals, axis=1)
        data["phys_grid_ybus_residual_mean"] = np.nanmean(estimate.bus_residuals, axis=1)
        data["phys_grid_ybus_residual_std"] = np.nanstd(estimate.bus_residuals, axis=1)
        data["phys_grid_voltage_spread"] = np.nanstd(mag, axis=1) / np.maximum(np.nanmean(np.abs(mag), axis=1), 1e-9)
        return pd.DataFrame({key: finite_or_zero(value) for key, value in data.items()}, index=index)

    def _swing_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        freq = np.column_stack([_col(frame, f"BUS{bus}_Freq_mean", 60.0) for bus in PMU_BUSES])
        rocof = np.column_stack([_col(frame, f"BUS{bus}_ROCOF_mean", 0.0) for bus in PMU_BUSES])
        freq_dev = np.column_stack([_col(frame, f"BUS{bus}_Freq_max_abs_nominal_dev", 0.0) for bus in PMU_BUSES])
        rocof_peak = np.column_stack([_col(frame, f"BUS{bus}_ROCOF_max_abs_nominal_dev", 0.0) for bus in PMU_BUSES])
        data = {
            "swing_freq_mean": np.nanmean(freq, axis=1),
            "swing_freq_spread": np.nanmax(freq, axis=1) - np.nanmin(freq, axis=1),
            "swing_freq_abs_dev_max": np.nanmax(np.abs(freq_dev), axis=1),
            "swing_rocof_abs_mean": np.nanmean(np.abs(rocof), axis=1),
            "swing_rocof_abs_max": np.nanmax(np.abs(rocof_peak), axis=1),
        }
        data["swing_accel_proxy"] = data["swing_rocof_abs_max"] / np.maximum(data["swing_freq_abs_dev_max"], 1e-6)
        return pd.DataFrame({key: finite_or_zero(value) for key, value in data.items()}, index=frame.index)
