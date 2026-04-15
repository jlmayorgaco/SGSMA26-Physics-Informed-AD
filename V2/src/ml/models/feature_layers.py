"""Physics and topology feature layers for 8-PMU model families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .ieee39_physics import (
    ALL_BUSES,
    IEEE39_BRANCHES,
    PMU_BUSES,
    bus_index,
    electrical_distance_matrix,
    finite_or_zero,
    normalized_adjacency,
    phase_delta_deg,
    phasor,
    radians_to_unit,
    safe_log1p,
    ybus_matrix,
)


def _as_frame(values: pd.DataFrame | np.ndarray, feature_names: list[str] | None = None) -> pd.DataFrame:
    if isinstance(values, pd.DataFrame):
        return values
    names = feature_names or [f"f_{index}" for index in range(np.asarray(values).shape[1])]
    return pd.DataFrame(values, columns=names)


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
class BusSignalSpec:
    name: str
    default: float = 0.0


class GraphTopologyFeatureLayer(BaseEstimator, TransformerMixin):
    """Diffuse PMU disturbance indicators over the IEEE-39 graph."""

    def __init__(self, propagation_steps: int = 3, include_original: bool = True) -> None:
        self.propagation_steps = int(propagation_steps)
        self.include_original = bool(include_original)
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray | None = None) -> "GraphTopologyFeatureLayer":
        frame = _as_frame(X)
        self.feature_names_ = list(frame.columns)
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        frame = _as_frame(X, self.feature_names_)
        node_signals = self._node_signals(frame)
        adjacency = normalized_adjacency()
        propagated = [node_signals]
        current = node_signals
        for _ in range(max(0, self.propagation_steps)):
            current = np.einsum("ij,njc->nic", adjacency, current)
            propagated.append(current)
        graph_features = np.concatenate([layer.reshape(len(frame), -1) for layer in propagated], axis=1)
        graph_features = finite_or_zero(graph_features)
        if not self.include_original:
            return graph_features
        return np.hstack([finite_or_zero(frame.to_numpy(dtype=float)), graph_features])

    def _node_signals(self, frame: pd.DataFrame) -> np.ndarray:
        signals = np.zeros((len(frame), len(ALL_BUSES), 8), dtype=float)
        for pmu in PMU_BUSES:
            idx = bus_index(pmu)
            prefix = f"BUS{pmu}"
            v_disturbance = _mean_cols(
                frame,
                [
                    f"{prefix}_VA_MAG_relative_max_abs_delta",
                    f"{prefix}_VB_MAG_relative_max_abs_delta",
                    f"{prefix}_VC_MAG_relative_max_abs_delta",
                ],
            )
            i_disturbance = _mean_cols(
                frame,
                [
                    f"{prefix}_IA_MAG_relative_max_abs_delta",
                    f"{prefix}_IB_MAG_relative_max_abs_delta",
                    f"{prefix}_IC_MAG_relative_max_abs_delta",
                ],
            )
            freq = _col(frame, f"{prefix}_Freq_max_abs_nominal_dev")
            rocof = _col(frame, f"{prefix}_ROCOF_max_abs_nominal_dev")
            missing = _col(frame, f"{prefix}_missing_fraction")
            v_unbalance = _col(frame, f"{prefix}_v_mag_phase_spread_ratio")
            i_unbalance = _col(frame, f"{prefix}_i_mag_phase_spread_ratio")
            bad_data_score = np.maximum(missing, np.maximum(v_unbalance, i_unbalance))
            signals[:, idx, :] = np.column_stack(
                [
                    v_disturbance,
                    i_disturbance,
                    freq,
                    rocof,
                    missing,
                    v_unbalance,
                    i_unbalance,
                    bad_data_score,
                ]
            )
        return finite_or_zero(signals)


class PhysicsYbusFeatureLayer(BaseEstimator, TransformerMixin):
    """Reconstruct hidden-bus voltages from PMUs and add Ybus/KCL residual cues."""

    def __init__(self, include_original: bool = True) -> None:
        self.include_original = bool(include_original)
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray | None = None) -> "PhysicsYbusFeatureLayer":
        frame = _as_frame(X)
        self.feature_names_ = list(frame.columns)
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        frame = _as_frame(X, self.feature_names_)
        pmu_voltage = self._pmu_voltage_phasors(frame)
        reconstructed = self._reconstruct_all_bus_voltage(pmu_voltage)
        residuals = self._pmu_kcl_residuals(frame, reconstructed)
        bus_residuals = self._spread_residuals(residuals)
        branch_features = self._branch_features(reconstructed, bus_residuals)
        grid_features = self._grid_features(reconstructed, residuals, bus_residuals)
        physics_features = np.hstack(
            [
                finite_or_zero(np.abs(reconstructed)),
                finite_or_zero(np.angle(reconstructed)),
                bus_residuals,
                branch_features,
                grid_features,
            ]
        )
        if not self.include_original:
            return physics_features
        return np.hstack([finite_or_zero(frame.to_numpy(dtype=float)), physics_features])

    def _pmu_voltage_phasors(self, frame: pd.DataFrame) -> np.ndarray:
        values = np.zeros((len(frame), len(PMU_BUSES)), dtype=complex)
        for col, bus in enumerate(PMU_BUSES):
            prefix = f"BUS{bus}"
            magnitude = _mean_cols(
                frame,
                [
                    f"{prefix}_VA_MAG_mean",
                    f"{prefix}_VB_MAG_mean",
                    f"{prefix}_VC_MAG_mean",
                ],
                default=1.0,
            )
            angle = _col(frame, f"{prefix}_VA_ANG_mean")
            if f"{prefix}_VB_ANG_mean" in frame and f"{prefix}_VC_ANG_mean" in frame:
                angle_b = _col(frame, f"{prefix}_VB_ANG_mean") + 120.0
                angle_c = _col(frame, f"{prefix}_VC_ANG_mean") - 120.0
                angle_matrix = np.column_stack([angle, angle_b, angle_c])
                valid = np.isfinite(angle_matrix)
                counts = np.sum(valid, axis=1)
                totals = np.sum(np.where(valid, angle_matrix, 0.0), axis=1)
                angle = np.where(counts > 0, totals / np.maximum(counts, 1), 0.0)
            values[:, col] = phasor(magnitude, angle)
        return values

    def _reconstruct_all_bus_voltage(self, pmu_voltage: np.ndarray) -> np.ndarray:
        known_idx = np.array([bus_index(bus) for bus in PMU_BUSES], dtype=int)
        hidden_idx = np.array([bus_index(bus) for bus in ALL_BUSES if bus not in PMU_BUSES], dtype=int)
        ybus = ybus_matrix()
        y_hh = ybus[np.ix_(hidden_idx, hidden_idx)]
        y_hk = ybus[np.ix_(hidden_idx, known_idx)]
        regularized = y_hh + np.eye(len(hidden_idx), dtype=complex) * 1e-6
        reconstructed = np.zeros((pmu_voltage.shape[0], len(ALL_BUSES)), dtype=complex)
        reconstructed[:, known_idx] = pmu_voltage
        try:
            hidden = np.linalg.solve(regularized, -(y_hk @ pmu_voltage.T)).T
        except np.linalg.LinAlgError:
            hidden = np.linalg.lstsq(regularized, -(y_hk @ pmu_voltage.T), rcond=None)[0].T
        reconstructed[:, hidden_idx] = hidden
        return reconstructed

    def _pmu_kcl_residuals(self, frame: pd.DataFrame, reconstructed: np.ndarray) -> np.ndarray:
        ybus_current = reconstructed @ ybus_matrix().T
        residuals = np.zeros((len(frame), len(PMU_BUSES)), dtype=float)
        for col, bus in enumerate(PMU_BUSES):
            prefix = f"BUS{bus}"
            measured_i_mag = _mean_cols(
                frame,
                [
                    f"{prefix}_IA_MAG_mean",
                    f"{prefix}_IB_MAG_mean",
                    f"{prefix}_IC_MAG_mean",
                ],
            )
            measured_i_angle = _col(frame, f"{prefix}_IA_ANG_mean")
            measured_i = phasor(measured_i_mag, measured_i_angle)
            estimate = ybus_current[:, bus_index(bus)]
            scale = np.nanmedian(np.abs(measured_i[np.isfinite(measured_i)]))
            scale = float(scale) if np.isfinite(scale) and scale > 1e-9 else 1.0
            residuals[:, col] = np.abs(estimate - measured_i) / scale
        return safe_log1p(finite_or_zero(residuals))

    def _spread_residuals(self, residuals: np.ndarray) -> np.ndarray:
        distances = electrical_distance_matrix()[:, [bus_index(bus) for bus in PMU_BUSES]]
        scale = max(float(np.nanmedian(distances)), 1e-6)
        weights = np.exp(-distances / scale)
        weights = weights / np.maximum(np.sum(weights, axis=1, keepdims=True), 1e-9)
        return finite_or_zero(residuals @ weights.T)

    def _branch_features(self, reconstructed: np.ndarray, bus_residuals: np.ndarray) -> np.ndarray:
        features: list[np.ndarray] = []
        angles = np.rad2deg(np.angle(reconstructed))
        magnitudes = np.abs(reconstructed)
        for left, right in IEEE39_BRANCHES:
            li = bus_index(left)
            ri = bus_index(right)
            mag_delta = np.abs(magnitudes[:, li] - magnitudes[:, ri])
            angle_delta = np.abs(phase_delta_deg(angles[:, li], angles[:, ri]))
            endpoint_residual = np.maximum(bus_residuals[:, li], bus_residuals[:, ri])
            features.extend([safe_log1p(mag_delta), angle_delta / 180.0, endpoint_residual])
        return finite_or_zero(np.column_stack(features))

    def _grid_features(
        self,
        reconstructed: np.ndarray,
        residuals: np.ndarray,
        bus_residuals: np.ndarray,
    ) -> np.ndarray:
        mag = np.abs(reconstructed)
        angle = np.rad2deg(np.angle(reconstructed))
        cos_a, sin_a = radians_to_unit(angle)
        return finite_or_zero(
            np.column_stack(
                [
                    np.nanmax(residuals, axis=1),
                    np.nanmean(residuals, axis=1),
                    np.nanmax(bus_residuals, axis=1),
                    np.nanmean(bus_residuals, axis=1),
                    np.nanstd(bus_residuals, axis=1),
                    np.nanstd(mag, axis=1) / np.maximum(np.nanmean(np.abs(mag), axis=1), 1e-9),
                    np.nanstd(cos_a, axis=1),
                    np.nanstd(sin_a, axis=1),
                ]
            )
        )
