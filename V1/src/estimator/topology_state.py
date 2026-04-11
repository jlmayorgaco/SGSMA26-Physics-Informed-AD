"""Topology-constrained IEEE-39 voltage-state reconstruction from 8 PMUs.

This module provides a lightweight alternative to the full UKF: estimate the
voltage magnitude and angle at all 39 buses by solving a Ybus-weighted harmonic
extension problem with the 8 PMU buses as Dirichlet boundary conditions.

It is not a dynamic state estimator. It is a fast Kirchhoff/topology-consistent
state proxy that gives downstream classifiers/localizers features for non-PMU
buses such as Bus 7 and line endpoints such as Bus 24--23.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from src.io.load_csv import PMU_BUSES

if TYPE_CHECKING:
    import pandas as pd
    from src.grid.load_case import GridCase


def _circular_mean_deg(values: np.ndarray) -> float:
    """Return circular mean of angle samples in degrees."""
    vals = values[np.isfinite(values)]
    if len(vals) == 0:
        return float("nan")
    rad = np.deg2rad(vals)
    return float(np.rad2deg(np.angle(np.mean(np.exp(1j * rad)))))


def _angle_diff_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Wrapped angle difference a-b in degrees."""
    return (a - b + 180.0) % 360.0 - 180.0


@dataclass
class TopologyStateEstimator:
    """Ybus-graph harmonic estimator for full 39-bus voltage states."""

    grid: "GridCase"
    regularization: float = 1e-6

    def __post_init__(self) -> None:
        y_abs = np.abs(np.asarray(self.grid.Ybus, dtype=complex))
        np.fill_diagonal(y_abs, 0.0)
        # Use admittance magnitude as edge strength. Larger |Y_ij| means a
        # stronger Kirchhoff coupling, so the harmonic extension changes more
        # smoothly across that branch.
        self.W = y_abs
        self.L = np.diag(self.W.sum(axis=1)) - self.W
        self.base_vm = np.array([b.vm_pu for b in self.grid.buses], dtype=float)
        self.base_va = np.array([b.va_deg for b in self.grid.buses], dtype=float)
        self.bus_to_idx = {bus: i for i, bus in enumerate(self.grid.ext_bus_order)}
        self.pmu_indices = np.array(self.grid.pmu_bus_indices, dtype=int)

    @property
    def bus_order(self) -> list[int]:
        return list(self.grid.ext_bus_order)

    def _extend(self, obs_idx: np.ndarray, obs_values: np.ndarray, base: np.ndarray) -> np.ndarray:
        """Harmonically extend observed values to all buses."""
        n = len(base)
        full = base.astype(float).copy()
        valid = np.isfinite(obs_values)
        obs_idx = obs_idx[valid]
        obs_values = obs_values[valid]
        if len(obs_idx) == 0:
            return full

        full[obs_idx] = obs_values
        unobs = np.array([i for i in range(n) if i not in set(obs_idx.tolist())], dtype=int)
        if len(unobs) == 0:
            return full

        A = self.L[np.ix_(unobs, unobs)] + self.regularization * np.eye(len(unobs))
        rhs = -self.L[np.ix_(unobs, obs_idx)] @ obs_values
        try:
            full[unobs] = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            full[unobs] = np.linalg.lstsq(A, rhs, rcond=None)[0]
        return full

    def observed_voltage(self, df: "pd.DataFrame", start: int, end: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return observed PMU indices, mean voltage magnitudes, and VA angles.

        Magnitude uses the three-phase average; angle uses phase A because VB/VC
        include nominal +/-120 degree phase offsets.
        """
        window = df.iloc[max(0, start): max(0, end)]
        obs_idx: list[int] = []
        vm_obs: list[float] = []
        va_obs: list[float] = []

        for bus, idx in zip(PMU_BUSES, self.pmu_indices):
            mag_cols = [f"BUS{bus}_{ph}_MAG" for ph in ("VA", "VB", "VC")]
            mag_cols = [c for c in mag_cols if c in window.columns]
            ang_col = f"BUS{bus}_VA_ANG"
            if not mag_cols or ang_col not in window.columns:
                continue

            mags = window[mag_cols].to_numpy(dtype=float)
            mag_vals = mags[np.isfinite(mags)]
            ang_vals = window[ang_col].to_numpy(dtype=float)
            if len(mag_vals) == 0 or not np.isfinite(ang_vals).any():
                continue

            obs_idx.append(int(idx))
            # Convert volts to p.u. using each bus base kV line-to-neutral.
            base_v_ln = self.grid.buses[int(idx)].base_kv * 1000.0 / np.sqrt(3.0)
            vm_obs.append(float(np.nanmean(mag_vals) / max(base_v_ln, 1.0)))
            va_obs.append(_circular_mean_deg(ang_vals))

        return np.array(obs_idx, dtype=int), np.array(vm_obs, dtype=float), np.array(va_obs, dtype=float)

    def estimate_window(self, df: "pd.DataFrame", start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
        """Estimate all-bus (vm_pu, va_deg) for a time window."""
        obs_idx, vm_obs, va_obs = self.observed_voltage(df, start, end)
        vm = self._extend(obs_idx, vm_obs, self.base_vm)
        va = self._extend(obs_idx, va_obs, self.base_va)
        return vm, va

    def residual_energy(
        self,
        df: "pd.DataFrame",
        onset_frame: int,
        *,
        fps: float = 30.0,
        window_sec: float = 3.0,
        baseline_sec: float = 30.0,
    ) -> np.ndarray:
        """Return all-bus normalized residual energy around an event onset."""
        lookahead = max(1, int(round(fps * window_sec)))
        baseline = max(1, int(round(fps * baseline_sec)))
        t0 = max(0, int(onset_frame))
        b0 = max(0, t0 - baseline)
        b1 = max(b0 + 1, t0)
        e1 = min(len(df), t0 + lookahead)

        vm0, va0 = self.estimate_window(df, b0, b1)
        vm1, va1 = self.estimate_window(df, t0, e1)
        dvm = vm1 - vm0
        dva_rad = np.deg2rad(_angle_diff_deg(va1, va0))

        # Remove common-mode motion. Frequency swings and reference-angle drift
        # are system-wide; localization needs the spatially differential part
        # that Kirchhoff/Ybus propagation leaves across the network.
        dvm = dvm - np.nanmedian(dvm)
        dva_rad = dva_rad - np.nanmedian(dva_rad)

        # Normalize by practical PMU disturbance scales: 0.2% voltage and
        # roughly 0.1 electrical degree. These constants affect ranking only.
        e = np.sqrt((dvm / 0.002) ** 2 + (dva_rad / np.deg2rad(0.1)) ** 2)
        return np.where(np.isfinite(e), e, 0.0)

    def residual_by_bus(self, df: "pd.DataFrame", onset_frame: int, **kwargs) -> dict[int, float]:
        """Return ``{competition_bus: residual_energy}`` for all 39 buses."""
        energy = self.residual_energy(df, onset_frame, **kwargs)
        return {bus: float(energy[i]) for i, bus in enumerate(self.bus_order)}


def state_feature_summary(
    df: "pd.DataFrame",
    onset_frame: int,
    estimator: TopologyStateEstimator,
    *,
    fps: float = 30.0,
    window_sec: float = 3.0,
) -> tuple[list[float], dict[int, float]]:
    """Return compact full-state features and the underlying all-bus energies."""
    by_bus = estimator.residual_by_bus(
        df, onset_frame, fps=fps, window_sec=window_sec
    )
    items = sorted(by_bus.items(), key=lambda kv: -kv[1])
    top_bus, top_energy = items[0] if items else (0, 0.0)
    energies = np.array([v for _, v in items], dtype=float)
    total = float(energies.sum())
    entropy = 0.0
    if total > 1e-12:
        p = energies / total
        entropy = float(-np.sum(p * np.log(p + 1e-30)))

    bus7 = by_bus.get(7, 0.0)
    bus23 = by_bus.get(23, 0.0)
    bus24 = by_bus.get(24, 0.0)
    line_2423 = 0.5 * (bus23 + bus24) + abs(bus24 - bus23)
    features = [
        float(top_bus),
        float(top_energy),
        entropy,
        float(bus7),
        float(bus23),
        float(bus24),
        float(line_2423),
    ]
    return features, by_bus
