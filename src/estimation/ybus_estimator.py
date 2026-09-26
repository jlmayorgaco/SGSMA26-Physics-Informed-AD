"""YBUS-based estimation helpers for m4."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def _legacy_m4():
    workspace_root = Path(__file__).resolve().parents[2]
    root_str = str(workspace_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return import_module("m4_andes_faults_type1")


def current_base_amp_from_kv(kv_ll: float, system_base_mva: float = 100.0) -> float:
    kv = float(kv_ll)
    if abs(kv) < 1e-12:
        kv = 345.0
    return system_base_mva * 1e6 / (np.sqrt(3.0) * kv * 1e3)


def voltage_base_phase_volts_from_kv(kv_ll: float) -> float:
    kv = float(kv_ll)
    if abs(kv) < 1e-12:
        kv = 345.0
    return kv * 1e3 / np.sqrt(3.0)


def robust_rocof(freq_hz: np.ndarray, t: np.ndarray) -> np.ndarray:
    freq = np.asarray(freq_hz, dtype=float)
    ts = np.asarray(t, dtype=float)
    if len(freq) < 3:
        return np.zeros_like(freq)
    rocof = np.gradient(freq, ts)
    rocof = pd.Series(rocof).rolling(window=5, center=True, min_periods=1).median().to_numpy(dtype=float)
    return np.clip(rocof, -20.0, 20.0)


def positive_sequence_from_abc(va: complex, vb: complex, vc: complex) -> complex:
    a = np.exp(1j * 2.0 * np.pi / 3.0)
    return (va + a * vb + (a**2) * vc) / 3.0


def build_observed_positive_sequence_vector(simulation_dfs: dict, kv_map: dict, pmu_available: list[str], ti: int) -> np.ndarray:
    return _legacy_m4().build_observed_positive_sequence_vector(simulation_dfs, kv_map, pmu_available, int(ti))


def solve_unknown_voltages_static(Yuu: np.ndarray, Yuk: np.ndarray, Vk: np.ndarray) -> np.ndarray:
    rhs = -Yuk @ Vk
    Yuu_reg = Yuu + 1e-8 * np.eye(len(Yuu), dtype=complex)
    try:
        return np.linalg.solve(Yuu_reg, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(Yuu_reg, rhs, rcond=None)[0]


def compute_estimated_currents_from_voltage(V_est: np.ndarray, ybus: np.ndarray) -> np.ndarray:
    currents = np.zeros_like(V_est, dtype=complex)
    for ti in range(V_est.shape[0]):
        currents[ti, :] = ybus @ V_est[ti, :]
    return currents
