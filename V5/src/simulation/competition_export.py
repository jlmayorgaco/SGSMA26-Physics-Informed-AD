"""Competition-format export helpers for m4 simulation outputs."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
import sys

import numpy as np


def _legacy_m4():
    workspace_root = Path(__file__).resolve().parents[2]
    root_str = str(workspace_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return import_module("m4_andes_faults_type1")


def event_series(t: np.ndarray, mode: str = "windowed") -> np.ndarray:
    """Build event labels in parity with legacy m4."""
    t_arr = np.asarray(t, dtype=float)
    if mode == "file_constant":
        return np.ones(len(t_arr), dtype=int)
    fault_start_s = 5.0
    fault_clear_s = 5.1
    return ((t_arr >= fault_start_s) & (t_arr <= fault_clear_s)).astype(int)


def raw_col(bus_id: str, suffix: str) -> str:
    return f"BUS{bus_id}_{suffix}"


def build_competition_dataframe_for_bus(bus_id: str, bus_signals: dict, artifacts: dict, rng):
    return _legacy_m4().build_competition_dataframe_for_bus(str(bus_id), bus_signals, artifacts, rng)


def export_simulation_outputs(signals_by_bus: dict[str, dict], artifacts: dict, fault_bus: str):
    return _legacy_m4().export_simulation_outputs(signals_by_bus, artifacts, str(fault_bus))
