"""Network extraction and signal synthesis helpers for m4."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
import sys


def _legacy_m4():
    workspace_root = Path(__file__).resolve().parents[2]
    root_str = str(workspace_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return import_module("m4_andes_faults_type1")


def get_bus_kv_map(system) -> dict[str, float]:
    return _legacy_m4().get_bus_kv_map(system)


def build_ybus_from_lines(system):
    return _legacy_m4().build_ybus_from_lines(system)


def build_fault_shunt_current(system, t, v_df, a_df, fault_bus: str, kv_map: dict[str, float]):
    return _legacy_m4().build_fault_shunt_current(system, t, v_df, a_df, str(fault_bus), kv_map)


def extract_all_bus_signals(system, fault_bus: str):
    return _legacy_m4().extract_all_bus_signals(system, str(fault_bus))
