"""ANDES runtime helpers for m4 type-1 fault simulation."""

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


def build_fault_scenario(system, fault_bus: str):
    """Build type-1 fault device in ANDES system (legacy-parity)."""
    return _legacy_m4().build_fault_scenario(system, str(fault_bus))


def run_fault_simulation_andes(fault_bus: str):
    """Run full ANDES fault simulation (legacy-parity defaults)."""
    return _legacy_m4().run_fault_simulation_andes(str(fault_bus))
