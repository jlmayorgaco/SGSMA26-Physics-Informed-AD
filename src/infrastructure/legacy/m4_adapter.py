"""Thin adapter around legacy `m4_andes_faults_type1.py`."""

from __future__ import annotations

from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from typing import Any, Iterator
import sys


@contextmanager
def _temporary_attr(obj: Any, attr: str, value: Any) -> Iterator[None]:
    had_attr = hasattr(obj, attr)
    old_value = getattr(obj, attr, None)
    setattr(obj, attr, value)
    try:
        yield
    finally:
        if had_attr:
            setattr(obj, attr, old_value)
        else:
            delattr(obj, attr)


def _legacy_m4():
    workspace_root = Path(__file__).resolve().parents[4]
    root_str = str(workspace_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return import_module("m4_andes_faults_type1")


def run_type1_simulation(fault_bus: str | None = None) -> None:
    """Run legacy type-1 ANDES simulation pipeline."""
    m4 = _legacy_m4()
    if fault_bus is None:
        m4.main()
        return

    with _temporary_attr(m4, "FAULT_BUS", str(fault_bus)):
        m4.main()


def compute_estimation_metrics(simulation_dfs: dict, estimated_dfs: dict):
    """Proxy to legacy estimation metric computation."""
    return _legacy_m4().compute_estimation_metrics(simulation_dfs, estimated_dfs)


def export_estimation_report(fault_bus: str, simulation_dfs: dict, estimated_dfs: dict) -> None:
    """Proxy to legacy report export function."""
    _legacy_m4().export_estimation_report(fault_bus, simulation_dfs, estimated_dfs)
