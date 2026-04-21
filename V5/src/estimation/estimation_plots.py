"""Estimated-output plotting helpers for m4."""

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


def plot_estimated_signal(*args, **kwargs):
    return _legacy_m4().plot_estimated_signal(*args, **kwargs)


def generate_estimated_bus_plots(*args, **kwargs):
    return _legacy_m4().generate_estimated_bus_plots(*args, **kwargs)
