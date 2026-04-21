"""Simulation plotting helpers for m4 outputs."""

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


def spans_from_event(t, event_arr):
    return _legacy_m4().spans_from_event(t, event_arr)


def add_event_spans(ax, spans):
    return _legacy_m4().add_event_spans(ax, spans)


def plot_one_signal(*args, **kwargs):
    return _legacy_m4().plot_one_signal(*args, **kwargs)


def generate_bus_plots(*args, **kwargs):
    return _legacy_m4().generate_bus_plots(*args, **kwargs)
