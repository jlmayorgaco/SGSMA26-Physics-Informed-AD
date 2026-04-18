"""Thin adapter around legacy `m3_andes_calibration_raw.py`."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
import sys


def _legacy_m3():
    workspace_root = Path(__file__).resolve().parents[4]
    root_str = str(workspace_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return import_module("m3_andes_calibration_raw")


def run_event0_calibration() -> None:
    """Run legacy ANDES event-0 calibration pipeline."""
    _legacy_m3().run_raw_event0_calibration()
