"""Thin adapter around legacy `m2_noise_profiling_raw.py`."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
import sys


def _legacy_m2():
    workspace_root = Path(__file__).resolve().parents[4]
    root_str = str(workspace_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return import_module("m2_noise_profiling_raw")


def run_noise_profile_pipeline(chunks_dir: str | None = None, out_path: str | None = None) -> dict:
    """Run legacy raw noise profile generation."""
    m2 = _legacy_m2()
    if chunks_dir is None and out_path is None:
        return m2.generate_raw_profiles()
    return m2.generate_raw_profiles(
        chunks_dir=chunks_dir if chunks_dir is not None else m2.CHUNKS_DIR,
        out_path=out_path if out_path is not None else m2.PROFILE_OUT,
    )
