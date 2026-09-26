"""Event-0-aligned measurement model for m4 simulation signals."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any
import sys

import numpy as np
import pandas as pd

from src.simulation.event0_artifacts import resolve_profile_for_bus_signal


def _legacy_m4():
    workspace_root = Path(__file__).resolve().parents[2]
    root_str = str(workspace_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return import_module("m4_andes_faults_type1")


def convert_clean_to_event0_center(
    clean_values: np.ndarray,
    raw_suffix: str,
    artifacts: dict,
    bus_id: str,
) -> np.ndarray:
    """Align clean simulation channel to event-0 center when profile exists."""
    clean = np.asarray(clean_values, dtype=float)
    profile, _ = resolve_profile_for_bus_signal(artifacts, str(bus_id), str(raw_suffix))
    if profile is None:
        return clean.copy()
    center = float((profile.get("eda_stats", {}) or {}).get("median", 0.0))
    if raw_suffix.endswith("_MAG"):
        base = float(np.median(clean)) if len(clean) else 0.0
        if abs(base) < 1e-12:
            return clean.copy()
        return clean * (center / base)
    if raw_suffix == "Freq":
        return clean + (center - float(np.median(clean)))
    if raw_suffix == "ROCOF":
        return clean + (center - float(np.median(clean)))
    return clean.copy()


def apply_event0_measurement_model(
    bus_id: str,
    raw_suffix: str,
    clean_values: np.ndarray,
    t: np.ndarray,
    artifacts: dict,
    rng: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply legacy-parity event-0 measurement model, returning (clean_plot, noisy)."""
    _ = t
    m4 = _legacy_m4()
    clean_plot, noisy = m4.apply_event0_measurement_model(
        bus_id=str(bus_id),
        raw_suffix=str(raw_suffix),
        clean_values=np.asarray(clean_values, dtype=float),
        artifacts=artifacts,
        rng=rng,
    )
    clean_plot = np.asarray(clean_plot, dtype=float)
    noisy = np.asarray(noisy, dtype=float)
    if raw_suffix == "Freq":
        noisy = np.clip(noisy, m4.ESTIMATION_FREQ_CLIP[0], m4.ESTIMATION_FREQ_CLIP[1])
    if raw_suffix == "ROCOF":
        noisy = np.clip(noisy, m4.ESTIMATION_ROCOF_CLIP[0], m4.ESTIMATION_ROCOF_CLIP[1])
    if np.any(~np.isfinite(noisy)):
        noisy = pd.Series(noisy).interpolate(limit_direction="both").bfill().ffill().to_numpy(dtype=float)
    if np.any(~np.isfinite(clean_plot)):
        clean_plot = pd.Series(clean_plot).interpolate(limit_direction="both").bfill().ffill().to_numpy(dtype=float)
    return clean_plot, noisy
