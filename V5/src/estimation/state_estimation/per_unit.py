"""Per-unit conversions for PMU voltage/current phasors."""

from __future__ import annotations

import numpy as np


def voltage_base_phase_volts(kv_ll: float) -> float:
    """Return phase-to-neutral voltage base in volts."""
    kv = float(kv_ll)
    if abs(kv) < 1e-12:
        kv = 345.0
    return kv * 1e3 / np.sqrt(3.0)


def current_base_amps(kv_ll: float, base_mva: float) -> float:
    """Return current base in amps."""
    kv = float(kv_ll)
    if abs(kv) < 1e-12:
        kv = 345.0
    return float(base_mva) * 1e6 / (np.sqrt(3.0) * kv * 1e3)


def voltage_to_pu(voltage_complex: np.ndarray, kv_ll: float) -> np.ndarray:
    """Convert complex voltage phasor to per-unit."""
    return np.asarray(voltage_complex, dtype=complex) / voltage_base_phase_volts(kv_ll)


def current_to_pu(current_complex: np.ndarray, kv_ll: float, base_mva: float) -> np.ndarray:
    """Convert complex current phasor to per-unit."""
    return np.asarray(current_complex, dtype=complex) / current_base_amps(kv_ll, base_mva)

