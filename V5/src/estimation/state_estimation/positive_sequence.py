"""Positive-sequence helpers for 3-phase PMU phasors."""

from __future__ import annotations

import numpy as np


def mag_angle_deg_to_complex(magnitude: np.ndarray, angle_deg: np.ndarray) -> np.ndarray:
    """Convert magnitude + degree angle vectors into complex phasor vectors."""
    mag = np.asarray(magnitude, dtype=float)
    ang = np.deg2rad(np.asarray(angle_deg, dtype=float))
    return mag * (np.cos(ang) + 1j * np.sin(ang))


def positive_sequence_from_abc(va: np.ndarray, vb: np.ndarray, vc: np.ndarray) -> np.ndarray:
    """Compute positive-sequence from A/B/C complex phasors."""
    a = np.exp(1j * 2.0 * np.pi / 3.0)
    va_arr = np.asarray(va, dtype=complex)
    vb_arr = np.asarray(vb, dtype=complex)
    vc_arr = np.asarray(vc, dtype=complex)
    return (va_arr + a * vb_arr + (a**2) * vc_arr) / 3.0


def positive_sequence_from_mag_angle(
    va_mag: np.ndarray,
    va_ang_deg: np.ndarray,
    vb_mag: np.ndarray,
    vb_ang_deg: np.ndarray,
    vc_mag: np.ndarray,
    vc_ang_deg: np.ndarray,
) -> np.ndarray:
    """Build positive-sequence directly from per-phase magnitude/angle arrays."""
    va = mag_angle_deg_to_complex(va_mag, va_ang_deg)
    vb = mag_angle_deg_to_complex(vb_mag, vb_ang_deg)
    vc = mag_angle_deg_to_complex(vc_mag, vc_ang_deg)
    return positive_sequence_from_abc(va=va, vb=vb, vc=vc)

