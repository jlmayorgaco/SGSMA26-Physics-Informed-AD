"""Shared event-label helpers for type0 and type1 scenarios."""

from __future__ import annotations

import numpy as np


def event_series_type0(t: np.ndarray) -> np.ndarray:
    """Type0 labels: always normal operation (0)."""
    t_arr = np.asarray(t, dtype=float)
    return np.zeros(len(t_arr), dtype=int)


def event_series_type1(
    t: np.ndarray,
    mode: str = "windowed",
    fault_start_s: float = 5.0,
    fault_clear_s: float = 5.1,
) -> np.ndarray:
    """Type1 labels: windowed fault interval or constant file label."""
    t_arr = np.asarray(t, dtype=float)
    if mode == "file_constant":
        return np.ones(len(t_arr), dtype=int)
    return ((t_arr >= float(fault_start_s)) & (t_arr <= float(fault_clear_s))).astype(int)
