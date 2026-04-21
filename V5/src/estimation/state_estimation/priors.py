"""Prior builders for M6 regularized state estimation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_loadflow_prior(bus_order: list[str], pmu_metadata: dict) -> np.ndarray:
    """Build complex load-flow prior from metadata v_pu/theta_deg fields."""
    by_bus = {
        str(b["bus_label_canonical"]): (
            float(b.get("v_pu", 1.0)),
            float(b.get("theta_deg", 0.0)),
        )
        for b in pmu_metadata.get("buses", [])
    }
    vec = np.zeros(len(bus_order), dtype=complex)
    for i, bus in enumerate(bus_order):
        mag, ang = by_bus.get(bus, (1.0, 0.0))
        vec[i] = mag * np.exp(1j * np.deg2rad(ang))
    return vec


def initial_state_from_pmu_voltage(
    bus_order: list[str],
    prior: np.ndarray,
    pmu_frame: dict[str, complex],
) -> np.ndarray:
    """Initialize first frame with prior and overwrite available PMU voltages."""
    state = np.asarray(prior, dtype=complex).copy()
    pos = {bus: i for i, bus in enumerate(bus_order)}
    for bus, value in pmu_frame.items():
        if bus in pos and pd.notna(value):
            state[pos[bus]] = complex(value)
    return state

