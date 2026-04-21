"""Selection-operator helpers for PMU bus subsets."""

from __future__ import annotations

import numpy as np


def bus_index_map(bus_order: list[str]) -> dict[str, int]:
    """Build deterministic bus->index mapping."""
    return {str(bus): i for i, bus in enumerate(bus_order)}


def voltage_selection_matrix(bus_order: list[str], measured_buses: list[str]) -> np.ndarray:
    """Build S_v such that V_measured = S_v @ V_full."""
    pos = bus_index_map(bus_order)
    rows: list[np.ndarray] = []
    for bus in measured_buses:
        if bus not in pos:
            continue
        row = np.zeros(len(bus_order), dtype=complex)
        row[pos[bus]] = 1.0 + 0j
        rows.append(row)
    if not rows:
        return np.zeros((0, len(bus_order)), dtype=complex)
    return np.vstack(rows)

