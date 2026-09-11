"""Explicit separation between observed PMU channels and evaluation truth."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from pmu_hybrid.constants import PMU_BUSES


def observed_measurements(bus_ids: Iterable[int], voltage: Iterable[complex], pmu_buses: Iterable[int] = PMU_BUSES) -> dict[int, complex]:
    """Return only frozen PMU channels; hidden buses are never exposed."""
    ids = tuple(int(bus) for bus in bus_ids)
    values = np.asarray(tuple(voltage), dtype=complex)
    if values.shape != (len(ids),):
        raise ValueError("voltage vector does not match bus_ids")
    positions = {bus: index for index, bus in enumerate(ids)}
    return {int(bus): complex(values[positions[int(bus)]]) for bus in pmu_buses}


def evaluation_ground_truth(bus_ids: Iterable[int], voltage: Iterable[complex]) -> dict[int, dict[str, float]]:
    """Evaluation-only full electrical field, intentionally separate from observations."""
    ids = tuple(int(bus) for bus in bus_ids)
    values = np.asarray(tuple(voltage), dtype=complex)
    if values.shape != (len(ids),):
        raise ValueError("voltage vector does not match bus_ids")
    return {
        bus: {"V_RE": float(value.real), "V_IM": float(value.imag), "V_MAG": float(abs(value)), "V_ANGLE_RAD": float(np.angle(value))}
        for bus, value in zip(ids, values)
    }
