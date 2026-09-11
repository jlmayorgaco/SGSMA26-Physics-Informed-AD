"""Complex AC network operators with explicit branch-terminal semantics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Branch:
    """One oriented pi-model branch in system-base per unit."""

    identifier: str
    from_bus: int
    to_bus: int
    resistance_pu: float
    reactance_pu: float
    charging_pu: float = 0.0
    tap: float = 1.0
    shift_rad: float = 0.0
    in_service: bool = True

    @property
    def complex_tap(self) -> complex:
        return self.tap * np.exp(1j * self.shift_rad)


def branch_terminal_currents(branch: Branch, v_from: complex, v_to: complex) -> tuple[complex, complex]:
    """Return currents leaving the from and to terminals into the branch.

    The equations retain line charging and full complex off-nominal tap
    semantics.  They therefore apply equally to a line and to the pi-model
    representation of a transformer.
    """
    if not branch.in_service:
        return 0j, 0j
    z = complex(branch.resistance_pu, branch.reactance_pu)
    if abs(z) < 1e-15:
        raise ValueError(f"Branch {branch.identifier} has zero series impedance")
    y = 1.0 / z
    y_shunt = 0.5j * branch.charging_pu
    tap = branch.complex_tap
    if abs(tap) < 1e-15:
        raise ValueError(f"Branch {branch.identifier} has zero tap")
    i_from = (y + y_shunt) / (tap * np.conj(tap)) * v_from - y / np.conj(tap) * v_to
    i_to = -y / tap * v_from + (y + y_shunt) * v_to
    return complex(i_from), complex(i_to)


def branch_terminal_powers(branch: Branch, v_from: complex, v_to: complex, base_mva: float) -> tuple[complex, complex]:
    """Return complex power leaving both terminals into the branch in MVA."""
    i_from, i_to = branch_terminal_currents(branch, v_from, v_to)
    return v_from * np.conj(i_from) * base_mva, v_to * np.conj(i_to) * base_mva


def ybus(bus_ids: Iterable[int], branches: Iterable[Branch], shunts_pu: dict[int, complex] | None = None) -> np.ndarray:
    """Build the network admittance matrix in the supplied bus order."""
    ids = tuple(int(bus) for bus in bus_ids)
    position = {bus: index for index, bus in enumerate(ids)}
    matrix = np.zeros((len(ids), len(ids)), dtype=complex)
    for branch in branches:
        if not branch.in_service:
            continue
        i, j = position[branch.from_bus], position[branch.to_bus]
        z = complex(branch.resistance_pu, branch.reactance_pu)
        if abs(z) < 1e-15:
            raise ValueError(f"Branch {branch.identifier} has zero series impedance")
        y = 1.0 / z
        y_shunt = 0.5j * branch.charging_pu
        tap = branch.complex_tap
        matrix[i, i] += (y + y_shunt) / (tap * np.conj(tap))
        matrix[i, j] += -y / np.conj(tap)
        matrix[j, i] += -y / tap
        matrix[j, j] += y + y_shunt
    for bus, shunt in (shunts_pu or {}).items():
        matrix[position[int(bus)], position[int(bus)]] += complex(shunt)
    return matrix


def net_injections_mva(bus_ids: Iterable[int], voltages: np.ndarray, branches: Iterable[Branch], base_mva: float, shunts_pu: dict[int, complex] | None = None) -> dict[int, complex]:
    """Return net electrical injections, positive from a bus into the network."""
    ids = tuple(int(bus) for bus in bus_ids)
    vector = np.asarray(voltages, dtype=complex)
    if vector.shape != (len(ids),):
        raise ValueError("Voltage vector does not match supplied bus IDs")
    current = ybus(ids, branches, shunts_pu) @ vector
    apparent = vector * np.conj(current) * base_mva
    return {bus: complex(value) for bus, value in zip(ids, apparent)}


def polar_power_injections(bus_ids: Iterable[int], magnitudes: np.ndarray, angles_rad: np.ndarray, admittance: np.ndarray, base_mva: float) -> dict[int, complex]:
    """Evaluate the polar AC power-flow identities for an independent check."""
    ids = tuple(int(bus) for bus in bus_ids)
    magnitudes = np.asarray(magnitudes, dtype=float)
    angles_rad = np.asarray(angles_rad, dtype=float)
    if magnitudes.shape != (len(ids),) or angles_rad.shape != (len(ids),):
        raise ValueError("Polar state does not match supplied bus IDs")
    conductance = np.real(admittance)
    susceptance = np.imag(admittance)
    values: dict[int, complex] = {}
    for i, bus in enumerate(ids):
        difference = angles_rad[i] - angles_rad
        active = magnitudes[i] * np.sum(magnitudes * (conductance[i] * np.cos(difference) + susceptance[i] * np.sin(difference)))
        reactive = magnitudes[i] * np.sum(magnitudes * (conductance[i] * np.sin(difference) - susceptance[i] * np.cos(difference)))
        values[bus] = complex(active * base_mva, reactive * base_mva)
    return values


def gauge_aligned_angles(angles_rad: np.ndarray, reference_index: int) -> np.ndarray:
    """Remove the global voltage-angle gauge with one declared reference bus."""
    values = np.asarray(angles_rad, dtype=float)
    return values - values[reference_index]


def degrees(values_rad: np.ndarray) -> np.ndarray:
    """Convert radians to degrees with ordinary NumPy broadcasting."""
    return np.asarray(values_rad, dtype=float) * 180.0 / math.pi
