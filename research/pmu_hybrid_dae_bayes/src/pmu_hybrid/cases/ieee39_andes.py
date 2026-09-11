"""Feature-detected ANDES adapter for the bundled dynamic IEEE-39 case."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import numpy as np

from pmu_hybrid.physics.network import Branch, net_injections_mva


@dataclass(frozen=True)
class AndesStaticSolution:
    bus_ids: tuple[int, ...]
    voltage_pu: np.ndarray
    angle_rad: np.ndarray
    branches: tuple[Branch, ...]
    injections_mva: dict[int, complex]
    base_mva: float
    model_counts: dict[str, int]


def _value(model: Any, name: str, default: float | np.ndarray) -> np.ndarray:
    """Read a public ANDES parameter/service array with an explicit fallback."""
    if not hasattr(model, name):
        return np.asarray(default)
    candidate = getattr(model, name)
    if not hasattr(candidate, "v"):
        return np.asarray(default)
    return np.asarray(candidate.v)


def load_native_system() -> Any:
    """Load, but do not solve, the source case with its native static inputs."""
    os.environ.setdefault("SYMPY_GROUND_TYPES", "python")
    import andes

    return andes.load(
        andes.get_case("ieee39/ieee39_full.xlsx"),
        setup=False,
        no_output=True,
    )


def solve_static() -> AndesStaticSolution:
    """Load the native ANDES case and solve PFlow through its stable API."""
    system = load_native_system()
    system.setup()
    if system.PFlow.run() is False:
        raise RuntimeError("ANDES IEEE-39 PFlow returned False")

    bus_ids = tuple(int(bus) for bus in _value(system.Bus, "idx", []))
    voltage_pu = _value(system.Bus, "v", np.ones(len(bus_ids))).astype(float)
    angle_rad = _value(system.Bus, "a", np.zeros(len(bus_ids))).astype(float)
    if len(bus_ids) != 39 or voltage_pu.shape != (39,) or angle_rad.shape != (39,):
        raise RuntimeError("Unexpected ANDES Bus result layout")

    line = system.Line
    from_buses = _value(line, "bus1", []).astype(int)
    to_buses = _value(line, "bus2", []).astype(int)
    resistances = _value(line, "r", np.zeros(line.n)).astype(float)
    reactances = _value(line, "x", np.zeros(line.n)).astype(float)
    charging = _value(line, "b", np.zeros(line.n)).astype(float)
    taps = _value(line, "tap", np.ones(line.n)).astype(float)
    shifts = _value(line, "phi", np.zeros(line.n)).astype(float)
    enabled = _value(line, "u", np.ones(line.n)).astype(bool)
    identifiers = tuple(str(item) for item in _value(line, "idx", np.arange(line.n)))
    branches = tuple(
        Branch(
            identifier=identifier,
            from_bus=int(from_bus),
            to_bus=int(to_bus),
            resistance_pu=float(r),
            reactance_pu=float(x),
            charging_pu=float(b),
            tap=float(tap) if abs(float(tap)) > 1e-15 else 1.0,
            shift_rad=float(shift),
            in_service=bool(active),
        )
        for identifier, from_bus, to_bus, r, x, b, tap, shift, active in zip(
            identifiers, from_buses, to_buses, resistances, reactances, charging, taps, shifts, enabled,
        )
    )
    base_mva = float(getattr(system.config, "mva", 100.0))
    voltage = voltage_pu * np.exp(1j * angle_rad)
    injections = net_injections_mva(bus_ids, voltage, branches, base_mva)
    return AndesStaticSolution(
        bus_ids=bus_ids,
        voltage_pu=voltage_pu,
        angle_rad=angle_rad,
        branches=branches,
        injections_mva=injections,
        base_mva=base_mva,
        model_counts={
            name: int(getattr(system, name).n)
            for name in ("GENROU", "TGOV1N", "IEEEX1", "IEEEST", "BusFreq")
            if hasattr(system, name)
        },
    )
