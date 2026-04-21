"""Hybrid dynamic state layout for M8 estimators."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class HybridStateLayout:
    """Index layout for hybrid state vector."""

    bus_order: list[str]
    generator_buses: list[str]
    voltage_dim: int
    dynamic_dim: int
    total_dim: int
    vr_offset: int
    vi_offset: int
    delta_offset: int
    omega_offset: int
    bus_to_index: dict[str, int]
    gen_to_index: dict[str, int]


def build_hybrid_state_layout(bus_order: list[str], generator_buses: list[str]) -> HybridStateLayout:
    """Build deterministic state indexing for [Vr, Vi, delta, omega]."""
    buses = list(bus_order)
    gens = [g for g in generator_buses if g in buses]
    n_bus = len(buses)
    n_gen = len(gens)
    vr_offset = 0
    vi_offset = n_bus
    delta_offset = 2 * n_bus
    omega_offset = 2 * n_bus + n_gen
    return HybridStateLayout(
        bus_order=buses,
        generator_buses=gens,
        voltage_dim=2 * n_bus,
        dynamic_dim=2 * n_gen,
        total_dim=2 * n_bus + 2 * n_gen,
        vr_offset=vr_offset,
        vi_offset=vi_offset,
        delta_offset=delta_offset,
        omega_offset=omega_offset,
        bus_to_index={b: i for i, b in enumerate(buses)},
        gen_to_index={b: i for i, b in enumerate(gens)},
    )


def hybrid_state_from_voltage_prior(
    layout: HybridStateLayout,
    voltage_prior: list[complex] | tuple[complex, ...],
) -> list[float]:
    """Initialize hybrid state from complex voltage prior."""
    if len(voltage_prior) != len(layout.bus_order):
        raise ValueError("Voltage prior length does not match bus count.")
    x = [0.0] * layout.total_dim
    for i, z in enumerate(voltage_prior):
        x[layout.vr_offset + i] = float(z.real)
        x[layout.vi_offset + i] = float(z.imag)
    return x

