"""Declarative scenario grid for the comparative POC.

The grid mirrors the journal ablation target distribution.  ``limit`` can be
used by tests and smoke runs; full generation leaves it unset.
"""

from __future__ import annotations

from itertools import cycle

from poc.schema import PMU_BUSES, Scenario

GEN_BUSES = [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
LOAD_BUSES = [3, 4, 7, 8, 12, 15, 16, 18, 20, 21, 23, 24, 25, 26, 27, 28, 29, 31, 39]
BRANCHES = [
    (1, 2),
    (1, 39),
    (2, 3),
    (2, 25),
    (3, 4),
    (3, 18),
    (4, 5),
    (4, 14),
    (5, 6),
    (5, 8),
    (6, 7),
    (6, 11),
    (7, 8),
    (8, 9),
    (9, 39),
    (10, 11),
    (10, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (16, 17),
    (16, 19),
    (16, 21),
    (16, 24),
    (17, 18),
    (17, 27),
    (21, 22),
    (22, 23),
    (23, 24),
    (25, 26),
    (26, 27),
    (26, 28),
    (26, 29),
    (28, 29),
]


def build_scenario_grid(limit: int | None = None) -> list[Scenario]:
    """Build the target synthetic event plan without shuffling time order."""

    scenarios: list[Scenario] = []

    def add(category: str, label: int, bus: int, idx: int, t0: float, duration: float, **params) -> None:
        scenarios.append(
            Scenario(
                scenario_id=f"{category}_{idx:04d}",
                label=label,
                event_bus=bus,
                t_event=t0,
                duration=duration,
                category=category,
                params=params,
            )
        )

    idx = 0
    buses = cycle(range(1, 40))
    impedances = cycle([0.001, 0.01, 0.05, 0.1])
    clear_cycles = cycle([3, 5, 8])
    starts = cycle([5.0, 8.0, 12.0, 18.0, 24.0])
    for _ in range(200):
        bus = next(buses)
        cycles = next(clear_cycles)
        add(
            "faults",
            1,
            bus,
            idx,
            next(starts),
            cycles / 60.0,
            fault_impedance=next(impedances),
            clearing_cycles=cycles,
        )
        idx += 1

    for i, branch in zip(range(100), cycle(BRANCHES)):
        add("line_outages", 2, branch[0], i, next(starts), 3.0, line=branch, load_level=["low", "nominal", "high"][i % 3])

    for i, bus in zip(range(200), cycle(GEN_BUSES)):
        delta = [5, 10, 25, 50][i % 4] * (1 if i % 2 else -1)
        add("gen_changes", 3, bus, i, next(starts), 5.0, delta_mw=delta, ramp_rate=["fast", "slow"][i % 2])

    for i, bus in zip(range(200), cycle(LOAD_BUSES)):
        delta = [5, 10, 25, 50][i % 4] * (1 if i % 2 else -1)
        add("load_changes", 4, bus, i, next(starts), 5.0, delta_mw=delta, ramp_rate=["fast", "slow"][i % 2])

    for i, bus in zip(range(100), cycle(PMU_BUSES)):
        add("pmu_dropouts", 5, bus, i, next(starts), [1.0, 5.0, 20.0, 60.0][i % 4], dropout_bus=bus)

    for i in range(100):
        add("cyber_physical", 6, 29, i, next(starts), 5.0, dropout_bus=29, physical_bus=GEN_BUSES[i % len(GEN_BUSES)])

    for i, bus in zip(range(100), cycle(PMU_BUSES)):
        add("bad_data", 7, bus, i, next(starts), 2.0, bad_bus=bus, corruption=["spike", "drift"][i % 2])

    return scenarios[:limit] if limit is not None else scenarios

