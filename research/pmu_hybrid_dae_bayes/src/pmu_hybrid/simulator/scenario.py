"""Deterministic physical scenario declarations and configuration trajectories."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable

import numpy as np

from pmu_hybrid.physics.configuration import ConfigurationJump, PhysicalConfiguration
from pmu_hybrid.utils.seeds import derive_seed


@dataclass(frozen=True)
class PMUDropout:
    """An integrity event affecting the observation mask, never q."""

    pmu_bus: int
    start_s: float
    end_s: float

    def __post_init__(self) -> None:
        if self.start_s < 0 or self.end_s <= self.start_s:
            raise ValueError("A PMU dropout requires 0 <= start_s < end_s")

    def to_dict(self) -> dict[str, object]:
        return {"pmu_bus": self.pmu_bus, "start_s": self.start_s, "end_s": self.end_s}


@dataclass(frozen=True)
class ScenarioSpec:
    """Leakage-safe unit of splitting, simulation, resumption and evaluation."""

    scenario_id: str
    seed: int
    operating_point_id: str
    model_seed: int
    duration_s: float
    sample_rate_hz: float
    physical_jumps: tuple[ConfigurationJump, ...] = ()
    pmu_dropouts: tuple[PMUDropout, ...] = ()
    simulator_backend: str = "andes"

    def __post_init__(self) -> None:
        if not self.scenario_id or self.duration_s <= 0 or self.sample_rate_hz <= 0:
            raise ValueError("Scenario requires ID, positive duration and positive sample rate")
        identifiers = [jump.event_id for jump in self.physical_jumps]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Physical event IDs must be unique")
        if any(jump.onset_s > self.duration_s for jump in self.physical_jumps):
            raise ValueError("Event onset must lie inside the scenario")
        if any(dropout.end_s > self.duration_s for dropout in self.pmu_dropouts):
            raise ValueError("PMU dropout must lie inside the scenario")

    @property
    def scenario_seed(self) -> int:
        return derive_seed(self.seed, f"scenario/{self.scenario_id}/{self.operating_point_id}/{self.model_seed}")

    @property
    def time_s(self) -> np.ndarray:
        frame_count = int(round(self.duration_s * self.sample_rate_hz)) + 1
        return np.arange(frame_count, dtype=float) / self.sample_rate_hz

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "scenario_seed": self.scenario_seed,
            "operating_point_id": self.operating_point_id,
            "model_seed": self.model_seed,
            "duration_s": self.duration_s,
            "sample_rate_hz": self.sample_rate_hz,
            "simulator_backend": self.simulator_backend,
            "physical_jumps": [jump.to_dict() for jump in self.physical_jumps],
            "pmu_dropouts": [dropout.to_dict() for dropout in self.pmu_dropouts],
            "split_unit": "scenario_operating_point_model_seed",
            "event_labels_used_as_estimator_input": False,
        }

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def configuration_trajectory(spec: ScenarioSpec, times_s: Iterable[float] | None = None) -> tuple[PhysicalConfiguration, ...]:
    """Apply simultaneous/sequential q jumps causally over a declared frame grid."""
    times = spec.time_s if times_s is None else np.asarray(tuple(times_s), dtype=float)
    current = PhysicalConfiguration()
    active: set[str] = set()
    activated: set[str] = set()
    history: list[PhysicalConfiguration] = []
    ordered = tuple(sorted(spec.physical_jumps, key=lambda jump: (jump.onset_s, jump.event_id)))
    for time in times:
        for jump in ordered:
            if jump.event_id not in activated and jump.onset_s <= time:
                current = current.apply(jump)
                active.add(jump.event_id)
                activated.add(jump.event_id)
        for jump in ordered:
            if jump.family.value == "FAULT" and jump.clear_time_s is not None and jump.clear_time_s <= time and jump.event_id in active:
                current = current.clear_fault(jump.event_id)
                active.remove(jump.event_id)
        history.append(current)
    return tuple(history)


def data_present_mask(spec: ScenarioSpec, pmu_buses: Iterable[int]) -> np.ndarray:
    """Return the causal availability mask after explicitly declared PMU outages."""
    buses = tuple(int(bus) for bus in pmu_buses)
    mask = np.ones((len(spec.time_s), len(buses)), dtype=bool)
    positions = {bus: index for index, bus in enumerate(buses)}
    for dropout in spec.pmu_dropouts:
        if dropout.pmu_bus not in positions:
            raise ValueError(f"Dropout references unobserved PMU bus {dropout.pmu_bus}")
        active = (spec.time_s >= dropout.start_s) & (spec.time_s < dropout.end_s)
        mask[active, positions[dropout.pmu_bus]] = False
    return mask
