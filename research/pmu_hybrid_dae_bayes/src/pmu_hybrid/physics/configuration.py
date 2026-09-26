"""Sparse physical configuration jumps; events are never categorical labels."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Mapping


class EventFamily(str, Enum):
    FAULT = "FAULT"
    LINE = "LINE"
    GENERATION = "GENERATION"
    LOAD = "LOAD"
    IBR = "IBR"


@dataclass(frozen=True)
class PhysicalConfiguration:
    """The persistent physical state q changed by sparse, interpretable jumps."""

    line_status: Mapping[str, bool] = field(default_factory=dict)
    fault_shunts_pu: Mapping[str, complex] = field(default_factory=dict)
    generator_setpoint_delta_pu: Mapping[str, float] = field(default_factory=dict)
    load_setpoint_delta_pu: Mapping[str, float] = field(default_factory=dict)
    ibr_mode_or_status: Mapping[str, str] = field(default_factory=dict)

    def apply(self, jump: "ConfigurationJump") -> "PhysicalConfiguration":
        """Apply one active physical jump without changing unrelated sources."""
        lines = dict(self.line_status)
        faults = dict(self.fault_shunts_pu)
        generation = dict(self.generator_setpoint_delta_pu)
        loads = dict(self.load_setpoint_delta_pu)
        ibr = dict(self.ibr_mode_or_status)
        if jump.family is EventFamily.LINE:
            lines[jump.source] = bool(jump.line_in_service)
        elif jump.family is EventFamily.FAULT:
            faults[jump.event_id] = complex(jump.amplitude)
        elif jump.family is EventFamily.GENERATION:
            generation[jump.source] = generation.get(jump.source, 0.0) + float(jump.amplitude)
        elif jump.family is EventFamily.LOAD:
            loads[jump.source] = loads.get(jump.source, 0.0) + float(jump.amplitude)
        elif jump.family is EventFamily.IBR:
            if jump.ibr_mode is None:
                raise ValueError("IBR jump requires an explicit mode or status")
            ibr[jump.source] = jump.ibr_mode
        else:  # defensive future-proofing for new enumerated mechanisms
            raise ValueError(f"Unsupported physical event family {jump.family!r}")
        return replace(
            self, line_status=lines, fault_shunts_pu=faults,
            generator_setpoint_delta_pu=generation, load_setpoint_delta_pu=loads,
            ibr_mode_or_status=ibr,
        )

    def clear_fault(self, event_id: str) -> "PhysicalConfiguration":
        """Clear one transient fault while retaining every persistent jump."""
        faults = dict(self.fault_shunts_pu)
        faults.pop(event_id, None)
        return replace(self, fault_shunts_pu=faults)


@dataclass(frozen=True)
class ConfigurationJump:
    """A sparse event support element with source, family, onset and amplitude."""

    event_id: str
    family: EventFamily
    source: str
    onset_s: float
    amplitude: float | complex = 0.0
    duration_s: float | None = None
    line_in_service: bool = False
    ibr_mode: str | None = None

    def __post_init__(self) -> None:
        if not self.event_id or not self.source:
            raise ValueError("event_id and source must be non-empty")
        if self.onset_s < 0:
            raise ValueError("onset_s must be non-negative")
        if self.duration_s is not None and self.duration_s <= 0:
            raise ValueError("duration_s must be positive when provided")
        if self.family is EventFamily.FAULT and self.duration_s is None:
            raise ValueError("Faults must declare an explicit clearing duration")
        if self.family is EventFamily.LINE and self.duration_s is not None:
            raise ValueError("A line configuration jump is persistent; use a second jump to reclose")
        if self.family in (EventFamily.GENERATION, EventFamily.LOAD) and abs(complex(self.amplitude)) < 1e-12:
            raise ValueError("An active generation/load support cannot have zero amplitude")

    @property
    def clear_time_s(self) -> float | None:
        return None if self.duration_s is None else self.onset_s + self.duration_s

    def to_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "family": self.family.value,
            "source": self.source,
            "onset_s": self.onset_s,
            "amplitude_real_pu": float(complex(self.amplitude).real),
            "amplitude_imag_pu": float(complex(self.amplitude).imag),
            "duration_s": self.duration_s,
            "line_in_service": self.line_in_service,
            "ibr_mode": self.ibr_mode,
        }
