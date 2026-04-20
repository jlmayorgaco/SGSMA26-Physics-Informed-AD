"""Configuration contract for m4 type-1 fault simulation."""

from __future__ import annotations

from dataclasses import dataclass, field


DEFAULT_PMU_BUSES = ["39", "29", "10", "22", "19", "2", "5", "6"]


def validate_fault_bus(fault_bus: str) -> str:
    """Validate fault bus identifier in IEEE39 range."""
    bus = str(fault_bus)
    if bus not in {str(i) for i in range(1, 40)}:
        raise ValueError(f"FAULT_BUS must be in 1..39, got {fault_bus!r}")
    return bus


@dataclass(frozen=True)
class Type1Config:
    """m4 parity-first runtime configuration."""

    fault_bus: str = "39"
    pre_fault_s: float = 5.0
    fault_start_s: float = 5.0
    fault_duration_s: float = 0.10
    post_fault_s: float = 5.0
    event_label_mode: str = "windowed"
    sim_tstep: float = 1.0 / 30.0
    pmu_buses: list[str] = field(default_factory=lambda: DEFAULT_PMU_BUSES.copy())

    estimation_mode: str = "ybus_temporal_regularized"
    estimation_temporal_lambda: float = 5e-2
    estimation_prior_lambda: float = 1e-3
    estimation_use_positive_sequence: bool = True
    current_estimation_mode: str = "nodal_injection_from_ybus"

    run_name_template: str = "SIM0001_BUS{fault_bus}_Event1"
    base_output_dir: str = "output"

    def normalized_fault_bus(self) -> str:
        return validate_fault_bus(self.fault_bus)

    def run_folder_name(self) -> str:
        return self.run_name_template.format(fault_bus=self.normalized_fault_bus())
