"""Configuration contract for m4 type-0 (normal/no-fault) simulation."""

from __future__ import annotations

from dataclasses import dataclass, field


DEFAULT_PMU_BUSES = ["39", "29", "10", "22", "19", "2", "5", "6"]


@dataclass(frozen=True)
class Type0Config:
    """Runtime configuration for normal-operation type0 scenario."""

    sim_tf: float = 10.1
    sim_tstep: float = 1.0 / 30.0
    event_label_mode: str = "all_zero"
    data_present_default: int = 1
    pmu_buses: list[str] = field(default_factory=lambda: DEFAULT_PMU_BUSES.copy())

    estimation_mode: str = "ybus_temporal_regularized"
    estimation_temporal_lambda: float = 5e-2
    estimation_prior_lambda: float = 1e-3
    estimation_use_positive_sequence: bool = True
    current_estimation_mode: str = "nodal_injection_from_ybus"

    run_name_template: str = "SIM0001_NORMAL_Type0_NEWARCH"
    base_output_dir: str = "output"

    def run_folder_name(self) -> str:
        return self.run_name_template
