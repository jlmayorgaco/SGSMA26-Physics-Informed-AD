"""Typed models for M6 quasi-static PMU state estimation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class NetworkModel:
    """Canonical network model used by the estimator."""

    ybus: np.ndarray
    bus_order: list[str]
    bus_kv_map: dict[str, float]
    base_mva: float = 100.0
    rated_frequency_hz: float = 60.0
    slack_bus: str | None = None


@dataclass(slots=True)
class EstimationConfig:
    """Configurable regularization and measurement weights."""

    lambda_reg: float = 5e-2
    mu_reg: float = 1e-3
    voltage_weight: float = 1.0
    current_weight: float = 0.25
    pseudo_weight: float = 0.01
    condition_guard: float = 1e-9


@dataclass(slots=True)
class EstimationDiagnostics:
    """Per-frame solver diagnostics."""

    timestamp: float
    frame_index: int
    measurement_count: int
    voltage_measurement_count: int
    current_measurement_count: int
    residual_norm: float
    matrix_condition_number: float
    used_previous_state: bool
    n_valid_pmus: int = 0
    n_pmus_used: int = 0
    pmu_buses_used: list[str] = field(default_factory=list)
    pmu_buses_excluded: list[str] = field(default_factory=list)
    dropped_reasons: dict[str, str] = field(default_factory=dict)
    solver_status: str = "ok"
    solver_message: str = ""
    data_term_value: float = 0.0
    temporal_prior_term_value: float = 0.0
    loadflow_prior_term_value: float = 0.0
    total_objective_value: float = 0.0


@dataclass(slots=True)
class EstimationResult:
    """Time-series estimation result container."""

    timestamps: np.ndarray
    bus_order: list[str]
    voltage_estimates_pu: np.ndarray
    diagnostics: list[EstimationDiagnostics] = field(default_factory=list)
    data_present_used: np.ndarray | None = None
