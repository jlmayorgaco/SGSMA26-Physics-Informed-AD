"""Typed models for M5 metadata/model layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class BusRecord:
    bus_id_numeric: int | None
    bus_label_original: str
    bus_label_canonical: str
    kv_ll: float | None
    bus_type_raw: int | None
    bus_type_name: str
    v_pu: float | None
    theta_deg: float | None
    pmu_id: int | None
    has_pmu: bool


@dataclass(slots=True)
class PMUMapEntry:
    pmu_id: int
    bus_label_canonical: str


@dataclass(slots=True)
class RawBranchRecord:
    from_bus: str
    to_bus: str
    circuit_id: str
    r_pu: float
    x_pu: float
    b_pu: float
    tap_ratio: float
    phase_shift_deg: float
    status: int


@dataclass(slots=True)
class RawTransformerRecord:
    from_bus: str
    to_bus: str
    circuit_id: str
    r_pu: float
    x_pu: float
    tap_ratio: float
    phase_shift_deg: float
    status: int


@dataclass(slots=True)
class RawGeneratorRecord:
    bus_label_canonical: str
    gen_id: str
    p_mw: float
    q_mvar: float
    status: int


@dataclass(slots=True)
class RawLoadRecord:
    bus_label_canonical: str
    load_id: str
    p_mw: float
    q_mvar: float
    status: int


@dataclass(slots=True)
class MatrixMetadata:
    bus_order: list[str]
    shape: tuple[int, int]
    invertible: bool | None = None
    condition_number: float | None = None
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SystemMetadata:
    system_name: str
    base_mva: float
    rated_frequency_hz: float
    bus_count: int
    generator_count: int
    line_count: int
    transformer_count: int
    buses: list[BusRecord]
    pmu_bus_ids: list[str]
    pmu_map: list[PMUMapEntry]
    raw_sections_summary: dict[str, Any]
    ybus_metadata: MatrixMetadata | None = None
    zbus_metadata: MatrixMetadata | None = None
