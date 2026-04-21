"""Scenario templates for M9."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def _physical(event_id: str, label: int, event_type: str, subtype: str, start: float, duration: float, **kwargs: Any) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "event_label": label,
        "event_type": event_type,
        "subtype": subtype,
        "start_time_s": float(start),
        "end_time_s": float(start + duration),
        "duration_s": float(duration),
        "severity": float(kwargs.pop("severity", 1.0)),
        "shape": kwargs.pop("shape", "step"),
        **kwargs,
    }


def _cyber(event_id: str, label: int, event_type: str, subtype: str, start: float, duration: float, pmus: list[str], channels: list[str] | None = None, **params: Any) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "event_label": label,
        "event_type": event_type,
        "subtype": subtype,
        "start_time_s": float(start),
        "end_time_s": float(start + duration),
        "duration_s": float(duration),
        "target_pmus": pmus,
        "target_channels": channels or ["ALL"],
        "params": params,
    }


TEMPLATES: dict[str, dict[str, Any]] = {
    "TEMPLATE_EVENT0_NORMAL": {
        "name": "TEMPLATE_EVENT0_NORMAL",
        "description": "Quiet normal operation with PMU noise only.",
        "duration_s": 12.0,
        "physical_events": [],
        "cyber_events": [],
    },
    "TEMPLATE_EVENT1_FAULT": {
        "name": "TEMPLATE_EVENT1_FAULT",
        "description": "Three-phase bus fault at BUS39.",
        "duration_s": 12.0,
        "physical_events": [_physical("P001", 1, "fault", "3LG", 4.0, 0.22, target_bus="BUS39", target_line=None, severity=0.9, andes_params={"fault_type": "BusFault", "rf": 0.0, "xf": 0.0})],
        "cyber_events": [],
    },
    "TEMPLATE_EVENT2_LINE_OUTAGE": {
        "name": "TEMPLATE_EVENT2_LINE_OUTAGE",
        "description": "Line outage on 24-23.",
        "duration_s": 12.0,
        "physical_events": [_physical("P001", 2, "line_outage", "branch_trip", 4.0, 4.0, target_bus=None, target_line="BUS24-BUS23", severity=0.55, andes_params={"from_bus": "BUS24", "to_bus": "BUS23", "action": "trip"})],
        "cyber_events": [],
    },
    "TEMPLATE_EVENT3_GENERATION_CHANGE": {
        "name": "TEMPLATE_EVENT3_GENERATION_CHANGE",
        "description": "Generation step/down-ramp associated with BUS2 PMU area.",
        "duration_s": 12.0,
        "physical_events": [_physical("P001", 3, "generation_change", "generation_drop", 4.0, 4.0, target_bus="BUS2", target_line=None, severity=0.45, shape="ramp", andes_params={"p_change_fraction": -0.15})],
        "cyber_events": [],
    },
    "TEMPLATE_EVENT4_LOAD_CHANGE": {
        "name": "TEMPLATE_EVENT4_LOAD_CHANGE",
        "description": "Load change/drop at BUS7.",
        "duration_s": 12.0,
        "physical_events": [_physical("P001", 4, "load_change", "load_drop", 4.0, 4.5, target_bus="BUS7", target_line=None, severity=0.40, shape="ramp", andes_params={"p_change_fraction": -0.20})],
        "cyber_events": [],
    },
    "TEMPLATE_EVENT5_MISSING_ONLY": {
        "name": "TEMPLATE_EVENT5_MISSING_ONLY",
        "description": "Missing data at BUS29 only.",
        "duration_s": 12.0,
        "physical_events": [],
        "cyber_events": [_cyber("C001", 5, "missing_data", "full_dropout", 4.0, 1.0, ["BUS29"])],
    },
    "TEMPLATE_EVENT6_CONCURRENT_MISSING_PLUS_PHYSICAL": {
        "name": "TEMPLATE_EVENT6_CONCURRENT_MISSING_PLUS_PHYSICAL",
        "description": "BUS29 missing data overlaps a generation change.",
        "duration_s": 12.0,
        "physical_events": [_physical("P001", 3, "generation_change", "generation_drop", 4.0, 3.0, target_bus="BUS2", target_line=None, severity=0.50, shape="ramp", andes_params={"p_change_fraction": -0.18})],
        "cyber_events": [_cyber("C001", 6, "missing_data", "full_dropout", 4.25, 1.5, ["BUS29"])],
    },
    "TEMPLATE_EVENT7_BAD_DATA": {
        "name": "TEMPLATE_EVENT7_BAD_DATA",
        "description": "Bad-data spike and bias on BUS10 while frames remain present.",
        "duration_s": 12.0,
        "physical_events": [],
        "cyber_events": [_cyber("C001", 7, "bad_data", "spike", 4.0, 1.0, ["BUS10"], ["VA_MAG", "IA_MAG"], amplitude=8.0), _cyber("C002", 7, "bad_data", "bias", 5.0, 2.0, ["BUS10"], ["VA_ANG"], bias=5.0)],
    },
    "TEMPLATE_EVENT8_UNKNOWN_COMPOSITE": {
        "name": "TEMPLATE_EVENT8_UNKNOWN_COMPOSITE",
        "description": "Ambiguous composite event combining physical and multiple cyber corruptions.",
        "duration_s": 12.0,
        "physical_events": [_physical("P001", 8, "unknown_composite", "weak_physical_disturbance", 4.0, 2.0, target_bus="BUS19", target_line=None, severity=0.25, shape="step", andes_params={})],
        "cyber_events": [_cyber("C001", 8, "bad_data", "angle_wrap_corruption", 4.2, 1.2, ["BUS5"], ["VA_ANG", "VB_ANG", "VC_ANG"]), _cyber("C002", 8, "bad_data", "gain_error", 4.4, 1.8, ["BUS6"], ["IA_MAG", "IB_MAG", "IC_MAG"], gain=1.35)],
    },
    "TEMPLATE_OFFICIAL_STYLE_MULTI_EVENT": {
        "name": "TEMPLATE_OFFICIAL_STYLE_MULTI_EVENT",
        "description": "Competition-visible pattern: BUS29 missing data, BUS39 3LG fault, line outage 24-23, generation change at BUS2, load change at BUS7, and concurrent missing+physical.",
        "duration_s": 24.0,
        "physical_events": [
            _physical("P001", 1, "fault", "3LG", 5.0, 0.22, target_bus="BUS39", target_line=None, severity=0.9, andes_params={"fault_type": "BusFault", "rf": 0.0, "xf": 0.0}),
            _physical("P002", 2, "line_outage", "branch_trip", 8.0, 3.0, target_bus=None, target_line="BUS24-BUS23", severity=0.55, andes_params={"from_bus": "BUS24", "to_bus": "BUS23", "action": "trip"}),
            _physical("P003", 3, "generation_change", "generation_drop", 13.0, 3.0, target_bus="BUS2", target_line=None, severity=0.50, shape="ramp", andes_params={"p_change_fraction": -0.18}),
            _physical("P004", 3, "generation_change", "generation_drop", 17.0, 2.5, target_bus="BUS2", target_line=None, severity=0.35, shape="ramp", andes_params={"p_change_fraction": -0.12}),
            _physical("P005", 4, "load_change", "load_drop", 20.0, 2.5, target_bus="BUS7", target_line=None, severity=0.45, shape="ramp", andes_params={"p_change_fraction": -0.20}),
        ],
        "cyber_events": [
            _cyber("C001", 5, "missing_data", "full_dropout", 2.0, 1.0, ["BUS29"]),
            _cyber("C002", 5, "missing_data", "full_dropout", 11.0, 1.0, ["BUS29"]),
            _cyber("C003", 6, "missing_data", "full_dropout", 13.2, 1.5, ["BUS29"]),
        ],
    },
}

ALIASES = {
    "EVENT0_NORMAL": "TEMPLATE_EVENT0_NORMAL",
    "EVENT1_FAULT": "TEMPLATE_EVENT1_FAULT",
    "EVENT2_LINE_OUTAGE": "TEMPLATE_EVENT2_LINE_OUTAGE",
    "EVENT3_GENERATION_CHANGE": "TEMPLATE_EVENT3_GENERATION_CHANGE",
    "EVENT4_LOAD_CHANGE": "TEMPLATE_EVENT4_LOAD_CHANGE",
    "EVENT5_MISSING_ONLY": "TEMPLATE_EVENT5_MISSING_ONLY",
    "EVENT6_CONCURRENT_MISSING_PLUS_PHYSICAL": "TEMPLATE_EVENT6_CONCURRENT_MISSING_PLUS_PHYSICAL",
    "EVENT7_BAD_DATA": "TEMPLATE_EVENT7_BAD_DATA",
    "EVENT8_UNKNOWN_COMPOSITE": "TEMPLATE_EVENT8_UNKNOWN_COMPOSITE",
    "OFFICIAL_STYLE_MULTI_EVENT": "TEMPLATE_OFFICIAL_STYLE_MULTI_EVENT",
    "SIM0001": "TEMPLATE_OFFICIAL_STYLE_MULTI_EVENT",
}


def list_templates() -> list[str]:
    return list(TEMPLATES.keys())


def get_template(name: str) -> dict[str, Any]:
    key = ALIASES.get(str(name).upper(), str(name))
    if key not in TEMPLATES:
        raise KeyError(f"Unknown M9 scenario template: {name!r}. Available: {', '.join(list_templates())}")
    return deepcopy(TEMPLATES[key])
