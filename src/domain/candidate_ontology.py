"""Canonical location spaces for the SGSMA 2026 IEEE-39 task.

The competition topology stores transmission lines and transformers in one
branch table.  Candidate construction must separate them before training or
scoring.  This module also records the naming conversions needed by the
organizer data, whose generation events are reported at the high-voltage bus
rather than at the low-voltage generator terminal.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import pandas as pd


ALL_BUSES = tuple(f"BUS{bus}" for bus in range(1, 40))
OBSERVED_PMU_BUSES = (2, 5, 6, 10, 19, 22, 29, 39)
PMU_ID_TO_BUS = {1: 39, 2: 29, 3: 10, 4: 22, 5: 19, 6: 2, 7: 5, 8: 6}

# Low-voltage generator terminal -> high-voltage bus used by the organizer's
# location labels.  BUS39 is directly reported as BUS39 in the supplied case.
GENERATOR_DEVICE_TO_REPORT_BUS = {
    30: 2,
    31: 6,
    32: 10,
    33: 19,
    34: 20,
    35: 22,
    36: 23,
    37: 25,
    38: 29,
    39: 39,
}

# Buses with non-zero active or reactive demand in the supplied PSS/E RAW.
LOAD_BUSES = (3, 4, 7, 8, 12, 15, 16, 18, 20, 21, 23, 24, 25, 26, 27, 28, 29, 39)


def canonical_line_label(left: int, right: int) -> str:
    """Return an orientation-independent LINE label."""

    a, b = sorted((int(left), int(right)))
    return f"LINE{a}-{b}"


def _parse_bus_suffix(location: str, prefix: str) -> int:
    match = re.fullmatch(rf"{prefix}(\d+)", str(location).strip().upper())
    if not match:
        raise ValueError(f"Expected {prefix}<bus>, received {location!r}")
    return int(match.group(1))


def canonicalize_line(location: str) -> str:
    match = re.fullmatch(r"LINE(\d+)-(\d+)", str(location).strip().upper())
    if not match:
        raise ValueError(f"Expected LINE<a>-<b>, received {location!r}")
    return canonical_line_label(int(match.group(1)), int(match.group(2)))


@dataclass(frozen=True)
class CandidateOntology:
    """Task-valid candidate sets derived from the supplied topology."""

    transmission_lines: tuple[str, ...]
    transformers: tuple[str, ...]

    @classmethod
    def from_branch_csv(cls, path: str | Path, atol: float = 1e-9) -> "CandidateOntology":
        frame = pd.read_csv(path)
        required = {"from_bus", "to_bus", "tap", "shift_deg"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Branch table is missing columns: {sorted(missing)}")

        line_labels: list[str] = []
        transformer_labels: list[str] = []
        for row in frame.itertuples(index=False):
            label = canonical_line_label(int(row.from_bus), int(row.to_bus))
            is_transformer = abs(float(row.tap) - 1.0) > atol or abs(float(row.shift_deg)) > atol
            (transformer_labels if is_transformer else line_labels).append(label)

        lines = tuple(sorted(set(line_labels), key=_line_sort_key))
        transformers = tuple(sorted(set(transformer_labels), key=_line_sort_key))
        if set(lines).intersection(transformers):
            raise ValueError("Transmission-line and transformer sets overlap")
        return cls(transmission_lines=lines, transformers=transformers)

    @property
    def generator_report_buses(self) -> tuple[str, ...]:
        return tuple(f"BUS{bus}" for bus in GENERATOR_DEVICE_TO_REPORT_BUS.values())

    @property
    def load_buses(self) -> tuple[str, ...]:
        return tuple(f"BUS{bus}" for bus in LOAD_BUSES)

    @property
    def pmu_sites(self) -> tuple[str, ...]:
        return tuple(f"BUS{bus}" for bus in OBSERVED_PMU_BUSES)

    def candidates_for_event(self, event: int, physical_event: int | None = None) -> tuple[str, ...]:
        event = int(event)
        physical = int(physical_event) if physical_event is not None else None
        if event == 0:
            return ("none",)
        if event in {5, 7}:
            return self.pmu_sites
        if event == 6 and physical is not None:
            return self.candidates_for_event(physical)
        if event == 1:
            return ALL_BUSES
        if event == 2:
            return self.transmission_lines
        if event == 3:
            return self.generator_report_buses
        if event == 4:
            return self.load_buses
        if event == 8:
            return ("none",) + ALL_BUSES + self.transmission_lines
        raise ValueError(f"Unsupported event label: {event}")

    def validate(self, event: int, location: str, physical_event: int | None = None) -> bool:
        return str(location) in set(self.candidates_for_event(event, physical_event))

    def audit_fitted_classes(self, event: int, fitted: Iterable[str]) -> dict[str, object]:
        expected = set(self.candidates_for_event(event))
        observed = {
            canonicalize_line(value) if int(event) == 2 else str(value)
            for value in fitted
        }
        missing = sorted(expected - observed, key=_natural_location_key)
        invalid = sorted(observed - expected, key=_natural_location_key)
        covered = sorted(expected & observed, key=_natural_location_key)
        return {
            "event": int(event),
            "expected_count": len(expected),
            "fitted_count": len(observed),
            "covered_count": len(covered),
            "coverage": len(covered) / max(len(expected), 1),
            "covered": covered,
            "missing": missing,
            "invalid": invalid,
        }


def canonicalize_frozen_location(
    event: int,
    location: str,
    *,
    physical_event: int | None = None,
) -> str:
    """Map frozen-model labels to the organizer-facing location convention.

    This conversion fixes one-to-one naming differences only.  It cannot add
    transmission-line classes that were absent during training.
    """

    event = int(event)
    physical = int(physical_event) if physical_event is not None else event
    text = str(location).strip()
    if event == 0 or text.lower() == "none":
        return "none"
    if event in {5, 7}:
        bus = _parse_bus_suffix(text, "PMU") if text.upper().startswith("PMU") else _parse_bus_suffix(text, "BUS")
        return f"BUS{bus}"
    if event == 2 or (event == 6 and physical == 2):
        return canonicalize_line(text)
    if event == 3 or (event == 6 and physical == 3):
        bus = _parse_bus_suffix(text, "BUS")
        return f"BUS{GENERATOR_DEVICE_TO_REPORT_BUS.get(bus, bus)}"
    if text.upper().startswith("BUS"):
        return f"BUS{_parse_bus_suffix(text, 'BUS')}"
    return text


def _line_sort_key(label: str) -> tuple[int, int]:
    match = re.fullmatch(r"LINE(\d+)-(\d+)", label)
    if not match:
        return (10_000, 10_000)
    return int(match.group(1)), int(match.group(2))


def _natural_location_key(label: str) -> tuple[str, int, int]:
    if label == "none":
        return ("", 0, 0)
    if label.startswith("LINE"):
        left, right = _line_sort_key(label)
        return ("LINE", left, right)
    match = re.search(r"(\d+)$", label)
    return (re.sub(r"\d+$", "", label), int(match.group(1)) if match else 0, 0)

