from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class PhysicalBranch:
    from_bus: int
    to_bus: int
    resistance: float
    reactance: float
    charging: float
    tap: float = 1.0
    shift_deg: float = 0.0


def _clean_record(line: str) -> list[str]:
    body = line.split("/", 1)[0].strip()
    return [part.strip().strip("'\"") for part in body.split(",")]


def _section(lines: list[str], start_marker: str, end_marker: str) -> list[str]:
    start = None
    for idx, line in enumerate(lines):
        if start_marker in line.lower():
            start = idx + 1
            break
    if start is None:
        return []
    out = []
    for line in lines[start:]:
        low = line.lower()
        if end_marker in low:
            break
        if line.strip() and not line.lstrip().startswith("0"):
            out.append(line)
    return out


def _parse_raw_physical_bus_map(raw_path: Path) -> dict[int, int]:
    lines = raw_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    mapping: dict[int, int] = {}
    for line in lines[3:]:
        if line.lstrip().startswith("0"):
            break
        parts = _clean_record(line)
        if len(parts) < 2:
            continue
        try:
            internal_bus = int(parts[0])
        except ValueError:
            continue
        match = re.search(r"BUS\s*([0-9]+)", parts[1].upper())
        if match:
            mapping[internal_bus] = int(match.group(1))
    return mapping


def _parse_branch_section_physical(raw_path: Path) -> list[PhysicalBranch]:
    lines = raw_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    bus_map = _parse_raw_physical_bus_map(raw_path)
    raw_branches = _section(lines, "starting branch section", "end branch section")
    branches: list[PhysicalBranch] = []
    for line in raw_branches:
        parts = _clean_record(line)
        if len(parts) < 6:
            continue
        try:
            from_internal = int(parts[0])
            to_internal = int(parts[1])
            r = float(parts[3])
            x = float(parts[4])
            b = float(parts[5])
            tap = float(parts[9]) if len(parts) > 9 and parts[9] else 0.0
            shift = float(parts[10]) if len(parts) > 10 and parts[10] else 0.0
        except ValueError:
            continue
        if from_internal not in bus_map or to_internal not in bus_map:
            continue
        if abs(r) < 1e-12 and abs(x) < 1e-12:
            continue
        branches.append(
            PhysicalBranch(
                from_bus=bus_map[from_internal],
                to_bus=bus_map[to_internal],
                resistance=r,
                reactance=x,
                charging=b,
                tap=tap if abs(tap) > 1e-12 else 1.0,
                shift_deg=shift,
            )
        )
    return branches


def _build_full_ybus(bus_numbers: list[int], branches: list[PhysicalBranch]) -> np.ndarray:
    index = {int(bus): idx for idx, bus in enumerate(bus_numbers)}
    ybus = np.zeros((len(bus_numbers), len(bus_numbers)), dtype=complex)
    for branch in branches:
        if branch.from_bus not in index or branch.to_bus not in index:
            continue
        i = index[branch.from_bus]
        j = index[branch.to_bus]
        z = complex(branch.resistance, branch.reactance)
        if abs(z) < 1e-12:
            continue
        y = 1.0 / z
        b_shunt = 1j * branch.charging / 2.0
        tap = branch.tap if abs(branch.tap) > 1e-12 else 1.0
        phase = np.deg2rad(branch.shift_deg)
        a = tap * np.exp(1j * phase)
        ybus[i, i] += (y + b_shunt) / (a * np.conj(a))
        ybus[i, j] -= y / np.conj(a)
        ybus[j, i] -= y / a
        ybus[j, j] += y + b_shunt
    return ybus
