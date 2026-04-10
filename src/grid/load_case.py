"""Parse the IEEE 39-bus .raw file and build Ybus/Zbus.

pandapower 3.x dropped PSS/E RAW import, so we parse the file directly.
Format: PSS/E version 30 RAW file with sections:
  BUS | LOAD | SOURCE (generators) | BRANCH

Bus numbers in the .raw are PSS/E external IDs (1-39), which map to bus NAMES
like 'BUS1', 'BUS10', etc. — not the same numbering as the competition CSVs.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# PMU buses: competition CSV name → PSS/E bus number (from .raw parsing)
# Populated dynamically after parsing; used for validation.
PMU_BUS_NAMES = [2, 5, 6, 10, 19, 22, 29, 39]  # competition BUSk numbers

# Spec Table 1 reference voltages at PMU bus NAMES (p.u., degrees)
# Source: columns 9-10 of bus section in the actual .raw file
SPEC_TABLE1: dict[int, tuple[float, float]] = {
    2:  (1.0487, -5.75),
    5:  (1.0053, -8.61),
    6:  (1.0077, -7.95),
    10: (1.0172, -5.43),
    19: (1.0499, -1.02),
    22: (1.0498,  0.67),
    29: (1.0499,  0.75),
    39: (1.0300, -10.05),
}


@dataclass
class BusRecord:
    psse_num: int      # PSS/E bus number (1-39 in this file)
    name: str          # e.g. 'BUS2', 'BUS10'
    base_kv: float
    bus_type: int      # 1=PQ, 2=PV, 3=slack
    vm_pu: float       # base-case voltage magnitude
    va_deg: float      # base-case voltage angle


@dataclass
class BranchRecord:
    from_bus: int      # PSS/E bus number
    to_bus: int
    r: float           # resistance p.u.
    x: float           # reactance p.u.
    b: float           # shunt susceptance p.u.
    tap: float         # transformer tap (0 or 1 = line)
    phase_deg: float   # phase shift degrees


@dataclass
class GridCase:
    buses: list[BusRecord]
    branches: list[BranchRecord]
    gen_psse_nums: list[int]           # PSS/E bus numbers with generators

    # Derived quantities (populated by build_matrices)
    psse_to_idx: dict[int, int] = field(default_factory=dict)   # PSS/E num → matrix row
    name_to_psse: dict[str, int] = field(default_factory=dict)  # 'BUS2' → psse_num
    comp_to_psse: dict[int, int] = field(default_factory=dict)  # competition bus 2 → psse_num

    Ybus: np.ndarray = field(default=None)
    Zbus: np.ndarray = field(default=None)
    pmu_bus_indices: list[int] = field(default_factory=list)    # row indices in Ybus
    gen_indices: list[int] = field(default_factory=list)
    branch_list: list[tuple[int, int]] = field(default_factory=list)  # (comp_from, comp_to)
    ext_bus_order: list[int] = field(default_factory=list)      # competition bus number per row


def _parse_raw(raw_path: Path) -> tuple[list[BusRecord], list[BranchRecord], list[int]]:
    """Parse PSS/E v30 RAW file — bus, branch, and source (gen) sections."""
    with open(raw_path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    buses: list[BusRecord] = []
    branches: list[BranchRecord] = []
    gen_psse: list[int] = []

    # Skip first 3 header lines
    i = 3
    section = "bus"

    while i < len(lines):
        line = lines[i].strip()
        i += 1

        if not line or line.startswith("@"):
            continue

        # Section transitions
        if "end bus section" in line.lower():
            section = "load"
            continue
        if "end load section" in line.lower():
            section = "source"
            continue
        if "end source section" in line.lower():
            section = "branch"
            continue

        # Strip inline comments
        if "/" in line:
            line = line[: line.index("/")].strip()

        # End-of-section marker
        parts = [p.strip() for p in line.split(",")]
        if parts[0] == "0":
            if section == "branch":
                break
            section = {"bus": "load", "load": "source", "source": "branch"}.get(section, section)
            continue

        if section == "bus":
            try:
                psse_num = int(parts[0])
                name = parts[1].strip().strip("'\"")
                base_kv = float(parts[2])
                bus_type = int(parts[3])
                vm_pu = float(parts[8])
                va_deg = float(parts[9])
                buses.append(BusRecord(psse_num, name, base_kv, bus_type, vm_pu, va_deg))
            except (IndexError, ValueError):
                pass

        elif section == "source":
            try:
                psse_num = int(parts[0])
                gen_psse.append(psse_num)
            except (IndexError, ValueError):
                pass

        elif section == "branch":
            try:
                fb = int(parts[0])
                tb = int(parts[1])
                r = float(parts[3])
                x = float(parts[4])
                b = float(parts[5])
                tap = float(parts[9]) if len(parts) > 9 else 0.0
                phase_deg = float(parts[10]) if len(parts) > 10 else 0.0
                branches.append(BranchRecord(fb, tb, r, x, b, tap, phase_deg))
            except (IndexError, ValueError):
                pass

    log.info("Parsed: %d buses, %d branches, %d generators", len(buses), len(branches), len(gen_psse))
    return buses, branches, list(set(gen_psse))


def _build_ybus(
    buses: list[BusRecord],
    branches: list[BranchRecord],
    psse_to_idx: dict[int, int],
) -> np.ndarray:
    """Build N×N complex admittance matrix from branch data (p.u. on 100 MVA base)."""
    N = len(buses)
    Y = np.zeros((N, N), dtype=complex)

    for br in branches:
        fi = psse_to_idx[br.from_bus]
        ti = psse_to_idx[br.to_bus]

        if abs(br.x) < 1e-12:
            continue  # skip zero-impedance branches (would cause singularity)

        z_series = complex(br.r, br.x)
        y_series = 1.0 / z_series
        y_shunt = complex(0, br.b / 2.0)

        tap = br.tap if br.tap != 0 else 1.0
        phase = np.deg2rad(br.phase_deg)
        t = tap * np.exp(1j * phase)  # complex tap ratio

        # π model with off-nominal tap:
        # Y[f,f] += y_series / |t|^2 + y_shunt / |t|^2
        # Y[t,t] += y_series + y_shunt
        # Y[f,t] -= y_series / conj(t)
        # Y[t,f] -= y_series / t
        if abs(tap - 1.0) < 1e-6 and abs(br.phase_deg) < 1e-6:
            # Regular line
            Y[fi, fi] += y_series + y_shunt
            Y[ti, ti] += y_series + y_shunt
            Y[fi, ti] -= y_series
            Y[ti, fi] -= y_series
        else:
            # Transformer with off-nominal tap or phase shift
            Y[fi, fi] += (y_series + y_shunt) / (abs(t) ** 2)
            Y[ti, ti] += y_series + y_shunt
            Y[fi, ti] -= y_series / np.conj(t)
            Y[ti, fi] -= y_series / t

    # Add bus shunts if any (not in this case — no shunt elements in .raw)
    return Y


def load_case(raw_path: Path | str) -> GridCase:
    """Parse .raw, build Ybus/Zbus, map competition bus names."""
    raw_path = Path(raw_path)
    buses, branches, gen_psse = _parse_raw(raw_path)

    # Build index maps
    psse_to_idx = {b.psse_num: i for i, b in enumerate(buses)}
    name_to_psse = {b.name: b.psse_num for b in buses}

    # Map competition bus number k (from CSV filename "Busk_...") to PSS/E bus number
    # by looking for name "BUSk" in the parsed bus records.
    comp_to_psse: dict[int, int] = {}
    for comp_k in PMU_BUS_NAMES:
        target_name = f"BUS{comp_k}"
        if target_name in name_to_psse:
            comp_to_psse[comp_k] = name_to_psse[target_name]
        else:
            log.warning("Competition bus BUS%d not found in .raw names", comp_k)

    # Build Ybus
    Ybus = _build_ybus(buses, branches, psse_to_idx)

    # Build Zbus (full matrix inversion)
    try:
        Zbus = np.linalg.inv(Ybus)
    except np.linalg.LinAlgError:
        log.warning("Ybus singular — using pseudo-inverse for Zbus")
        Zbus = np.linalg.pinv(Ybus)

    # PMU bus indices in Ybus (rows)
    pmu_bus_indices: list[int] = []
    for comp_k in PMU_BUS_NAMES:
        psse_num = comp_to_psse.get(comp_k)
        if psse_num is not None and psse_num in psse_to_idx:
            pmu_bus_indices.append(psse_to_idx[psse_num])

    # Generator indices
    gen_indices = sorted({psse_to_idx[g] for g in gen_psse if g in psse_to_idx})

    # Branch list in competition bus numbering (best effort)
    psse_to_comp = {v: k for k, v in comp_to_psse.items()}
    branch_list: list[tuple[int, int]] = []
    for br in branches:
        fc = psse_to_comp.get(psse_to_idx.get(br.from_bus, -1), br.from_bus)
        tc = psse_to_comp.get(psse_to_idx.get(br.to_bus, -1), br.to_bus)
        branch_list.append((fc, tc))

    # ext_bus_order: competition bus number for each row in Ybus (None for non-competition buses)
    ext_bus_order: list[int] = []
    psse_to_comp_num = {v: k for k, v in comp_to_psse.items()}
    for b in buses:
        ext_bus_order.append(psse_to_comp_num.get(b.psse_num, b.psse_num))

    grid = GridCase(
        buses=buses,
        branches=branches,
        gen_psse_nums=gen_psse,
        psse_to_idx=psse_to_idx,
        name_to_psse=name_to_psse,
        comp_to_psse=comp_to_psse,
        Ybus=Ybus,
        Zbus=Zbus,
        pmu_bus_indices=pmu_bus_indices,
        gen_indices=gen_indices,
        branch_list=branch_list,
        ext_bus_order=ext_bus_order,
    )

    _check_voltages(grid)
    return grid


def _check_voltages(grid: GridCase) -> None:
    """Log base-case voltages at PMU buses and warn if >1% off spec."""
    for comp_k, (ref_vm, ref_va) in SPEC_TABLE1.items():
        psse_num = grid.comp_to_psse.get(comp_k)
        if psse_num is None:
            continue
        idx = grid.psse_to_idx.get(psse_num)
        if idx is None:
            continue
        bus = grid.buses[idx]
        err_v = abs(bus.vm_pu - ref_vm)
        err_a = abs(bus.va_deg - ref_va)
        status = "OK" if err_v < 0.005 and err_a < 0.5 else "WARN"
        log.info(
            "[%s] BUS%-2d (PSS/E %d): |V|=%.4f (spec %.4f, Δ%.4f), θ=%.2f° (spec %.2f°, Δ%.2f°)",
            status, comp_k, psse_num, bus.vm_pu, ref_vm, err_v, bus.va_deg, ref_va, err_a,
        )
