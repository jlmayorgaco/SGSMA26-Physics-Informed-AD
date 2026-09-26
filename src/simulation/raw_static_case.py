"""Import the competition PSS/E RAW operating point into the ANDES IEEE-39 DAE.

The competition RAW file uses the bus *names* (``BUS1`` ... ``BUS39``) as
the physical IEEE-39 identifiers while its numeric bus indices are permuted.
This module normalizes that convention, transplants the complete static case
into the dynamic ANDES workbook, and identifies the branch-current channel
seen by each PMU from voltage/current phasor consistency.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

# Avoid the incompatible gmpy2 binary shipped with this Python 3.13 Windows
# environment when callers import ANDES after this module.
os.environ.setdefault("SYMPY_GROUND_TYPES", "python")


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_CASE = WORKSPACE_ROOT / "data" / "metadata" / "IEEE_39_Bus_Power_System.raw"


def _record(line: str) -> list[str]:
    body = line.split("/", 1)[0].strip()
    return [part.strip().strip("'\"") for part in body.split(",")]


def _section(lines: list[str], start: str, end: str) -> list[list[str]]:
    active = False
    records: list[list[str]] = []
    for line in lines:
        lowered = line.lower()
        if start in lowered:
            active = True
            continue
        if active and end in lowered:
            break
        if active and line.strip() and not line.lstrip().startswith("0"):
            records.append(_record(line))
    return records


@dataclass(frozen=True)
class RawBus:
    bus: int
    internal_bus: int
    name: str
    voltage_kv: float
    bus_type: int
    voltage_pu: float
    angle_deg: float


@dataclass(frozen=True)
class RawGenerator:
    bus: int
    p_mw: float
    q_mvar: float
    voltage_setpoint_pu: float


@dataclass(frozen=True)
class RawBranch:
    row: int
    from_bus: int
    to_bus: int
    resistance_pu: float
    reactance_pu: float
    charging_pu: float
    tap: float
    shift_deg: float


@dataclass(frozen=True)
class RawStaticCase:
    path: str
    system_base_mva: float
    buses: dict[int, RawBus]
    loads_mva: dict[int, complex]
    generators: dict[int, RawGenerator]
    branches: tuple[RawBranch, ...]


@dataclass(frozen=True)
class PMUBranchMapping:
    bus: int
    other_bus: int
    branch_row: int
    polarity: int
    score: float
    raw_current_a: float
    case_current_a: float
    raw_relative_angle_deg: float
    case_relative_angle_deg: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_raw_static_case(path: str | Path = DEFAULT_RAW_CASE) -> RawStaticCase:
    """Parse the compact competition RAW dialect using BUS names as IDs."""

    source = Path(path)
    lines = source.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not lines:
        raise ValueError(f"Empty RAW case: {source}")
    header = _record(lines[0])
    system_base_mva = float(header[1]) if len(header) > 1 else 100.0

    buses: dict[int, RawBus] = {}
    internal_to_physical: dict[int, int] = {}
    for line in lines[3:]:
        if line.lstrip().startswith("0"):
            break
        values = _record(line)
        match = re.search(r"BUS\s*([0-9]+)", values[1].upper())
        if not match:
            continue
        physical = int(match.group(1))
        internal = int(values[0])
        internal_to_physical[internal] = physical
        buses[physical] = RawBus(
            bus=physical,
            internal_bus=internal,
            name=values[1],
            voltage_kv=float(values[2]),
            bus_type=int(values[3]),
            voltage_pu=float(values[8]),
            angle_deg=float(values[9]),
        )

    loads: dict[int, complex] = {}
    for values in _section(lines, "starting load section", "end load section"):
        if int(float(values[2])) != 1:
            continue
        bus = internal_to_physical[int(values[0])]
        # At V=1 pu, constant-P, constant-I and constant-Y terms sum to
        # the supplied RAW operating-point demand.
        p_mw = sum(float(values[idx]) for idx in (5, 7, 9))
        q_mvar = sum(float(values[idx]) for idx in (6, 8, 10))
        loads[bus] = loads.get(bus, 0j) + complex(p_mw, q_mvar)

    generators: dict[int, RawGenerator] = {}
    for values in _section(lines, "starting source section", "end source section"):
        status = int(float(values[14])) if len(values) > 14 else 1
        if status != 1:
            continue
        bus = internal_to_physical[int(values[0])]
        generators[bus] = RawGenerator(
            bus=bus,
            p_mw=float(values[2]),
            q_mvar=float(values[3]),
            voltage_setpoint_pu=float(values[6]),
        )

    branches: list[RawBranch] = []
    for values in _section(lines, "starting branch section", "end branch section"):
        try:
            internal_from = int(values[0])
            internal_to = int(values[1])
        except (ValueError, IndexError):
            continue
        if internal_from not in internal_to_physical or internal_to not in internal_to_physical:
            continue
        tap = float(values[9]) if len(values) > 9 and values[9] else 0.0
        branches.append(
            RawBranch(
                row=len(branches),
                from_bus=internal_to_physical[internal_from],
                to_bus=internal_to_physical[internal_to],
                resistance_pu=float(values[3]),
                reactance_pu=float(values[4]),
                charging_pu=float(values[5]),
                tap=tap if abs(tap) > 1e-12 else 1.0,
                shift_deg=float(values[10]) if len(values) > 10 and values[10] else 0.0,
            )
        )

    if len(buses) != 39 or len(branches) != 46:
        raise ValueError(
            f"Expected the 39-bus/46-branch competition case, got {len(buses)} buses and {len(branches)} branches"
        )
    return RawStaticCase(
        path=str(source.resolve()),
        system_base_mva=system_base_mva,
        buses=buses,
        loads_mva=loads,
        generators=generators,
        branches=tuple(branches),
    )


def apply_raw_static_case(system, raw_case: RawStaticCase) -> dict[str, Any]:
    """Transplant RAW topology, dispatch and operating point before setup.

    The bundled ANDES IEEE-39 workbook contains two fixed shunts at BUS4 and
    BUS5 which are absent from the competition RAW. Leaving them enabled is
    enough to create multi-percent voltage errors, so they are explicitly
    disabled here.
    """

    system_buses = {int(bus) for bus in system.Bus.idx.v}
    if system_buses != set(raw_case.buses):
        raise ValueError("ANDES and RAW cases do not contain the same physical bus IDs")
    if system.Line.n != len(raw_case.branches):
        raise ValueError("ANDES and RAW cases do not contain the same number of branches")

    if hasattr(system, "Shunt") and system.Shunt.n:
        system.Shunt.u.v = np.zeros(system.Shunt.n, dtype=int)

    for idx, bus_value in enumerate(system.Bus.idx.v):
        bus = raw_case.buses[int(bus_value)]
        system.Bus.Vn.v[idx] = bus.voltage_kv
        system.Bus.v0.v[idx] = bus.voltage_pu
        system.Bus.a0.v[idx] = math.radians(bus.angle_deg)

    for idx, bus_value in enumerate(system.PQ.bus.v):
        bus = int(bus_value)
        demand = raw_case.loads_mva.get(bus, 0j)
        system.PQ.p0.v[idx] = demand.real / raw_case.system_base_mva
        system.PQ.q0.v[idx] = demand.imag / raw_case.system_base_mva
        system.PQ.Vn.v[idx] = raw_case.buses[bus].voltage_kv

    for model in (system.PV, system.Slack):
        for idx, bus_value in enumerate(model.bus.v):
            bus = int(bus_value)
            source = raw_case.generators[bus]
            model.p0.v[idx] = source.p_mw / raw_case.system_base_mva
            model.q0.v[idx] = source.q_mvar / raw_case.system_base_mva
            model.v0.v[idx] = source.voltage_setpoint_pu
            model.Vn.v[idx] = raw_case.buses[bus].voltage_kv
    slack_bus = int(system.Slack.bus.v[0])
    system.Slack.a0.v[0] = math.radians(raw_case.buses[slack_bus].angle_deg)

    if hasattr(system, "GENROU"):
        for idx, bus_value in enumerate(system.GENROU.bus.v):
            system.GENROU.Vn.v[idx] = raw_case.buses[int(bus_value)].voltage_kv

    # Reuse the existing Line device rows but replace their endpoints and all
    # electrical data. Vn1/Vn2 and Sn are essential: ANDES otherwise converts
    # several impedances using the ratings from the bundled, different case.
    for idx, branch in enumerate(raw_case.branches):
        values = {
            "bus1": branch.from_bus,
            "bus2": branch.to_bus,
            "Sn": raw_case.system_base_mva,
            "Vn1": raw_case.buses[branch.from_bus].voltage_kv,
            "Vn2": raw_case.buses[branch.to_bus].voltage_kv,
            "r": branch.resistance_pu,
            "x": branch.reactance_pu,
            "b": branch.charging_pu,
            "tap": branch.tap,
            "phi": math.radians(branch.shift_deg),
        }
        for name, value in values.items():
            getattr(system.Line, name).v[idx] = value

    return {
        "raw_static_case": raw_case.path,
        "raw_static_bus_count": len(raw_case.buses),
        "raw_static_branch_count": len(raw_case.branches),
        "disabled_builtin_shunts": int(getattr(system.Shunt, "n", 0)),
    }


def _branch_terminal_current(
    branch: RawBranch,
    voltages: dict[int, complex],
    terminal_bus: int,
) -> complex:
    y_series = 1.0 / complex(branch.resistance_pu, branch.reactance_pu)
    y_shunt = 0.5j * branch.charging_pu
    tap = branch.tap * np.exp(1j * math.radians(branch.shift_deg))
    v_from = voltages[branch.from_bus]
    v_to = voltages[branch.to_bus]
    # Full off-nominal transformer terminal equations.  The former
    # ``v_from / tap`` shortcut omitted the conjugate-tap factor at the from
    # terminal and therefore violated nodal current balance for transformers.
    i_from = (y_series + y_shunt) / (tap * np.conj(tap)) * v_from - y_series / np.conj(tap) * v_to
    i_to = -y_series / tap * v_from + (y_series + y_shunt) * v_to
    if terminal_bus == branch.from_bus:
        return complex(i_from)
    if terminal_bus == branch.to_bus:
        return complex(i_to)
    raise ValueError(f"BUS{terminal_bus} is not incident on branch row {branch.row}")


def identify_pmu_branch_mappings(
    raw_case: RawStaticCase,
    frames: dict[int, pd.DataFrame],
    baseline_end_s: float,
    baseline_duration_s: float = 1.3,
) -> dict[str, dict[str, Any]]:
    """Identify a branch terminal and CT polarity for each PMU current channel."""

    voltages = {
        bus: data.voltage_pu * np.exp(1j * math.radians(data.angle_deg))
        for bus, data in raw_case.buses.items()
    }
    mappings: dict[str, dict[str, Any]] = {}
    for bus, frame in frames.items():
        time_s = frame["TIMESTAMP"].to_numpy(float)
        select = (time_s >= baseline_end_s - baseline_duration_s) & (time_s <= baseline_end_s)
        if np.sum(select) < 3:
            continue
        raw_i = float(np.nanmedian(frame.loc[select, f"BUS{bus}_IA_MAG"]))
        voltage_angle = np.deg2rad(frame.loc[select, f"BUS{bus}_VA_ANG"].to_numpy(float))
        current_angle = np.deg2rad(frame.loc[select, f"BUS{bus}_IA_ANG"].to_numpy(float))
        raw_relative = math.degrees(np.angle(np.mean(np.exp(1j * (current_angle - voltage_angle)))))
        base_a = raw_case.system_base_mva * 1e6 / (
            math.sqrt(3.0) * raw_case.buses[bus].voltage_kv * 1e3
        )

        candidates: list[PMUBranchMapping] = []
        for branch in raw_case.branches:
            if bus not in (branch.from_bus, branch.to_bus):
                continue
            other = branch.to_bus if bus == branch.from_bus else branch.from_bus
            terminal_current = _branch_terminal_current(branch, voltages, bus)
            for polarity in (1, -1):
                current = polarity * terminal_current
                case_i = abs(current) * base_a
                case_relative = math.degrees(np.angle(current) - np.angle(voltages[bus]))
                case_relative = (case_relative + 180.0) % 360.0 - 180.0
                angle_error = abs((case_relative - raw_relative + 180.0) % 360.0 - 180.0)
                magnitude_error = abs(math.log(max(case_i, 1.0) / max(raw_i, 1.0)))
                score = angle_error + 10.0 * magnitude_error
                candidates.append(
                    PMUBranchMapping(
                        bus=bus,
                        other_bus=other,
                        branch_row=branch.row,
                        polarity=polarity,
                        score=score,
                        raw_current_a=raw_i,
                        case_current_a=case_i,
                        raw_relative_angle_deg=raw_relative,
                        case_relative_angle_deg=case_relative,
                    )
                )
        if candidates:
            best = min(candidates, key=lambda item: item.score)
            mappings[str(bus)] = best.to_dict()
    return mappings


def replace_pmu_currents_with_branch_channels(
    system,
    signals_by_bus: dict[str, dict[str, np.ndarray]],
    mappings: dict[str, dict[str, Any]],
) -> None:
    """Replace nodal-injection currents with the calibrated branch channels."""

    if not mappings:
        return
    from m3_andes_calibration_raw import get_timeseries

    _, voltage_frame, angle_frame = get_timeseries(system)
    voltage_frame.columns = [str(value) for value in voltage_frame.columns]
    angle_frame.columns = [str(value) for value in angle_frame.columns]
    bus_positions = {int(bus): idx for idx, bus in enumerate(system.Bus.idx.v)}

    for bus_key, mapping in mappings.items():
        bus = int(bus_key)
        if bus_key not in signals_by_bus:
            continue
        row = int(mapping["branch_row"])
        from_bus = int(system.Line.bus1.v[row])
        to_bus = int(system.Line.bus2.v[row])
        terminal_bus = int(mapping.get("terminal_bus", bus))
        if bus not in (from_bus, to_bus) and terminal_bus not in (from_bus, to_bus):
            raise ValueError(f"Stored PMU mapping BUS{bus} is inconsistent with Line row {row}")
        if terminal_bus not in (from_bus, to_bus):
            raise ValueError(
                f"Stored current terminal BUS{terminal_bus} is inconsistent with Line row {row}"
            )

        r = float(system.Line.r.v[row])
        x = float(system.Line.x.v[row])
        charging = float(system.Line.b.v[row])
        tap = float(system.Line.tap.v[row]) * np.exp(1j * float(system.Line.phi.v[row]))
        vf = voltage_frame[str(from_bus)].to_numpy(float) * np.exp(
            1j * angle_frame[str(from_bus)].to_numpy(float)
        )
        vt = voltage_frame[str(to_bus)].to_numpy(float) * np.exp(
            1j * angle_frame[str(to_bus)].to_numpy(float)
        )
        y_series = 1.0 / complex(r, x)
        y_shunt = 0.5j * charging
        vf_tapped = vf / tap
        i_from = (vf_tapped - vt) * y_series + y_shunt * vf_tapped
        i_to = (vt - vf_tapped) * y_series + y_shunt * vt
        # A PMU may report the opposite transformer/branch terminal while its
        # voltage channel remains referenced to the named PMU bus.  R3 stores
        # that semantics explicitly instead of hiding it in an arbitrary
        # current-angle bias.
        current_pu = i_from if terminal_bus == from_bus else i_to
        current_pu = int(mapping.get("polarity", 1)) * current_pu

        bus_position = bus_positions[terminal_bus]
        voltage_kv = float(system.Bus.Vn.v[bus_position])
        base_a = float(system.config.mva) * 1e6 / (math.sqrt(3.0) * voltage_kv * 1e3)
        magnitude_a = np.abs(current_pu) * base_a
        angle_deg = (np.rad2deg(np.angle(current_pu)) + 180.0) % 360.0 - 180.0
        signals = signals_by_bus[bus_key]
        for phase, shift in zip("ABC", (0.0, -120.0, 120.0)):
            signals[f"I{phase}_MAG"] = magnitude_a.copy()
            signals[f"I{phase}_ANG"] = (angle_deg + shift + 180.0) % 360.0 - 180.0
