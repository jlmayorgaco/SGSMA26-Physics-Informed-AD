"""Independent pandapower case39 static adapter."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PandapowerBranchFlow:
    identifier: str
    from_bus: int
    to_bus: int
    from_power_mva: complex
    to_power_mva: complex


@dataclass(frozen=True)
class PandapowerStaticSolution:
    bus_ids: tuple[int, ...]
    voltage_pu: np.ndarray
    angle_rad: np.ndarray
    injections_mva: dict[int, complex]
    branch_flows: tuple[PandapowerBranchFlow, ...]
    base_mva: float


def _bus_identifier(name: object, index: int) -> int:
    """Recover IEEE bus numbering from pandapower's name or 0-based index."""
    match = re.search(r"(?:BUS)?\s*([0-9]+)", str(name).upper())
    if match:
        candidate = int(match.group(1))
        if 1 <= candidate <= 39:
            return candidate
    return int(index) + 1


def solve_static() -> PandapowerStaticSolution:
    """Solve pandapower.networks.case39 without importing any campaign truth."""
    import pandapower as pp
    import pandapower.networks as networks

    net = networks.case39()
    pp.runpp(net, calculate_voltage_angles=True)
    if not bool(net.converged):
        raise RuntimeError("pandapower case39 did not converge")
    index_to_bus = {
        int(index): _bus_identifier(row.get("name", ""), int(index))
        for index, row in net.bus.iterrows()
    }
    ordered = sorted(index_to_bus.items(), key=lambda item: item[1])
    bus_ids = tuple(bus for _, bus in ordered)
    network_indices = [index for index, _ in ordered]
    result = net.res_bus.loc[network_indices]
    voltage_pu = result.vm_pu.to_numpy(dtype=float)
    angle_rad = np.deg2rad(result.va_degree.to_numpy(dtype=float))
    # pandapower's res_bus sign is consumption-positive; this campaign uses
    # electrical injection-positive, matching I = YV.
    injections = {
        bus: complex(-float(row.p_mw), -float(row.q_mvar))
        for bus, (_, row) in zip(bus_ids, result.iterrows())
    }
    flows: list[PandapowerBranchFlow] = []
    for index, row in net.line.iterrows():
        outcome = net.res_line.loc[index]
        flows.append(PandapowerBranchFlow(
            identifier=f"line:{index}",
            from_bus=index_to_bus[int(row.from_bus)],
            to_bus=index_to_bus[int(row.to_bus)],
            from_power_mva=complex(float(outcome.p_from_mw), float(outcome.q_from_mvar)),
            to_power_mva=complex(float(outcome.p_to_mw), float(outcome.q_to_mvar)),
        ))
    for index, row in net.trafo.iterrows():
        outcome = net.res_trafo.loc[index]
        flows.append(PandapowerBranchFlow(
            identifier=f"trafo:{index}",
            from_bus=index_to_bus[int(row.hv_bus)],
            to_bus=index_to_bus[int(row.lv_bus)],
            from_power_mva=complex(float(outcome.p_hv_mw), float(outcome.q_hv_mvar)),
            to_power_mva=complex(float(outcome.p_lv_mw), float(outcome.q_lv_mvar)),
        ))
    return PandapowerStaticSolution(
        bus_ids=bus_ids,
        voltage_pu=voltage_pu,
        angle_rad=angle_rad,
        injections_mva=injections,
        branch_flows=tuple(flows),
        base_mva=float(net.sn_mva),
    )


def _andes_array(model: Any, name: str, default: np.ndarray) -> np.ndarray:
    """Read an ANDES parameter without depending on private object layout."""
    candidate = getattr(model, name, None)
    return np.asarray(candidate.v if candidate is not None and hasattr(candidate, "v") else default)


def _solution_from_network(net: Any, branch_flows: list[PandapowerBranchFlow]) -> PandapowerStaticSolution:
    """Convert a solved canonical pandapower net to the campaign convention."""
    if not bool(net.converged):
        raise RuntimeError("Translated canonical pandapower case did not converge")
    ordered = sorted((int(index), int(index)) for index in net.bus.index)
    bus_ids = tuple(bus for _, bus in ordered)
    indices = [index for index, _ in ordered]
    result = net.res_bus.loc[indices]
    return PandapowerStaticSolution(
        bus_ids=bus_ids,
        voltage_pu=result.vm_pu.to_numpy(dtype=float),
        angle_rad=np.deg2rad(result.va_degree.to_numpy(dtype=float)),
        injections_mva={
            bus: complex(-float(row.p_mw), -float(row.q_mvar))
            for bus, (_, row) in zip(bus_ids, result.iterrows())
        },
        branch_flows=tuple(branch_flows),
        base_mva=float(net.sn_mva),
    )


def solve_canonical_from_andes() -> tuple[PandapowerStaticSolution, dict[str, Any]]:
    """Translate ANDES' static inputs into pandapower and independently solve them.

    The bundled cases share IEEE-39 topology but not the same operating point.
    This function is the explicit, recorded conversion rather than an
    undocumented adjustment of output vectors.
    """
    from pmu_hybrid.cases.ieee39_andes import load_native_system
    import pandapower as pp

    system = load_native_system()
    base_mva = float(getattr(system.config, "mva", 100.0))
    net = pp.create_empty_network(sn_mva=base_mva)
    buses = tuple(int(value) for value in _andes_array(system.Bus, "idx", np.array([])))
    nominal_kv = _andes_array(system.Bus, "Vn", np.ones(len(buses))).astype(float)
    for bus, voltage_kv in zip(buses, nominal_kv):
        pp.create_bus(net, vn_kv=float(voltage_kv), name=f"BUS{bus}", index=bus)

    load_count = 0
    for enabled, bus, p0, q0 in zip(
        _andes_array(system.PQ, "u", np.ones(system.PQ.n)).astype(bool),
        _andes_array(system.PQ, "bus", np.array([])).astype(int),
        _andes_array(system.PQ, "p0", np.array([])).astype(float),
        _andes_array(system.PQ, "q0", np.array([])).astype(float),
    ):
        if enabled:
            pp.create_load(net, bus=int(bus), p_mw=float(p0 * base_mva), q_mvar=float(q0 * base_mva), name=f"ANDES_PQ_{bus}")
            load_count += 1
    generator_count = 0
    for enabled, bus, p0, v0, pmax, pmin, qmax, qmin in zip(
        _andes_array(system.PV, "u", np.ones(system.PV.n)).astype(bool),
        _andes_array(system.PV, "bus", np.array([])).astype(int),
        _andes_array(system.PV, "p0", np.array([])).astype(float),
        _andes_array(system.PV, "v0", np.ones(system.PV.n)).astype(float),
        _andes_array(system.PV, "pmax", np.full(system.PV.n, np.nan)).astype(float),
        _andes_array(system.PV, "pmin", np.full(system.PV.n, np.nan)).astype(float),
        _andes_array(system.PV, "qmax", np.full(system.PV.n, np.nan)).astype(float),
        _andes_array(system.PV, "qmin", np.full(system.PV.n, np.nan)).astype(float),
    ):
        if enabled:
            pp.create_gen(
                net, bus=int(bus), p_mw=float(p0 * base_mva), vm_pu=float(v0),
                max_p_mw=float(pmax * base_mva), min_p_mw=float(pmin * base_mva),
                max_q_mvar=float(qmax * base_mva), min_q_mvar=float(qmin * base_mva),
                name=f"ANDES_PV_{bus}",
            )
            generator_count += 1
    for enabled, bus, v0, a0 in zip(
        _andes_array(system.Slack, "u", np.ones(system.Slack.n)).astype(bool),
        _andes_array(system.Slack, "bus", np.array([])).astype(int),
        _andes_array(system.Slack, "v0", np.ones(system.Slack.n)).astype(float),
        _andes_array(system.Slack, "a0", np.zeros(system.Slack.n)).astype(float),
    ):
        if enabled:
            pp.create_ext_grid(net, bus=int(bus), vm_pu=float(v0), va_degree=float(np.rad2deg(a0)), name=f"ANDES_SLACK_{bus}")
            generator_count += 1
    shunt_count = 0
    if hasattr(system, "Shunt"):
        for enabled, bus, conductance, susceptance, nominal in zip(
            _andes_array(system.Shunt, "u", np.ones(system.Shunt.n)).astype(bool),
            _andes_array(system.Shunt, "bus", np.array([])).astype(int),
            _andes_array(system.Shunt, "g", np.zeros(system.Shunt.n)).astype(float),
            _andes_array(system.Shunt, "b", np.zeros(system.Shunt.n)).astype(float),
            _andes_array(system.Shunt, "Sn", np.full(system.Shunt.n, base_mva)).astype(float),
        ):
            if enabled:
                # pandapower uses load-positive signs. Ysh = g + jb consumes
                # P = g and Q = -b in the I = YV injection convention.
                pp.create_shunt(net, bus=int(bus), p_mw=float(conductance * nominal), q_mvar=float(-susceptance * nominal), name=f"ANDES_SHUNT_{bus}")
                shunt_count += 1

    line = system.Line
    identifiers = [str(item) for item in _andes_array(line, "idx", np.arange(line.n))]
    line_count = transformer_count = 0
    flow_metadata: list[tuple[str, str, int]] = []
    for identifier, enabled, from_bus, to_bus, resistance, reactance, charging, tap, shift, vn_from, vn_to in zip(
        identifiers,
        _andes_array(line, "u", np.ones(line.n)).astype(bool),
        _andes_array(line, "bus1", np.array([])).astype(int),
        _andes_array(line, "bus2", np.array([])).astype(int),
        _andes_array(line, "r", np.array([])).astype(float),
        _andes_array(line, "x", np.array([])).astype(float),
        _andes_array(line, "b", np.zeros(line.n)).astype(float),
        _andes_array(line, "tap", np.ones(line.n)).astype(float),
        _andes_array(line, "phi", np.zeros(line.n)).astype(float),
        _andes_array(line, "Vn1", np.ones(line.n)).astype(float),
        _andes_array(line, "Vn2", np.ones(line.n)).astype(float),
    ):
        is_transformer = not np.isclose(vn_from, vn_to)
        if not is_transformer:
            index = pp.create_impedance(
                net, from_bus=int(from_bus), to_bus=int(to_bus), rft_pu=float(resistance), xft_pu=float(reactance),
                sn_mva=base_mva, bf_pu=float(charging / 2.0), bt_pu=float(charging / 2.0),
                name=identifier, in_service=bool(enabled),
            )
            flow_metadata.append((identifier, "impedance", int(index)))
            line_count += 1
            continue
        if vn_from >= vn_to:
            hv_bus, lv_bus, tap_side = int(from_bus), int(to_bus), "hv"
        else:
            hv_bus, lv_bus, tap_side = int(to_bus), int(from_bus), "lv"
        index = pp.create_transformer_from_parameters(
            net, hv_bus=hv_bus, lv_bus=lv_bus, sn_mva=base_mva,
            vn_hv_kv=float(max(vn_from, vn_to)), vn_lv_kv=float(min(vn_from, vn_to)),
            vkr_percent=float(resistance * 100.0), vk_percent=float(np.hypot(resistance, reactance) * 100.0),
            pfe_kw=0.0, i0_percent=0.0, shift_degree=float(np.rad2deg(shift)),
            tap_side=tap_side, tap_neutral=0, tap_pos=1,
            tap_step_percent=float((tap - 1.0) * 100.0), name=identifier, in_service=bool(enabled),
        )
        flow_metadata.append((identifier, "trafo", int(index)))
        transformer_count += 1
    pp.runpp(net, calculate_voltage_angles=True)
    flows: list[PandapowerBranchFlow] = []
    for identifier, kind, index in flow_metadata:
        if kind == "impedance":
            row = net.impedance.loc[index]
            result = net.res_impedance.loc[index]
            flows.append(PandapowerBranchFlow(
                identifier=identifier, from_bus=int(row.from_bus), to_bus=int(row.to_bus),
                from_power_mva=complex(float(result.p_from_mw), float(result.q_from_mvar)),
                to_power_mva=complex(float(result.p_to_mw), float(result.q_to_mvar)),
            ))
        else:
            row = net.trafo.loc[index]
            result = net.res_trafo.loc[index]
            flows.append(PandapowerBranchFlow(
                identifier=identifier, from_bus=int(row.hv_bus), to_bus=int(row.lv_bus),
                from_power_mva=complex(float(result.p_hv_mw), float(result.q_hv_mvar)),
                to_power_mva=complex(float(result.p_lv_mw), float(result.q_lv_mvar)),
            ))
    return _solution_from_network(net, flows), {
        "authority": "ANDES bundled ieee39_full.xlsx static inputs",
        "target": "pandapower independently solved translation",
        "bus_count": len(buses),
        "pq_load_count": load_count,
        "generator_count": generator_count,
        "shunt_count": shunt_count,
        "impedance_branch_count": line_count,
        "transformer_branch_count": transformer_count,
        "translated_branch_count": len(flow_metadata),
    }
