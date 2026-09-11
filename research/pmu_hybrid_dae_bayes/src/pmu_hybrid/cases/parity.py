"""Static parity diagnostics that fail loudly instead of tuning around mismatch."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pmu_hybrid.cases.ieee39_andes import AndesStaticSolution, solve_static as solve_andes
from pmu_hybrid.cases.ieee39_pandapower import (
    PandapowerBranchFlow, PandapowerStaticSolution, solve_canonical_from_andes,
    solve_static as solve_pandapower,
)
from pmu_hybrid.physics.network import branch_terminal_powers, degrees, gauge_aligned_angles
from pmu_hybrid.utils.manifests import base_manifest, write_json


@dataclass(frozen=True)
class ParityTolerances:
    voltage_magnitude_pu: float = 1e-5
    voltage_angle_deg: float = 1e-3
    injection_mva: float = 1e-2
    branch_flow_mva: float = 1e-2


def _edge_key(from_bus: int, to_bus: int) -> tuple[int, int]:
    return tuple(sorted((int(from_bus), int(to_bus))))


def _match_branch_flows(andes: AndesStaticSolution, pandapower: PandapowerStaticSolution) -> tuple[pd.DataFrame, list[str]]:
    """Compare both terminal flows after a transparent topology-only match."""
    pp_by_edge: dict[tuple[int, int], list[PandapowerBranchFlow]] = defaultdict(list)
    for flow in pandapower.branch_flows:
        pp_by_edge[_edge_key(flow.from_bus, flow.to_bus)].append(flow)
    position = {bus: index for index, bus in enumerate(andes.bus_ids)}
    voltages = andes.voltage_pu * np.exp(1j * andes.angle_rad)
    rows: list[dict[str, Any]] = []
    unmatched: list[str] = []
    used: set[str] = set()
    for branch in andes.branches:
        key = _edge_key(branch.from_bus, branch.to_bus)
        candidates = [item for item in pp_by_edge[key] if item.identifier not in used]
        if not candidates:
            unmatched.append(branch.identifier)
            continue
        # Parallel elements have the same endpoints. Pair deterministically by
        # the lexicographic imported identifier, then preserve both terminals.
        candidate = sorted(candidates, key=lambda item: item.identifier)[0]
        used.add(candidate.identifier)
        power_from, power_to = branch_terminal_powers(
            branch, voltages[position[branch.from_bus]], voltages[position[branch.to_bus]], andes.base_mva,
        )
        if (candidate.from_bus, candidate.to_bus) == (branch.from_bus, branch.to_bus):
            pp_from, pp_to = candidate.from_power_mva, candidate.to_power_mva
        else:
            pp_from, pp_to = candidate.to_power_mva, candidate.from_power_mva
        rows.append({
            "andes_branch": branch.identifier,
            "pandapower_branch": candidate.identifier,
            "from_bus": branch.from_bus,
            "to_bus": branch.to_bus,
            "andes_p_from_mw": power_from.real,
            "andes_q_from_mvar": power_from.imag,
            "pandapower_p_from_mw": pp_from.real,
            "pandapower_q_from_mvar": pp_from.imag,
            "andes_p_to_mw": power_to.real,
            "andes_q_to_mvar": power_to.imag,
            "pandapower_p_to_mw": pp_to.real,
            "pandapower_q_to_mvar": pp_to.imag,
            "max_terminal_component_error_mva": max(
                abs(power_from.real - pp_from.real), abs(power_from.imag - pp_from.imag),
                abs(power_to.real - pp_to.real), abs(power_to.imag - pp_to.imag),
            ),
        })
    unmatched.extend(item.identifier for item in pandapower.branch_flows if item.identifier not in used)
    return pd.DataFrame(rows), unmatched


def compare(andes: AndesStaticSolution, pandapower: PandapowerStaticSolution, tolerances: ParityTolerances = ParityTolerances()) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Produce full state/injection/branch diagnostics and one strict gate."""
    if set(andes.bus_ids) != set(pandapower.bus_ids):
        raise ValueError("ANDES and pandapower do not expose the same IEEE-39 bus IDs")
    ordered = tuple(sorted(andes.bus_ids))
    andes_position = {bus: index for index, bus in enumerate(andes.bus_ids)}
    pp_position = {bus: index for index, bus in enumerate(pandapower.bus_ids)}
    reference = 39
    andes_angles = gauge_aligned_angles(andes.angle_rad, andes_position[reference])
    pp_angles = gauge_aligned_angles(pandapower.angle_rad, pp_position[reference])
    state = pd.DataFrame({
        "bus": ordered,
        "andes_vm_pu": [andes.voltage_pu[andes_position[bus]] for bus in ordered],
        "pandapower_vm_pu": [pandapower.voltage_pu[pp_position[bus]] for bus in ordered],
        "andes_va_deg_gauge": [degrees(andes_angles)[andes_position[bus]] for bus in ordered],
        "pandapower_va_deg_gauge": [degrees(pp_angles)[pp_position[bus]] for bus in ordered],
    })
    state["vm_abs_error_pu"] = (state.andes_vm_pu - state.pandapower_vm_pu).abs()
    state["va_abs_error_deg"] = (state.andes_va_deg_gauge - state.pandapower_va_deg_gauge).abs()
    injection = pd.DataFrame({
        "bus": ordered,
        "andes_p_mw": [andes.injections_mva[bus].real for bus in ordered],
        "andes_q_mvar": [andes.injections_mva[bus].imag for bus in ordered],
        "pandapower_p_mw": [pandapower.injections_mva[bus].real for bus in ordered],
        "pandapower_q_mvar": [pandapower.injections_mva[bus].imag for bus in ordered],
    })
    injection["p_abs_error_mw"] = (injection.andes_p_mw - injection.pandapower_p_mw).abs()
    injection["q_abs_error_mvar"] = (injection.andes_q_mvar - injection.pandapower_q_mvar).abs()
    branch, unmatched = _match_branch_flows(andes, pandapower)
    summary = {
        "reference_bus": reference,
        "bus_count": len(ordered),
        "andes_branch_count": len(andes.branches),
        "pandapower_branch_count": len(pandapower.branch_flows),
        "matched_branch_count": int(len(branch)),
        "unmatched_branches": unmatched,
        "max_vm_abs_error_pu": float(state.vm_abs_error_pu.max()),
        "max_va_abs_error_deg": float(state.va_abs_error_deg.max()),
        "max_p_injection_abs_error_mw": float(injection.p_abs_error_mw.max()),
        "max_q_injection_abs_error_mvar": float(injection.q_abs_error_mvar.max()),
        "max_branch_terminal_component_error_mva": float(branch.max_terminal_component_error_mva.max()) if len(branch) else float("inf"),
        "tolerances": asdict(tolerances),
    }
    summary["gate_pass"] = bool(
        summary["matched_branch_count"] == len(andes.branches) == len(pandapower.branch_flows)
        and not unmatched
        and summary["max_vm_abs_error_pu"] <= tolerances.voltage_magnitude_pu
        and summary["max_va_abs_error_deg"] <= tolerances.voltage_angle_deg
        and summary["max_p_injection_abs_error_mw"] <= tolerances.injection_mva
        and summary["max_q_injection_abs_error_mvar"] <= tolerances.injection_mva
        and summary["max_branch_terminal_component_error_mva"] <= tolerances.branch_flow_mva
    )
    summary["status"] = "PASS" if summary["gate_pass"] else "FAIL"
    summary["interpretation"] = (
        "Numerically consistent shared IEEE-39 static data."
        if summary["gate_pass"] else
        "Static sources differ or the translation/matching convention is unresolved; estimator phases remain blocked."
    )
    return {"bus_state": state, "bus_injections": injection, "branch_flows": branch}, summary


def run(root: Path) -> dict[str, Any]:
    """Execute static parity and save only auditable machine-readable outputs."""
    root = root.resolve()
    repository_root = root.parents[2]
    results = root / "output" / "results" / "e01_static_parity"
    manifests = root / "output" / "manifests"
    reports = root / "output" / "reports"
    results.mkdir(parents=True, exist_ok=True)
    andes = solve_andes()
    source_pandapower = solve_pandapower()
    source_tables, source_summary = compare(andes, source_pandapower)
    canonical_pandapower, conversion = solve_canonical_from_andes()
    tables, summary = compare(andes, canonical_pandapower)
    for prefix, table_group in (("source", source_tables), ("canonical", tables)):
        for name, table in table_group.items():
            table.to_csv(results / f"{prefix}_{name}.csv", index=False)
            try:
                table.to_parquet(results / f"{prefix}_{name}.parquet", index=False)
            except (ImportError, ValueError):
                pass
    for name, table in tables.items():
        # Stable unprefixed names always refer to the shared canonical gate.
        table.to_csv(results / f"{name}.csv", index=False)
        try:
            table.to_parquet(results / f"{name}.parquet", index=False)
        except (ImportError, ValueError):
            pass
    manifest = {
        **base_manifest(repository_root, seed=20260911),
        "phase": "E01",
        "status": summary["status"],
        "andes_models": andes.model_counts,
        "direct_shipped_case_parity": source_summary,
        "conversion_manifest": conversion,
        "canonical_parity": summary,
    }
    write_json(manifests / "static_parity.json", manifest)
    (reports / "static_parity.md").write_text(
        "# ANDES - pandapower static parity\n\n"
        f"Canonical status: **{summary['status']}**\n\n"
        "## Direct shipped-case comparison\n\n"
        f"{json.dumps(source_summary, indent=2, sort_keys=True)}\n\n"
        "## ANDES-to-pandapower conversion manifest\n\n"
        f"{json.dumps(conversion, indent=2, sort_keys=True)}\n\n"
        "## Canonical shared-case gate\n\n"
        f"{json.dumps(summary, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    return manifest
