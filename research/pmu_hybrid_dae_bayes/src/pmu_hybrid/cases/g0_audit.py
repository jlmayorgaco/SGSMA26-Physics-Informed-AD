"""G0-A/G0-B static source inventory and admittance audit.

The ANDES workbook is authoritative.  pandapower is used only as an
independent target representation; no dynamic or estimator code is touched.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pmu_hybrid.cases.ieee39_andes import load_native_system
from pmu_hybrid.cases.ieee39_pandapower import solve_canonical_from_andes
from pmu_hybrid.physics.network import Branch, ybus


def _values(model: Any, name: str, default: Any = None) -> np.ndarray:
    value = getattr(model, name, None)
    if value is not None and hasattr(value, "v"):
        return np.asarray(value.v)
    if default is None:
        return np.asarray([])
    return np.asarray(default)


def _table(model: Any, source_model: str) -> pd.DataFrame:
    frame = model.as_df().reset_index(drop=True).copy()
    frame.insert(0, "source_model", source_model)
    return frame


def canonical_inventory(system: Any) -> dict[str, pd.DataFrame]:
    """Extract raw static rows without solving the power flow."""
    bus = _table(system.Bus, "Bus")
    line = _table(system.Line, "Line")
    line["element_kind"] = np.where(line["trans"].astype(bool), "transformer", "line")
    line["endpoint_pair"] = line.apply(lambda r: f"{int(r.bus1)}-{int(r.bus2)}", axis=1)
    shunt = _table(system.Shunt, "Shunt") if hasattr(system, "Shunt") else pd.DataFrame()
    pq = _table(system.PQ, "PQ")
    pv = _table(system.PV, "PV")
    slack = _table(system.Slack, "Slack")
    generator = pd.concat([pv, slack], ignore_index=True, sort=False)
    generator["generator_kind"] = np.where(generator["source_model"].eq("Slack"), "slack", "pv")
    return {
        "bus": bus,
        "branch": line,
        "transformer": line[line["element_kind"].eq("transformer")].copy(),
        "shunt": shunt,
        "generator": generator,
        "load": pq,
    }


def _source_branches(table: pd.DataFrame) -> tuple[Branch, ...]:
    return tuple(
        Branch(
            identifier=str(row.idx), from_bus=int(row.bus1), to_bus=int(row.bus2),
            resistance_pu=float(row.r), reactance_pu=float(row.x), charging_pu=float(row.b),
            tap=float(row.tap), shift_rad=float(row.phi), in_service=bool(row.u),
        )
        for row in table.itertuples(index=False)
    )


def _component_matrices(bus_ids: tuple[int, ...], branches: tuple[Branch, ...], shunts: dict[int, complex]) -> dict[str, np.ndarray]:
    n = len(bus_ids)
    pos = {bus: i for i, bus in enumerate(bus_ids)}
    components = {name: np.zeros((n, n), dtype=complex) for name in ("series", "line_charging", "transformer", "bus_shunt")}
    for branch in branches:
        if not branch.in_service:
            continue
        i, j = pos[branch.from_bus], pos[branch.to_bus]
        z = complex(branch.resistance_pu, branch.reactance_pu)
        ys = 1.0 / z
        tap = branch.complex_tap
        primitive = np.array([
            [(ys + 0.5j * branch.charging_pu) / (tap * np.conj(tap)), -ys / np.conj(tap)],
            [-ys / tap, ys + 0.5j * branch.charging_pu],
        ])
        target = None
        if branch.identifier.startswith("Line_") and int(branch.identifier.split("_")[-1]) <= 34:
            target = components["series"]
            target[np.ix_([i, j], [i, j])] += np.array([
                [ys / (tap * np.conj(tap)), -ys / np.conj(tap)],
                [-ys / tap, ys],
            ])
            components["line_charging"][i, i] += 0.5j * branch.charging_pu / (tap * np.conj(tap))
            components["line_charging"][j, j] += 0.5j * branch.charging_pu
        else:
            components["transformer"][np.ix_([i, j], [i, j])] += primitive
    for bus, value in shunts.items():
        components["bus_shunt"][pos[bus], pos[bus]] += value
    return components


def _pp_ybus_before_pf(net: Any) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    from pandapower.pd2ppc import _pd2ppc
    from pandapower.pypower.makeYbus import makeYbus
    from pandapower.run import _init_runpp_options

    _init_runpp_options(net, "nr", True, "flat", "auto", 1e-8, "pi", "power", False, False, True, False)
    ppc, _ = _pd2ppc(net)
    matrix, _, _ = makeYbus(ppc["baseMVA"], ppc["bus"], ppc["branch"])
    return matrix.toarray(), ppc, net._pd2ppc_lookups


def _pp_components(net: Any, ppc: dict[str, Any], lookups: dict[str, Any]) -> dict[str, np.ndarray]:
    from pandapower.pypower.makeYbus import branch_vectors
    from pandapower.pypower.idx_brch import BR_B, BR_G, BR_R, BR_R_ASYM, BR_STATUS, BR_X, BR_X_ASYM, F_BUS, T_BUS
    from pandapower.pypower.idx_bus import BS, GS

    n = len(ppc["bus"])
    out = {name: np.zeros((n, n), dtype=complex) for name in ("series", "line_charging", "transformer", "bus_shunt")}
    branch = ppc["branch"]
    Ytt, Yff, Yft, Ytf = branch_vectors(branch, len(branch))
    t0, t1 = lookups["branch"]["trafo"]
    i0, i1 = lookups["branch"]["impedance"]
    for k in range(len(branch)):
        if not bool(branch[k, BR_STATUS]):
            continue
        f, t = int(branch[k, F_BUS]), int(branch[k, T_BUS])
        primitive = np.array([[Yff[k], Yft[k]], [Ytf[k], Ytt[k]]])
        target = out["transformer"] if t0 <= k < t1 else out["series"]
        target[np.ix_([f, t], [f, t])] += primitive
        if i0 <= k < i1:
            # For pandapower impedance elements the branch B is the total
            # charging; makeYbus splits it equally at both terminals.
            out["series"][f, f] -= 0.5j * branch[k, BR_B]
            out["series"][t, t] -= 0.5j * branch[k, BR_B]
            out["line_charging"][f, f] += 0.5j * branch[k, BR_B]
            out["line_charging"][t, t] += 0.5j * branch[k, BR_B]
    out["bus_shunt"][np.arange(n), np.arange(n)] = (ppc["bus"][:, GS] + 1j * ppc["bus"][:, BS]) / ppc["baseMVA"]
    return out


def _primitive_audit(inventory: dict[str, pd.DataFrame], ppc: dict[str, Any], lookups: dict[str, Any]) -> pd.DataFrame:
    from pandapower.pypower.makeYbus import branch_vectors
    from pandapower.pypower.idx_brch import BR_B, BR_R, BR_STATUS, BR_X, F_BUS, SHIFT, TAP, T_BUS

    rows: list[dict[str, Any]] = []
    branches = inventory["branch"]
    Ytt, Yff, Yft, Ytf = branch_vectors(ppc["branch"], len(ppc["branch"]))
    t0, t1 = lookups["branch"]["trafo"]
    i0, i1 = lookups["branch"]["impedance"]
    for k, row in enumerate(branches.itertuples(index=False)):
        source = Branch(str(row.idx), int(row.bus1), int(row.bus2), float(row.r), float(row.x), float(row.b), float(row.tap), float(row.phi), bool(row.u))
        z = complex(source.resistance_pu, source.reactance_pu)
        ys = 1 / z
        tap = source.complex_tap
        expected = np.array([[ (ys + .5j*source.charging_pu)/(tap*np.conj(tap)), -ys/np.conj(tap)], [-ys/tap, ys + .5j*source.charging_pu]])
        pp_k = t0 + (k - 34) if bool(row.trans) else i0 + k
        actual = np.array([[Yff[pp_k], Yft[pp_k]], [Ytf[pp_k], Ytt[pp_k]]])
        inverse_bus = {int(value): int(bus) for bus, value in enumerate(lookups["bus"]) if int(value) >= 0}
        pp_order = [inverse_bus[int(ppc["branch"][pp_k, F_BUS])], inverse_bus[int(ppc["branch"][pp_k, T_BUS])]]
        if pp_order == [int(row.bus2), int(row.bus1)]:
            expected = expected[::-1, ::-1]
        rows.append({"identifier": str(row.idx), "kind": "transformer" if bool(row.trans) else "line", "from_bus": int(row.bus1), "to_bus": int(row.bus2), "source_tap": float(row.tap), "source_shift_degree": float(np.rad2deg(row.phi)), "pp_tap": float(ppc["branch"][pp_k, TAP]), "pp_shift_degree": float(ppc["branch"][pp_k, SHIFT]), "source_b_pu": float(row.b), "pp_b_pu": float(ppc["branch"][pp_k, BR_B]), "source_r_pu": float(row.r), "pp_r_pu": float(ppc["branch"][pp_k, BR_R]), "source_x_pu": float(row.x), "pp_x_pu": float(ppc["branch"][pp_k, BR_X]), "max_primitive_abs_error": float(np.max(np.abs(expected - actual))), "in_service": bool(ppc["branch"][pp_k, BR_STATUS])})
    return pd.DataFrame(rows)


def run(root: Path) -> dict[str, Any]:
    root = root.resolve()
    results = root / "output" / "results"
    reports = root / "output" / "reports"
    results.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    source = load_native_system()
    inventory = canonical_inventory(source)
    for name, frame in inventory.items():
        frame.to_csv(results / f"g0_canonical_{name}_table.csv", index=False)

    bus_ids = tuple(int(x) for x in inventory["bus"]["idx"])
    branches = _source_branches(inventory["branch"])
    shunts = {int(row.bus): complex(float(row.g) * float(row.Sn) / 100.0, float(row.b) * float(row.Sn) / 100.0) for row in inventory["shunt"].itertuples(index=False) if bool(row.u)}
    source.setup()
    from andes.linsolvers.scipy import spmatrix_to_csc
    source_y = spmatrix_to_csc(source.build_ybus()).toarray()
    expected_y = ybus(bus_ids, branches, shunts)
    net, conversion = solve_canonical_from_andes(return_net=True)
    target_y, ppc, lookups = _pp_ybus_before_pf(net)
    components_source = _component_matrices(bus_ids, branches, shunts)
    components_target = _pp_components(net, ppc, lookups)

    entries: list[dict[str, Any]] = []
    for i, bus_i in enumerate(bus_ids):
        for j, bus_j in enumerate(bus_ids):
            delta = source_y[i, j] - target_y[i, j]
            entries.append({"from_bus": bus_i, "to_bus": bus_j, "andes_real": source_y[i, j].real, "andes_imag": source_y[i, j].imag, "pandapower_real": target_y[i, j].real, "pandapower_imag": target_y[i, j].imag, "diff_real": delta.real, "diff_imag": delta.imag, "abs_diff": abs(delta)})
    pd.DataFrame(entries).to_csv(results / "g0_ybus_complex_entries.csv", index=False)

    matrix_rows = [{"component": "source_backend_vs_formula", "max_abs_diff": float(np.max(np.abs(source_y - expected_y))), "frobenius_norm": float(np.linalg.norm(source_y - expected_y))}, {"component": "source_backend_vs_pandapower", "max_abs_diff": float(np.max(np.abs(source_y - target_y))), "frobenius_norm": float(np.linalg.norm(source_y - target_y))}]
    for name in components_source:
        matrix_rows.append({"component": name, "max_abs_diff": float(np.max(np.abs(components_source[name] - components_target[name]))), "frobenius_norm": float(np.linalg.norm(components_source[name] - components_target[name]))})
    pd.DataFrame(matrix_rows).to_csv(results / "g0_ybus_matrix_diff.csv", index=False)
    primitive = _primitive_audit(inventory, ppc, lookups)
    primitive.to_csv(results / "g0_branch_primitive_admittances.csv", index=False)
    primitive[primitive.kind.eq("transformer")].to_csv(results / "g0_transformer_audit.csv", index=False)
    charging = primitive[primitive.kind.eq("line")][["identifier", "from_bus", "to_bus", "source_b_pu", "pp_b_pu", "max_primitive_abs_error"]].copy()
    charging.to_csv(results / "g0_shunt_charging_audit.csv", index=False)

    summary = {"status": "PASS" if float(np.max(np.abs(source_y - target_y))) <= 1e-10 and float(primitive.max_primitive_abs_error.max()) <= 1e-10 else "FAIL", "bus_count": len(bus_ids), "branch_count": len(branches), "transformer_count": int(inventory["transformer"].shape[0]), "shunt_count": int(inventory["shunt"].shape[0]), "max_source_formula_ybus_abs_error": matrix_rows[0]["max_abs_diff"], "max_ybus_abs_error": matrix_rows[1]["max_abs_diff"], "max_branch_primitive_abs_error": float(primitive.max_primitive_abs_error.max()), "conversion": conversion}
    (reports / "g0_ybus_audit.md").write_text("# G0-B Ybus and element audit\n\n" + f"Status: **{summary['status']}**\n\n" + "The ANDES `ieee39_full.xlsx` workbook is the authority. The target Ybus was reconstructed from the translated pandapower network before `runpp`; no solved state is used in this section.\n\n" + "```json\n" + json.dumps(summary, indent=2, sort_keys=True) + "\n```\n\n" + "The primitive rows prove line charging, transformer tap/phase conventions, and bus shunt signs element by element. `g0_ybus_complex_entries.csv` contains all 39x39 complex entries; `g0_ybus_matrix_diff.csv` contains component decomposition norms.\n", encoding="utf-8")
    return summary
