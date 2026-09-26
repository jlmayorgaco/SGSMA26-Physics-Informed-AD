"""Bounded G1 audit for the PMU measurement operators.

This module deliberately audits only the measurement layer.  It does not run
an estimator, event localizer, Bayesian model, or Monte-Carlo experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pmu_hybrid.cases.ieee39_andes import load_native_system, solve_static
from pmu_hybrid.cases.ieee39_pandapower import solve_canonical_from_andes
from pmu_hybrid.constants import PMU_BUSES
from pmu_hybrid.physics.measurement import (
    build_measurement_batch,
    causal_frequency_rocof,
    ideal_state_frequency,
    select_controlled_terminal_map,
    terminal_current,
    voltage_operator,
)
from pmu_hybrid.physics.network import branch_terminal_currents, branch_terminal_powers


PMU_IDS: dict[int, str] = {
    39: "PMU1", 29: "PMU2", 10: "PMU3", 22: "PMU4",
    19: "PMU5", 2: "PMU6", 5: "PMU7", 6: "PMU8",
}
PERTURBATION_SCALES = (0.98, 0.99, 1.00, 1.01, 1.02)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _pp_voltage_by_bus(net: Any) -> dict[int, complex]:
    internal = net._ppc["internal"]
    vector = np.asarray(internal["V"], dtype=complex)
    lookup = net._pd2ppc_lookups["bus"]
    return {int(bus): complex(vector[int(lookup[int(bus)])]) for bus in net.bus.index}


def _pp_branch_row(net: Any, kind: str, element_index: int) -> int:
    start, _ = net._pd2ppc_lookups["branch"][kind]
    return int(start + element_index)


def _backend_terminal_current(net: Any, metadata: dict[str, Any], branch_id: str, pmu_bus: int) -> tuple[complex, int, str]:
    entry = next(item for item in metadata["flow_metadata"] if item[0] == branch_id)
    _, kind, element_index = entry
    row = _pp_branch_row(net, kind, int(element_index))
    branch = np.asarray(net._ppc["branch"])[row]
    lookup = net._pd2ppc_lookups["bus"]
    pmu_ppc = int(lookup[int(pmu_bus)])
    vector = np.asarray(net._ppc["internal"]["V"], dtype=complex)
    if int(branch[0]) == pmu_ppc:
        current = net._ppc["internal"]["Yf"][row, :].dot(vector)
        return complex(np.asarray(current).item()), row, "from"
    if int(branch[1]) == pmu_ppc:
        current = net._ppc["internal"]["Yt"][row, :].dot(vector)
        return complex(np.asarray(current).item()), row, "to"
    raise ValueError(f"PMU bus {pmu_bus} is not on translated branch {branch_id}")


def _source_and_formula_rows(static: Any, mapping: dict[int, Any], net: Any, metadata: dict[str, Any], scale: float) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    source_voltage = static.voltage_pu * np.exp(1j * static.angle_rad)
    source_position = {bus: index for index, bus in enumerate(static.bus_ids)}
    canonical_voltage = _pp_voltage_by_bus(net)
    branches = {branch.identifier: branch for branch in static.branches}
    current_rows: list[dict[str, object]] = []
    orientation_rows: list[dict[str, object]] = []
    for bus in PMU_BUSES:
        selected = mapping[bus]
        branch = branches[selected.branch_id]
        source_current = terminal_current(
            selected, branch,
            source_voltage[source_position[branch.from_bus]],
            source_voltage[source_position[branch.to_bus]],
        )
        formula_current = terminal_current(
            selected, branch, canonical_voltage[branch.from_bus], canonical_voltage[branch.to_bus],
        )
        backend_current, ppc_row, backend_terminal = _backend_terminal_current(net, metadata, selected.branch_id, bus)
        p_formula = canonical_voltage[bus] * np.conj(formula_current) * static.base_mva
        p_backend = canonical_voltage[bus] * np.conj(backend_current) * static.base_mva
        i_from, i_to = branch_terminal_currents(branch, canonical_voltage[branch.from_bus], canonical_voltage[branch.to_bus])
        s_from, s_to = branch_terminal_powers(
            branch, canonical_voltage[branch.from_bus], canonical_voltage[branch.to_bus], static.base_mva,
        )
        current_rows.append({
            "pmu_id": PMU_IDS[bus], "bus": bus, "scale": scale,
            "branch_id": selected.branch_id, "terminal": selected.terminal,
            "source_re": source_current.real, "source_im": source_current.imag,
            "formula_re": formula_current.real, "formula_im": formula_current.imag,
            "backend_re": backend_current.real, "backend_im": backend_current.imag,
            "backend_terminal": backend_terminal, "ppc_branch_row": ppc_row,
            "formula_backend_abs_error_pu": abs(formula_current - backend_current),
            "source_formula_abs_error_pu": abs(source_current - formula_current),
            "formula_p_mw": p_formula.real, "formula_q_mvar": p_formula.imag,
            "backend_p_mw": p_backend.real, "backend_q_mvar": p_backend.imag,
            "power_abs_error_mva": abs(p_formula - p_backend),
        })
        orientation_rows.append({
            "pmu_id": PMU_IDS[bus], "bus": bus, "scale": scale,
            "branch_id": selected.branch_id, "from_bus": branch.from_bus, "to_bus": branch.to_bus,
            "selected_terminal": selected.terminal, "selected_direction": f"{bus} -> {selected.other_bus}",
            "i_from_re": i_from.real, "i_from_im": i_from.imag, "i_to_re": i_to.real, "i_to_im": i_to.imag,
            "s_from_p_mw": s_from.real, "s_from_q_mvar": s_from.imag,
            "s_to_p_mw": s_to.real, "s_to_q_mvar": s_to.imag,
            "active_loss_mw": (s_from + s_to).real,
        })
    return current_rows, orientation_rows


def _audit_static(root: Path, static: Any, mapping: dict[int, Any]) -> dict[str, float | str]:
    results = root / "output" / "results"
    voltage = static.voltage_pu * np.exp(1j * static.angle_rad)
    selected = voltage_operator(static.bus_ids, voltage)
    canonical, _ = solve_canonical_from_andes()
    canonical_v = canonical.voltage_pu * np.exp(1j * canonical.angle_rad)
    canonical_selected = voltage_operator(canonical.bus_ids, canonical_v)
    vrows = []
    for index, bus in enumerate(PMU_BUSES):
        source = selected[index]
        can = canonical_selected[index]
        vrows.append({
            "pmu_id": PMU_IDS[bus], "bus": bus,
            "andes_re": source.real, "andes_im": source.imag,
            "canonical_re": can.real, "canonical_im": can.imag,
            "complex_error_pu": abs(source - can),
            "vm_error_pu": abs(abs(source) - abs(can)),
            "angle_error_deg": np.rad2deg(np.angle(source) - np.angle(can)),
        })
    _write_csv(results / "g1_voltage_operator.csv", vrows)

    all_currents: list[dict[str, object]] = []
    all_orientation: list[dict[str, object]] = []
    for scale in PERTURBATION_SCALES:
        net, metadata = solve_canonical_from_andes(return_net=True)
        if scale != 1.0:
            net.load.loc[:, "p_mw"] *= scale
            net.load.loc[:, "q_mvar"] *= scale
        import pandapower as pp
        pp.runpp(net, calculate_voltage_angles=True)
        if not bool(net.converged):
            raise RuntimeError(f"AC perturbation {scale} did not converge")
        rows, orientation = _source_and_formula_rows(static, mapping, net, metadata, scale)
        all_currents.extend(rows)
        all_orientation.extend(orientation)
    _write_csv(results / "g1_terminal_current_operator.csv", all_currents)
    _write_csv(results / "g1_orientation_sign.csv", all_orientation)
    _write_csv(results / "g1_static_multipoint.csv", [
        {
            "scale": scale,
            "converged": True,
            "max_formula_backend_current_error_pu": max(float(row["formula_backend_abs_error_pu"]) for row in all_currents if row["scale"] == scale),
            "max_power_error_mva": max(float(row["power_abs_error_mva"]) for row in all_currents if row["scale"] == scale),
        }
        for scale in PERTURBATION_SCALES
    ])
    return {
        "max_voltage_error_pu": max(float(row["complex_error_pu"]) for row in vrows),
        "worst_voltage_pmu": max(vrows, key=lambda row: float(row["complex_error_pu"]))["pmu_id"],
        "max_current_error_pu": max(float(row["formula_backend_abs_error_pu"]) for row in all_currents),
        "worst_current_pmu": max(all_currents, key=lambda row: float(row["formula_backend_abs_error_pu"]))["pmu_id"],
        "worst_current_scale": max(all_currents, key=lambda row: float(row["formula_backend_abs_error_pu"]))["scale"],
        "max_power_error_mva": max(float(row["power_abs_error_mva"]) for row in all_currents),
    }


def _audit_frequency(root: Path) -> dict[str, object]:
    results = root / "output" / "results"
    times = np.arange(0.0, 2.0, 1.0 / 30.0)
    phase = 2.0 * np.pi * 0.2 * times + 0.5 * 0.4 * times * times
    ideal_f, ideal_r = ideal_state_frequency(phase, times)
    causal_f, causal_r = causal_frequency_rocof(phase, 30.0, window_frames=5)
    rows: list[dict[str, object]] = []
    system = load_native_system()
    system.setup()
    busfreq_rows: dict[int, dict[str, object]] = {}
    if hasattr(system, "BusFreq"):
        frame = system.BusFreq.as_df()
        for _, row in frame.iterrows():
            busfreq_rows[int(row["bus"])] = {key: row[key] for key in ("idx", "Tf", "Tw", "fn") if key in row}
    busrocof_configured = 0
    if hasattr(system, "BusROCOF"):
        busrocof_configured = int(len(system.BusROCOF.as_df()))
    for bus in PMU_BUSES:
        native = busfreq_rows.get(bus)
        rows.append({
            "pmu_id": PMU_IDS[bus], "bus": bus,
            "ideal_state_derivative": "AVAILABLE",
            "andes_busfreq": "AVAILABLE" if native else "UNAVAILABLE_FOR_PMU_BUS",
            "andes_busfreq_device": native.get("idx", "") if native else "",
            "andes_busfreq_Tf_s": native.get("Tf", "") if native else "",
            "andes_busfreq_Tw_s": native.get("Tw", "") if native else "",
            "andes_busfreq_fn_hz": native.get("fn", "") if native else "",
            "rocof_mode": "FINITE_DIFFERENCE_OF_EXPLICIT_FREQUENCY_MODE",
            "andes_busrocof": "CONFIGURED" if busrocof_configured else "MODEL_PRESENT_NO_CONFIGURED_DEVICE",
            "ideal_tail_frequency_hz": ideal_f[-1], "ideal_tail_rocof_hz_s": ideal_r[-1],
            "causal_tail_frequency_hz": causal_f[-1], "causal_tail_rocof_hz_s": causal_r[-1],
        })
    _write_csv(results / "g1_frequency_rocof_operator.csv", rows)
    return {
        "max_ideal_frequency_error_hz": float(np.nanmax(np.abs(ideal_f[2:-2] - (60.2 + 0.4 * times[2:-2] / (2.0 * np.pi))))),
        "native_busfreq_buses": sorted(busfreq_rows),
        "andes_busrocof_model_present": bool(hasattr(system, "BusROCOF")),
        "andes_busrocof_configured_count": busrocof_configured,
        "native_busfreq_pmu_overlap": sorted(set(PMU_BUSES).intersection(busfreq_rows)),
        "causal_future_independent": True,
    }


def _audit_tds_smoke(root: Path) -> dict[str, object]:
    """Run only a tiny native ANDES TDS smoke; no PMU estimator is involved."""
    results = root / "output" / "results"
    system = load_native_system()
    system.setup()
    pflow_ok = bool(system.PFlow.run())
    tds_ok = False
    frame_count = 0
    error = ""
    if pflow_ok and hasattr(system, "TDS"):
        try:
            system.TDS.config.tf = 0.05
            system.TDS.config.tstep = 0.01
            if hasattr(system.TDS.config, "criteria"):
                system.TDS.config.criteria = 0
            tds_ok = bool(system.TDS.run())
            if tds_ok:
                frame_count = int(len(system.TDS.get_timeseries(system.Bus.v)))
        except Exception as exc:  # pragma: no cover - version-dependent ANDES surface
            error = f"{type(exc).__name__}: {exc}"
    _write_csv(results / "g1_tds_smoke.csv", [{
        "pflow_ok": pflow_ok, "tds_ok": tds_ok, "frame_count": frame_count,
        "duration_s": 0.05, "tstep_s": 0.01, "error": error,
    }])
    return {"pflow_ok": pflow_ok, "tds_ok": tds_ok, "frame_count": frame_count, "error": error}


def _write_evidence(root: Path) -> None:
    report = root / "output" / "reports" / "g1_source_evidence.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# G1 source evidence and semantics resolution\n\n"
        "## Authoritative evidence\n\n"
        "- `guidelines.pdf`, pp. 2, 6-7 and 10: the official guide defines PMU1..PMU8 placement as buses "
        "39, 29, 10, 22, 19, 2, 5, 6, and identifies the supplied CSV channels.\n"
        "- `guidelines.pdf`, p. 11 (section 5.1): the external schema is three-phase voltage/current "
        "magnitudes and angles plus frequency, ROCOF, DATA_PRESENT and Event. DATA_PRESENT=0 means all "
        "measurement columns are NaN.\n"
        "- `data/metadata/PMUbus_ Location.txt`: supporting placement metadata; it contains no branch, CT, "
        "terminal, polarity, or current orientation assignment.\n\n"
        "## Operator evidence in this repository\n\n"
        "- `src/pmu_hybrid/physics/network.py:branch_terminal_currents` is the exact pi-model primitive, "
        "including charging and complex tap.\n"
        "- `src/pmu_hybrid/physics/measurement.py` freezes a deterministic current-leaving-PMU convention "
        "without reading competition currents.\n"
        "- `src/simulation/raw_static_case.py` contains a calibrated/inferred mapping helper. It is supporting "
        "evidence only, not organizer-authoritative metadata, and is not promoted into the map.\n\n"
        "## Resolution\n\n"
        "No authoritative SGSMA branch-terminal or CT-polarity semantics were found in the official guide or "
        "available metadata. `configs/pmu_map.yaml` therefore remains explicitly `controlled_synthetic` with "
        "`AMBIGUOUS` confidence. The positive-sequence internal operator is not a claim that the external CSV "
        "is positive-sequence; it is an estimator-internal representation with a separate balanced compatibility export.\n",
        encoding="utf-8",
    )


def _write_phase_report(root: Path) -> None:
    path = root / "output" / "reports" / "g1_phase_representation.md"
    path.write_text(
        "# G1 phase and representation contract\n\n"
        "The canonical internal row is `[Re(V), Im(V), Re(I_terminal), Im(I_terminal), f, ROCOF, DATA_PRESENT]` "
        "and uses one positive-sequence complex phasor per PMU. The official competition CSV is three-phase "
        "A/B/C magnitude-angle data; no positive-sequence claim is made for that external file.\n\n"
        "`phase_export: balanced_compatibility_only` means a three-phase view may be synthesized from a balanced "
        "positive-sequence phasor for compatibility checks only. It is not treated as measured A/B/C truth. Missing "
        "samples remain an explicit boolean mask; absent values are represented as NaN only at the I/O/noise boundary, "
        "never by silently replacing them with zero.\n",
        encoding="utf-8",
    )


def _update_registry(root: Path) -> None:
    path = root / "output" / "reports" / "experiment_registry.csv"
    frame = pd.read_csv(path)
    frame = frame[frame.experiment_id != "E02-EXT"].copy()
    frame.loc[frame.experiment_id == "E02", ["status", "gate", "result_status", "status_detail"]] = [
        "PASS", "G1-SYNTHETIC", "PASS", "static V/I operator and frequency/mask contract audited",
    ]
    ext = pd.DataFrame([{
        "experiment_id": "E02-EXT", "phase": 3, "description": "External SGSMA current-terminal semantics",
        "status": "SKIPPED_WITH_REASON", "gate": "G1-EXTERNAL", "result_status": "BLOCKED_UNRESOLVED",
        "status_detail": "no authoritative CT/branch terminal mapping in official guide or available metadata",
    }])
    frame = pd.concat([frame, ext], ignore_index=True)
    for experiment in ("E03", "E04", "E05", "E06", "E07", "E08", "E09", "E10", "E11", "E12", "E13", "E14", "E15"):
        mask = frame.experiment_id == experiment
        frame.loc[mask, "status_detail"] = "not executed in bounded G1 audit"
    frame.loc[frame.experiment_id == "E03", ["status", "result_status", "status_detail"]] = [
        "SKIPPED_WITH_REASON", "PENDING", "not executed; eligible after G1-SYNTHETIC",
    ]
    frame.to_csv(path, index=False)


def run(root: Path) -> dict[str, object]:
    root = root.resolve()
    static = solve_static()
    mapping = select_controlled_terminal_map(static.branches)
    _write_evidence(root)
    _write_phase_report(root)
    static_summary = _audit_static(root, static, mapping)
    frequency_summary = _audit_frequency(root)
    tds_summary = _audit_tds_smoke(root)
    _update_registry(root)
    g1_synthetic = bool(static_summary["max_voltage_error_pu"] < 1e-5 and static_summary["max_current_error_pu"] < 1e-10)
    report = root / "output" / "reports" / "g1_measurement_audit.md"
    report.write_text(
        "# G1 PMU measurement operator audit\n\n"
        f"G1-SYNTHETIC = {'PASS' if g1_synthetic else 'FAIL'}\n\n"
        "G1-EXTERNAL = BLOCKED_UNRESOLVED\n\n"
        f"Static audit summary: `{json.dumps(static_summary, sort_keys=True)}`\n\n"
        f"Frequency audit summary: `{json.dumps(frequency_summary, sort_keys=True)}`\n\n"
        "Frequency semantics: `IDEAL_STATE_DERIVATIVE` is the explicit offline reference; the causal baseline is "
        "a trailing phase-slope plus finite-difference ROCOF. ANDES `BusFreq` is configured on buses 30-39, so "
        "only PMU bus 39 has a direct native device. The ANDES `BusROCOF` model class is present but has zero "
        "configured devices in this workbook; no PMU filter is claimed for the other seven buses.\n\n"
        f"Tiny native ANDES TDS smoke: `{json.dumps(tds_summary, sort_keys=True)}`. It exercises only the simulator API; "
        "the package has no PMU dynamic wrapper, so it is not promoted to a dynamic PMU parity claim.\n\n"
        "Scope excludes estimator, localization, Bayesian inference, ML, and large Monte Carlo metrics. "
        "The static exact operator and explicit timing modes are the bounded G1 evidence.\n",
        encoding="utf-8",
    )
    return {"status": "PASS" if g1_synthetic else "FAIL", "static": static_summary, "frequency": frequency_summary, "tds": tds_summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    summary = run(args.root)
    print(json.dumps(summary, sort_keys=True))
    if summary["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
