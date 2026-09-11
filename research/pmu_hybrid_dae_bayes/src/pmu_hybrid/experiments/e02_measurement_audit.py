"""Phase 3: freeze a controlled PMU branch-terminal operator and audit it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pmu_hybrid.cases.ieee39_andes import solve_static
from pmu_hybrid.constants import CONTROLLED_SYNTHETIC_TERMINAL_MAP, PMU_BUSES
from pmu_hybrid.physics.measurement import select_controlled_terminal_map, terminal_current
from pmu_hybrid.utils.manifests import base_manifest, write_json


def _upstream_static_status(root: Path) -> str:
    path = root / "output" / "manifests" / "static_parity.json"
    if not path.exists():
        return "NOT_RUN"
    record = json.loads(path.read_text(encoding="utf-8"))
    return str(record.get("status", "UNKNOWN"))


def run(root: Path) -> dict[str, object]:
    root = root.resolve()
    repository_root = root.parents[1]
    static = solve_static()
    mapping = select_controlled_terminal_map(static.branches)
    branches = {branch.identifier: branch for branch in static.branches}
    position = {bus: index for index, bus in enumerate(static.bus_ids)}
    voltage = static.voltage_pu * np.exp(1j * static.angle_rad)
    rows: list[dict[str, object]] = []
    for bus in PMU_BUSES:
        chosen = mapping[bus]
        branch = branches[chosen.branch_id]
        current = terminal_current(
            chosen, branch, voltage[position[branch.from_bus]], voltage[position[branch.to_bus]],
        )
        rows.append({
            **chosen.to_dict(),
            "current_magnitude_pu": float(abs(current)),
            "current_angle_deg": float(np.rad2deg(np.angle(current))),
            "formula_status": "PASS" if np.isfinite(current.real) and np.isfinite(current.imag) else "FAIL",
        })
    operator_pass = len(rows) == len(PMU_BUSES) and all(row["formula_status"] == "PASS" for row in rows)
    upstream = _upstream_static_status(root)
    status = "PASS" if operator_pass and upstream == "PASS" else "BLOCKED_BY_G0" if operator_pass else "FAIL"
    payload: dict[str, object] = {
        **base_manifest(repository_root, seed=20260911),
        "phase": "E02",
        "status": status,
        "upstream_static_parity_status": upstream,
        "measurement_scope": CONTROLLED_SYNTHETIC_TERMINAL_MAP,
        "competition_current_metadata_status": "NOT_AVAILABLE_TO_CAMPAIGN",
        "terminal_selection_rule": "lowest_canonical_incident_edge",
        "pmu_buses": list(PMU_BUSES),
        "operator_formula_pass": operator_pass,
        "terminals": rows,
    }
    manifests = root / "output" / "manifests"
    reports = root / "output" / "reports"
    results = root / "output" / "results" / "e02_measurement_audit"
    results.mkdir(parents=True, exist_ok=True)
    write_json(manifests / "pmu_terminal_map.json", payload)
    import pandas as pd
    table = pd.DataFrame(rows)
    table.to_csv(results / "pmu_terminal_map.csv", index=False)
    table.to_parquet(results / "pmu_terminal_map.parquet", index=False)
    (reports / "measurement_audit.md").write_text(
        "# PMU measurement operator audit\n\n"
        f"Status: **{status}**\n\n"
        "The map is a controlled synthetic convention, not a claim about organizer current terminals. "
        "Each current is the declared branch-terminal current leaving its named PMU bus.\n\n"
        f"```json\n{json.dumps(payload, indent=2, sort_keys=True)}\n```\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.root)
    print(json.dumps({"status": result["status"], "manifest": "output/manifests/pmu_terminal_map.json"}))
    if result["status"] == "FAIL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
