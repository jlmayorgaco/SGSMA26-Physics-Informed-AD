"""One-command, gate-aware smoke runner for the implemented initial phases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pmu_hybrid.cases.parity import run as run_static_parity
from pmu_hybrid.experiments.e00_environment import run as run_environment
from pmu_hybrid.experiments.e02_measurement_audit import run as run_measurement
from pmu_hybrid.experiments.e03_scenario_smoke import run as run_scenario
from pmu_hybrid.utils.manifests import base_manifest, write_json


def run(root: Path) -> dict[str, object]:
    """Execute phases 1-4 in order and retain a non-cherry-picked summary."""
    root = root.resolve()
    environment = run_environment(root)
    parity = run_static_parity(root)
    measurement = run_measurement(root)
    scenario = run_scenario(root)
    statuses = {
        "E00": str(environment["status"]),
        "E01": str(parity["status"]),
        "E02": str(measurement["status"]),
        "E03": str(scenario["status"]),
    }
    payload: dict[str, object] = {
        **base_manifest(root.parents[1], seed=20260911),
        "phase": "SMOKE_1_4",
        "phase_statuses": statuses,
        "success_criteria": "E00 PASS; E01 is reported honestly; downstream DAE work remains gated by E01",
        "status": "PASS" if statuses["E00"] == "PASS" else "FAIL",
    }
    write_json(root / "output" / "reports" / "smoke_summary.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.root)
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
