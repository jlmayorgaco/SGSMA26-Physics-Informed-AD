"""Phase 4: run deterministic scenario/configuration/noise API smoke checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pmu_hybrid.constants import NOMINAL_FREQUENCY_HZ, PMU_BUSES
from pmu_hybrid.physics.configuration import ConfigurationJump, EventFamily
from pmu_hybrid.simulator.noise import MeasurementNoiseSpec, corrupt_measurements
from pmu_hybrid.simulator.scenario import PMUDropout, ScenarioSpec, configuration_trajectory, data_present_mask
from pmu_hybrid.utils.manifests import base_manifest, write_json


def _static_status(root: Path) -> str:
    source = root / "output" / "manifests" / "static_parity.json"
    return json.loads(source.read_text(encoding="utf-8")).get("status", "UNKNOWN") if source.exists() else "NOT_RUN"


def _phase_result(root: Path, filename: str) -> str:
    """Read a prior phase result without assuming a successful completion."""
    source = root / "output" / "manifests" / filename
    return str(json.loads(source.read_text(encoding="utf-8")).get("status", "UNKNOWN")) if source.exists() else "NOT_RUN"


def _write_registry(root: Path, results: dict[str, str], target: Path) -> None:
    """Materialize the no-cherry-picking registry with every planned phase."""
    registry = pd.read_csv(root / "configs" / "experiment_registry.csv")
    registry["result_status"] = registry.experiment_id.map(results).fillna("PENDING")
    registry["status"] = registry.result_status.map({
        "PASS": "PASS",
        "FAIL": "FAIL",
        "BLOCKED_BY_G0": "SKIPPED_WITH_REASON",
        "PASS_API_ONLY_BLOCKED_BY_G0": "SKIPPED_WITH_REASON",
    }).fillna("PENDING")
    registry["status_detail"] = registry.result_status.map({
        "BLOCKED_BY_G0": "operator verified but G0 static parity failed",
        "PASS_API_ONLY_BLOCKED_BY_G0": "API passed; DAE execution blocked by G0",
    }).fillna("")
    registry.to_csv(target, index=False)


def smoke_scenarios() -> tuple[ScenarioSpec, ...]:
    """Cover normal, persistent, transient, simultaneous and integrity mechanisms."""
    common = dict(seed=20260911, operating_point_id="OP_SMOKE_A", model_seed=71, duration_s=1.0, sample_rate_hz=30.0)
    return (
        ScenarioSpec(scenario_id="SMOKE_NORMAL", **common),
        ScenarioSpec(
            scenario_id="SMOKE_LINE", **common,
            physical_jumps=(ConfigurationJump("line_trip_1_2", EventFamily.LINE, "Line_1", 0.2, line_in_service=False),),
        ),
        ScenarioSpec(
            scenario_id="SMOKE_FAULT", **common,
            physical_jumps=(ConfigurationJump("fault_bus3", EventFamily.FAULT, "BUS3", 0.3, amplitude=complex(0.0, 5.0), duration_s=0.1),),
        ),
        ScenarioSpec(
            scenario_id="SMOKE_SIMULTANEOUS_GEN_LOAD", **common,
            physical_jumps=(
                ConfigurationJump("gen_step_bus30", EventFamily.GENERATION, "PV:30", 0.4, amplitude=-0.1),
                ConfigurationJump("load_step_bus3", EventFamily.LOAD, "PQ:3", 0.4, amplitude=0.2),
            ),
            pmu_dropouts=(PMUDropout(29, 0.6, 0.8),),
        ),
    )


def run(root: Path) -> dict[str, object]:
    root = root.resolve()
    repository_root = root.parents[1]
    scenarios = smoke_scenarios()
    records: list[dict[str, object]] = []
    for spec in scenarios:
        trajectory = configuration_trajectory(spec)
        mask = data_present_mask(spec, PMU_BUSES)
        baseline_shape = mask.shape
        baseline_voltage = np.ones(baseline_shape, dtype=complex)
        baseline_current = np.zeros(baseline_shape, dtype=complex)
        baseline_frequency = np.full(baseline_shape, NOMINAL_FREQUENCY_HZ)
        baseline_rocof = np.zeros(baseline_shape)
        corrupted = corrupt_measurements(
            baseline_voltage, baseline_current, baseline_frequency, baseline_rocof, mask,
            MeasurementNoiseSpec(ar1=0.3, cross_pmu_correlation=0.2, outlier_probability=0.01),
            seed=spec.scenario_seed, namespace="measurement-smoke",
        )
        replay = corrupt_measurements(
            baseline_voltage, baseline_current, baseline_frequency, baseline_rocof, mask,
            MeasurementNoiseSpec(ar1=0.3, cross_pmu_correlation=0.2, outlier_probability=0.01),
            seed=spec.scenario_seed, namespace="measurement-smoke",
        )
        records.append({
            **spec.to_dict(),
            "fingerprint": spec.fingerprint,
            "configuration_frames": len(trajectory),
            "final_active_line_count": sum(not active for active in trajectory[-1].line_status.values()),
            "final_active_fault_count": len(trajectory[-1].fault_shunts_pu),
            "missing_frame_count": int((~corrupted.data_present).sum()),
            "deterministic_noise_replay": bool(np.array_equal(corrupted.voltage_pu, replay.voltage_pu, equal_nan=True)),
            "execution_scope": "API_SMOKE_ONLY_NO_DAE_TRAJECTORY",
        })
    api_pass = all(record["deterministic_noise_replay"] for record in records)
    upstream = str(_static_status(root))
    status = "PASS" if api_pass and upstream == "PASS" else "PASS_API_ONLY_BLOCKED_BY_G0" if api_pass else "FAIL"
    payload: dict[str, object] = {
        **base_manifest(repository_root, seed=20260911),
        "phase": "E03",
        "status": status,
        "upstream_static_parity_status": upstream,
        "api_pass": api_pass,
        "scenario_count": len(records),
        "scenario_records": records,
        "rejected_simulations": [],
        "simulator_execution": "BLOCKED_UNTIL_G0_PASS",
    }
    manifests = root / "output" / "manifests"
    reports = root / "output" / "reports"
    results = root / "output" / "results" / "e03_scenario_smoke"
    rejected = root / "output" / "rejected"
    results.mkdir(parents=True, exist_ok=True)
    rejected.mkdir(parents=True, exist_ok=True)
    write_json(manifests / "scenario_smoke.json", payload)
    table = pd.DataFrame(records)
    table.to_csv(results / "scenario_registry.csv", index=False)
    table.to_parquet(results / "scenario_registry.parquet", index=False)
    pd.DataFrame(columns=["scenario_id", "reason_code", "detail"]).to_csv(rejected / "e03_rejected.csv", index=False)
    _write_registry(root, {
        "E00": _phase_result(root, "environment_audit.json"),
        "E01": _phase_result(root, "static_parity.json"),
        "E02": _phase_result(root, "pmu_terminal_map.json"),
        "E03": status,
    }, reports / "experiment_registry.csv")
    (reports / "scenario_smoke.md").write_text(
        "# Scenario/event/noise API smoke\n\n"
        f"Status: **{status}**\n\n"
        "The configuration and integrity APIs are deterministic and persisted. No DAE trajectory was produced while G0 is failed; this is not a simulator result or estimator score.\n\n"
        f"```json\n{json.dumps(payload, indent=2, sort_keys=True)}\n```\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.root)
    print(json.dumps({"status": result["status"], "manifest": "output/manifests/scenario_smoke.json"}))
    if result["status"] == "FAIL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
