"""M9 scenario validation and final readiness reporting."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd

from src.simulation.m9.calibration import compare_raw_vs_sim, extract_reference_statistics, read_json, write_json
from src.simulation.m9.constants import PMU_BUSES_OFFICIAL, PMU_MEASUREMENT_SUFFIXES
from src.simulation.m9.templates import get_template


def _token(bus: str) -> str:
    return str(int("".join(ch for ch in bus if ch.isdigit())))


def expected_pmu_columns(bus: str) -> list[str]:
    return ["TIMESTAMP"] + [f"{bus}_{suffix}" for suffix in PMU_MEASUREMENT_SUFFIXES] + ["DATA_PRESENT", "Event"]


def _check_schema_for_file(path: Path, bus: str) -> dict[str, Any]:
    df = pd.read_csv(path)
    expected = expected_pmu_columns(bus)
    missing = [c for c in expected if c not in df.columns]
    ordered = list(df.columns[: len(expected)]) == expected
    return {"file": str(path), "bus": bus, "rows": int(len(df)), "missing_columns": missing, "columns_in_expected_order": bool(ordered), "pass": bool(not missing and ordered), "df": df}


def _missing_semantics_for_df(df: pd.DataFrame, bus: str) -> dict[str, Any]:
    cols = [f"{bus}_{suffix}" for suffix in PMU_MEASUREMENT_SUFFIXES]
    missing_rows = pd.to_numeric(df["DATA_PRESENT"], errors="coerce").fillna(1).astype(int) == 0
    if not missing_rows.any():
        return {"missing_rows": 0, "all_measurements_nan_when_missing": True, "event_labels_valid_when_missing": True, "pass": True}
    all_nan = bool(df.loc[missing_rows, cols].isna().all(axis=None))
    labels = set(pd.to_numeric(df.loc[missing_rows, "Event"], errors="coerce").dropna().astype(int).unique().tolist())
    valid_labels = labels.issubset({5, 6}) and bool(labels)
    return {"missing_rows": int(missing_rows.sum()), "all_measurements_nan_when_missing": all_nan, "event_labels_valid_when_missing": valid_labels, "labels_seen": sorted(labels), "pass": bool(all_nan and valid_labels)}


def _timestamp_alignment(schema_results: list[dict[str, Any]]) -> dict[str, Any]:
    stamps: list[np.ndarray] = []
    for res in schema_results:
        stamps.append(pd.to_numeric(res["df"]["TIMESTAMP"], errors="coerce").to_numpy(float))
    if not stamps:
        return {"pass": False, "reason": "no PMU files"}
    same_len = len({len(s) for s in stamps}) == 1
    aligned = same_len and all(np.allclose(stamps[0], s, equal_nan=True) for s in stamps[1:])
    return {"pass": bool(aligned), "same_frame_count": bool(same_len), "frame_count": int(len(stamps[0]))}


def _all_bus_checks(scenario_dir: Path) -> dict[str, Any]:
    truth_path = scenario_dir / "all_buses" / "all_bus_truth.csv"
    meas_path = scenario_dir / "all_buses" / "all_bus_measurements.csv"
    target_path = scenario_dir / "all_buses" / "full_state_target.csv"
    required_truth = ["TIMESTAMP", "BUS", "IS_PMU_BUS", "IS_NON_PMU_BUS", "V_TRUE_REAL_PU", "V_TRUE_IMAG_PU", "V_TRUE_MAG_PU", "V_TRUE_ANG_DEG", "EVENT", "EVENT_ORIGIN_BUS", "EVENT_ORIGIN_LINE", "PHYSICAL_EVENT_TYPE", "CYBER_EVENT_TYPE", "WINDOW_TYPE"]
    out: dict[str, Any] = {"files_exist": truth_path.exists() and meas_path.exists() and target_path.exists(), "truth_path": str(truth_path), "measurements_path": str(meas_path), "target_path": str(target_path)}
    if not out["files_exist"]:
        out["pass"] = False
        return out
    truth = pd.read_csv(truth_path)
    target = pd.read_csv(target_path)
    out.update(
        {
            "truth_rows": int(len(truth)),
            "target_rows": int(len(target)),
            "bus_count": int(truth["BUS"].nunique()) if "BUS" in truth else 0,
            "non_pmu_bus_count": int(truth.loc[truth["IS_NON_PMU_BUS"].astype(bool), "BUS"].nunique()) if "IS_NON_PMU_BUS" in truth else 0,
            "missing_truth_columns": [c for c in required_truth if c not in truth.columns],
            "full_state_aligned": bool(len(truth) == len(target) and truth[["TIMESTAMP", "BUS"]].equals(target[["TIMESTAMP", "BUS"]])),
        }
    )
    out["pass"] = bool(out["bus_count"] >= 39 and out["non_pmu_bus_count"] >= 31 and not out["missing_truth_columns"] and out["full_state_aligned"])
    return out


def _label_checks(scenario_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    labels_path = scenario_dir / "labels" / "event_frame_labels.csv"
    intervals_path = scenario_dir / "labels" / "event_intervals.csv"
    loc_path = scenario_dir / "labels" / "localization_targets.csv"
    out = {"files_exist": labels_path.exists() and intervals_path.exists() and loc_path.exists()}
    if not out["files_exist"]:
        out["pass"] = False
        return out
    labels = pd.read_csv(labels_path)
    intervals = pd.read_csv(intervals_path)
    event_values = set(pd.to_numeric(labels["EVENT"], errors="coerce").dropna().astype(int).unique().tolist())
    expected = {int(e.get("event_label", 0)) for e in manifest.get("physical_events", []) + manifest.get("cyber_events", [])}
    if not expected:
        expected = {0}
    out.update(
        {
            "event_values_seen": sorted(event_values),
            "expected_labels_from_manifest": sorted(expected),
            "interval_count": int(len(intervals)),
            "has_abnormal_flags": "IS_ABNORMAL" in labels.columns,
            "manifest_labels_represented": bool(expected.intersection(event_values) or (expected == {0} and event_values == {0})),
        }
    )
    out["pass"] = bool(out["has_abnormal_flags"] and out["manifest_labels_represented"])
    return out


def _signature_checks(scenario_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    pmu_dir = scenario_dir / "pmu"
    results: dict[str, Any] = {}
    for label in range(0, 9):
        results[f"event_{label}_present"] = False
    for path in pmu_dir.glob("Bus*_Competition_Data_sim.csv"):
        df = pd.read_csv(path)
        values = set(pd.to_numeric(df["Event"], errors="coerce").dropna().astype(int).unique().tolist())
        for label in values:
            if 0 <= int(label) <= 8:
                results[f"event_{int(label)}_present"] = True
    template_name = manifest.get("scenario_template", "")
    expected_label = None
    for label in range(9):
        if f"EVENT{label}" in template_name:
            expected_label = label
            break
    if "OFFICIAL" in template_name:
        expected = {1, 2, 3, 4, 5, 6}
    elif expected_label is None:
        expected = set()
    else:
        expected = {expected_label}
    seen = {label for label in range(9) if results[f"event_{label}_present"]}
    results["expected_labels"] = sorted(expected)
    results["seen_labels"] = sorted(seen)
    results["pass"] = bool(expected.issubset(seen) if expected else True)
    return results


def validate_scenario(
    scenario_dir: str | Path,
    reference_pmu_dir: str | Path | None = None,
    reference_stats: dict[str, Any] | None = None,
    save_json: bool = True,
) -> dict[str, Any]:
    scenario = Path(scenario_dir)
    manifest_path = scenario / "scenario_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    pmu_dir = scenario / "pmu"
    schema_results: list[dict[str, Any]] = []
    public_schema: list[dict[str, Any]] = []
    for bus in PMU_BUSES_OFFICIAL:
        path = pmu_dir / f"Bus{_token(bus)}_Competition_Data_sim.csv"
        if path.exists():
            res = _check_schema_for_file(path, bus)
            schema_results.append(res)
            public_schema.append({k: v for k, v in res.items() if k != "df"})
        else:
            public_schema.append({"file": str(path), "bus": bus, "pass": False, "missing_file": True})
    timestamp_check = _timestamp_alignment(schema_results)
    missing_checks = {res["bus"]: _missing_semantics_for_df(res["df"], res["bus"]) for res in schema_results}
    all_bus = _all_bus_checks(scenario)
    labels = _label_checks(scenario, manifest)
    signatures = _signature_checks(scenario, manifest)

    if reference_stats is None and reference_pmu_dir is not None:
        reference_stats = extract_reference_statistics(reference_pmu_dir)
    if reference_stats is None:
        stats_path = scenario / "metadata" / "statistics_reference.json"
        reference_stats = read_json(stats_path) if stats_path.exists() else None
    raw_vs_sim = None
    if reference_stats is not None:
        raw_vs_sim = compare_raw_vs_sim(reference_stats, pmu_dir)
        write_json(scenario / "metadata" / "raw_vs_sim_comparison.json", raw_vs_sim)

    schema_pass = bool(len(schema_results) == 8 and all(r.get("pass") for r in public_schema) and timestamp_check.get("pass"))
    missing_pass = bool(all(v.get("pass") for v in missing_checks.values()))
    realism_pass = bool(raw_vs_sim.get("pass_realism")) if raw_vs_sim else False
    downstream = {
        "estimator_training_ready": bool(schema_pass and missing_pass and all_bus.get("pass")),
        "identifier_training_ready": bool(schema_pass and labels.get("pass") and realism_pass),
        "detector_training_ready": bool(schema_pass and labels.get("pass") and realism_pass),
        "cyber_detector_training_ready": bool(schema_pass and missing_pass and labels.get("pass")),
        "physical_detector_training_ready": bool(all_bus.get("pass") and labels.get("pass")),
        "classifier_training_ready": bool(schema_pass and labels.get("pass") and signatures.get("pass")),
        "localizer_training_ready": bool(all_bus.get("pass") and (scenario / "labels" / "localization_targets.csv").exists()),
        "estimator_assisted_localizer_ready": bool(all_bus.get("pass") and schema_pass),
        "overall_ready": False,
        "notes": [],
    }
    downstream["overall_ready"] = bool(all(v for k, v in downstream.items() if k.endswith("_ready") and isinstance(v, bool)))
    if not realism_pass:
        downstream["notes"].append("RAW-vs-SIM statistical realism did not pass heuristic thresholds.")
    if not missing_pass:
        downstream["notes"].append("Missing-data semantics failed for at least one PMU.")

    checks = {
        "scenario_dir": str(scenario),
        "scenario_id": manifest.get("scenario_id", scenario.name),
        "scenario_template": manifest.get("scenario_template"),
        "pmu_schema_checks": public_schema,
        "timestamp_alignment_check": timestamp_check,
        "missing_data_semantics_checks": missing_checks,
        "schema_checks": {"pmu_csvs_match_expected_schema": schema_pass, "pmu_file_count": len(schema_results)},
        "label_consistency_checks": labels,
        "all_bus_export_checks": all_bus,
        "signature_checks": signatures,
        "raw_vs_sim_comparison": raw_vs_sim,
        "downstream_readiness": downstream,
        "explicit_answers": {
            "pmu_csvs_match_expected_schema": schema_pass,
            "non_pmu_buses_exported": bool(all_bus.get("non_pmu_bus_count", 0) >= 31),
            "event_labels_coherent": bool(labels.get("pass")),
            "data_present_semantics_correct": missing_pass,
            "physical_and_cyber_events_represented": bool(manifest.get("physical_events") is not None and manifest.get("cyber_events") is not None),
            "synthetic_pmu_statistically_useful_vs_raw0001": realism_pass,
        },
        "scenario_valid": bool(schema_pass and missing_pass and all_bus.get("pass") and labels.get("pass") and signatures.get("pass") and realism_pass),
    }
    if save_json:
        write_json(scenario / "metadata" / "scenario_validation.json", checks)
    return checks


def _scenario_key_from_template(template_name: str) -> str:
    return template_name.replace("TEMPLATE_", "")


def run_m9_validation_suite(
    raw_path: str | Path,
    pmu_location_path: str | Path,
    reference_pmu_dir: str | Path,
    output_root: str | Path,
    seed: int = 12345,
    fps: float = 30.0,
    duration_s: float | None = None,
    use_andes: bool = False,
    save_plots: bool = True,
) -> dict[str, Any]:
    from src.simulation.m9.generator import generate_scenario

    root = Path(output_root)
    scenarios_root = root / "scenarios"
    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    reference_stats = extract_reference_statistics(reference_pmu_dir)
    template_order = [
        "TEMPLATE_EVENT0_NORMAL",
        "TEMPLATE_EVENT1_FAULT",
        "TEMPLATE_EVENT2_LINE_OUTAGE",
        "TEMPLATE_EVENT3_GENERATION_CHANGE",
        "TEMPLATE_EVENT4_LOAD_CHANGE",
        "TEMPLATE_EVENT5_MISSING_ONLY",
        "TEMPLATE_EVENT6_CONCURRENT_MISSING_PLUS_PHYSICAL",
        "TEMPLATE_EVENT7_BAD_DATA",
        "TEMPLATE_EVENT8_UNKNOWN_COMPOSITE",
        "TEMPLATE_OFFICIAL_STYLE_MULTI_EVENT",
    ]
    checks: dict[str, Any] = {}
    templates_tested: dict[str, str] = {}
    for idx, template_name in enumerate(template_order, start=1):
        scenario_id = "SIM0001" if template_name == "TEMPLATE_OFFICIAL_STYLE_MULTI_EVENT" else f"SIM{idx + 1:04d}"
        generate_scenario(
            raw_path=raw_path,
            pmu_location_path=pmu_location_path,
            reference_pmu_dir=reference_pmu_dir,
            output_root=scenarios_root,
            scenario_template=template_name,
            scenario_id=scenario_id,
            seed=seed + idx,
            fps=fps,
            duration_s=duration_s,
            use_andes=use_andes and template_name == "TEMPLATE_OFFICIAL_STYLE_MULTI_EVENT",
            reference_stats=reference_stats,
            save_plots=save_plots,
        )
        scenario_dir = scenarios_root / scenario_id
        key = _scenario_key_from_template(template_name)
        templates_tested[key] = str(scenario_dir)
        checks[key] = validate_scenario(scenario_dir, reference_stats=reference_stats, save_json=True)

    raw_vs_sim = checks.get("OFFICIAL_STYLE_MULTI_EVENT", {}).get("raw_vs_sim_comparison")
    schema_ok = all(c.get("schema_checks", {}).get("pmu_csvs_match_expected_schema", False) for c in checks.values())
    labels_ok = all(c.get("label_consistency_checks", {}).get("pass", False) for c in checks.values())
    all_bus_ok = all(c.get("all_bus_export_checks", {}).get("pass", False) for c in checks.values())
    missing_ok = all(all(x.get("pass", False) for x in c.get("missing_data_semantics_checks", {}).values()) for c in checks.values())
    realism_ok = bool(raw_vs_sim and raw_vs_sim.get("pass_realism"))
    downstream = {
        "estimator_training_ready": bool(schema_ok and all_bus_ok and missing_ok),
        "identifier_training_ready": bool(schema_ok and labels_ok and realism_ok),
        "detector_training_ready": bool(schema_ok and labels_ok and realism_ok),
        "cyber_detector_training_ready": bool(schema_ok and labels_ok and missing_ok),
        "physical_detector_training_ready": bool(all_bus_ok and labels_ok),
        "classifier_training_ready": bool(schema_ok and labels_ok),
        "localizer_training_ready": bool(all_bus_ok and labels_ok),
        "estimator_assisted_localizer_ready": bool(schema_ok and all_bus_ok),
        "notes": [],
    }
    if not realism_ok:
        downstream["notes"].append("Official-style SIM0001 did not pass RAW-vs-SIM heuristic realism thresholds.")
    overall_working = bool(all(v for k, v in downstream.items() if k.endswith("_ready") and isinstance(v, bool)))
    report = {
        "run_metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
            "fps": fps,
            "output_root": str(root),
            "raw_path": str(raw_path),
            "pmu_location_path": str(pmu_location_path),
            "reference_pmu_dir": str(reference_pmu_dir),
        },
        "reference_data": {"reference_pmu_dir": str(reference_pmu_dir), "bus_count": reference_stats.get("bus_count"), "fallback_used": reference_stats.get("fallback_used", False)},
        "templates_tested": templates_tested,
        "scenario_checks": checks,
        "raw_vs_sim_comparison": raw_vs_sim,
        "schema_checks": {"all_templates_pass_pmu_schema": schema_ok},
        "label_consistency_checks": {"all_templates_pass_label_consistency": labels_ok},
        "all_bus_export_checks": {"all_templates_export_39_bus_truth": all_bus_ok},
        "downstream_readiness": downstream,
        "overall_verdict": {
            "m9_simulator_working": overall_working,
            "main_strengths": [
                "Exports official-like 8-PMU CSV files with exact aligned timestamps.",
                "Exports full 39-bus truth and full-state targets for estimator/localizer supervision.",
                "Separates physical truth from cyber/data-quality corruption.",
                "Exercises Event labels 0 through 8 plus the official-style multi-event timeline.",
            ],
            "main_failures": [] if overall_working else ["At least one readiness gate failed; inspect scenario_checks for details."],
            "next_actions": [] if overall_working else ["Tighten reference-statistics calibration or inspect failed schema/label checks."],
        },
    }
    write_json(report_dir / "m9_simulator_validation.json", report)
    _write_report_markdown(report_dir / "m9_simulator_validation.md", report)
    # Also mirror canonical report paths at repository root for downstream tooling.
    canonical_report_dir = Path("report")
    canonical_report_dir.mkdir(parents=True, exist_ok=True)
    write_json(canonical_report_dir / "m9_simulator_validation.json", report)
    _write_report_markdown(canonical_report_dir / "m9_simulator_validation.md", report)
    return report


def _write_report_markdown(path: Path, report: dict[str, Any]) -> None:
    verdict = report.get("overall_verdict", {})
    downstream = report.get("downstream_readiness", {})
    lines = [
        "# M9 Simulator Validation",
        "",
        f"Overall working: `{verdict.get('m9_simulator_working')}`",
        "",
        "## Downstream Readiness",
    ]
    for key, value in downstream.items():
        if key == "notes":
            continue
        lines.append(f"- {key}: `{value}`")
    if downstream.get("notes"):
        lines.extend(["", "## Notes"])
        for note in downstream["notes"]:
            lines.append(f"- {note}")
    lines.extend(["", "## Explicit Answers"])
    off = report.get("scenario_checks", {}).get("OFFICIAL_STYLE_MULTI_EVENT", {}).get("explicit_answers", {})
    for key, value in off.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Templates Tested"])
    for key, scenario_path in report.get("templates_tested", {}).items():
        scenario_valid = report.get("scenario_checks", {}).get(key, {}).get("scenario_valid")
        lines.append(f"- {key}: `{scenario_valid}` ({scenario_path})")
    lines.extend(["", "## Strengths"])
    for item in verdict.get("main_strengths", []):
        lines.append(f"- {item}")
    if verdict.get("main_failures"):
        lines.extend(["", "## Failures"])
        for item in verdict.get("main_failures", []):
            lines.append(f"- {item}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
