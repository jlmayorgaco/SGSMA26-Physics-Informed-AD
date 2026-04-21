"""M2 RAW-informed cyber pipeline: fit, generate, validate, transfer-check."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

from src.simulation.m9.raw_informed_cyber_layer import RawInformedCyberLayer
from src.simulation.m9.raw_informed_generation import generate_raw_informed_dataset
from src.simulation.m9.raw_informed_parameters import extract_raw_informed_parameters
from src.simulation.m9.raw_informed_validation import evaluate_raw_vs_sim


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _safe_metric(payload: dict[str, Any], key: str) -> float:
    try:
        return float(payload.get("metrics", {}).get("global", {}).get(key, 0.0) or 0.0)
    except Exception:
        return 0.0


def _run_cmd(command: list[str], *, cwd: Path) -> dict[str, Any]:
    proc = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    return {
        "command": command,
        "returncode": int(proc.returncode),
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-60:]),
        "stderr_tail": "\n".join(proc.stderr.splitlines()[-60:]),
    }


def _load_or_build_before_holdout(
    *,
    workspace_root: Path,
    baseline_model_path: Path,
    baseline_threshold_path: Path,
    raw_holdout_root: Path,
    fallback_output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    existing = workspace_root / "output" / "detector_raw_holdout_eval" / "metrics" / "raw_holdout_report.json"
    if existing.exists():
        return _load_json(existing), None
    cmd = [
        sys.executable,
        "-m",
        "src.pipelines.m10_raw_holdout_eval",
        "--raw-input-root",
        str(raw_holdout_root),
        "--model-path",
        str(baseline_model_path),
        "--threshold-config-path",
        str(baseline_threshold_path),
        "--output-root",
        str(fallback_output_dir),
        "--run-name",
        "m2_before",
    ]
    run_meta = _run_cmd(cmd, cwd=workspace_root)
    report_path = fallback_output_dir / "metrics" / "raw_holdout_report.json"
    if report_path.exists():
        return _load_json(report_path), run_meta
    return {"metrics": {"global": {}}, "verdict": {"usable_on_raw": False}}, run_meta


def run_m2_raw_informed_cyber(
    *,
    workspace_root: Path,
    output_root: Path,
    chunks_root: Path,
    raw_path: Path,
    pmu_location_path: Path,
    reference_pmu_dir: Path,
    raw_holdout_root: Path,
    baseline_model_path: Path,
    baseline_threshold_path: Path,
    seed: int = 20260421,
    event5_count: int = 30,
    event7_count: int = 30,
    mixed_count: int = 40,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    metadata_dir = output_root / "metadata"
    metrics_dir = output_root / "metrics"
    report_dir = output_root / "report"
    plots_dir = output_root / "plots"
    for folder in (metadata_dir, metrics_dir, report_dir, plots_dir):
        folder.mkdir(parents=True, exist_ok=True)

    parameters = extract_raw_informed_parameters(
        chunks_root=chunks_root,
        output_metadata_dir=metadata_dir,
    )
    layer = RawInformedCyberLayer(
        event5_params=parameters["event5_params"],
        event7_params=parameters["event7_params"],
        event0_noise_baseline=parameters["event0_noise_baseline"],
        seed=seed,
        apply_shared_noise=True,
    )

    generated = generate_raw_informed_dataset(
        workspace_root=workspace_root,
        output_root=output_root,
        raw_path=raw_path,
        pmu_location_path=pmu_location_path,
        reference_pmu_dir=reference_pmu_dir,
        layer=layer,
        event5_count=event5_count,
        event7_count=event7_count,
        mixed_count=mixed_count,
        seed=seed,
    )

    validation = evaluate_raw_vs_sim(
        chunks_root=chunks_root,
        generated_manifest_path=generated["manifest_path"],
        workspace_root=workspace_root,
        output_root=output_root,
    )

    before_report, before_cmd = _load_or_build_before_holdout(
        workspace_root=workspace_root,
        baseline_model_path=baseline_model_path,
        baseline_threshold_path=baseline_threshold_path,
        raw_holdout_root=raw_holdout_root,
        fallback_output_dir=output_root / "raw_holdout_before_eval",
    )

    detector_after_root = output_root / "detector_after_cyber_fix"
    train_cmd = [
        sys.executable,
        "-m",
        "src.pipelines.m10_train_detector",
        "--workspace-root",
        str(workspace_root),
        "--train-split",
        str(generated["train_csv"]),
        "--val-split",
        str(generated["val_csv"]),
        "--test-split",
        str(generated["test_csv"]),
        "--output-root",
        str(detector_after_root),
        "--raw-path",
        str(raw_path),
        "--pmu-location-path",
        str(pmu_location_path),
        "--reference-pmu-dir",
        str(reference_pmu_dir),
        "--window-size",
        "64",
        "--window-stride",
        "16",
        "--skip-targeted-ready-data",
        "--skip-rebuild-splits-v3",
        "--preprocessing-modes",
        "ffill_bfill",
    ]
    train_run = _run_cmd(train_cmd, cwd=workspace_root)

    model_after = detector_after_root / "models" / "detector_model.pkl"
    threshold_after = detector_after_root / "config" / "threshold_config_v4.json"
    holdout_after_run: dict[str, Any] = {"returncode": -1, "stdout_tail": "", "stderr_tail": "", "command": []}
    after_report: dict[str, Any] = {"metrics": {"global": {}}, "verdict": {"usable_on_raw": False}}
    holdout_after_root = output_root / "raw_holdout_after_eval"
    if model_after.exists() and threshold_after.exists():
        holdout_after_cmd = [
            sys.executable,
            "-m",
            "src.pipelines.m10_raw_holdout_eval",
            "--raw-input-root",
            str(raw_holdout_root),
            "--model-path",
            str(model_after),
            "--threshold-config-path",
            str(threshold_after),
            "--output-root",
            str(holdout_after_root),
            "--run-name",
            "m2_after",
        ]
        holdout_after_run = _run_cmd(holdout_after_cmd, cwd=workspace_root)
        after_path = holdout_after_root / "metrics" / "raw_holdout_report.json"
        if after_path.exists():
            after_report = _load_json(after_path)
            shutil.copy2(after_path, metrics_dir / "raw_holdout_after_cyber_fix.json")

    # If the holdout output was not created, still materialize a placeholder file.
    raw_holdout_after_json_path = metrics_dir / "raw_holdout_after_cyber_fix.json"
    if not raw_holdout_after_json_path.exists():
        _write_json(raw_holdout_after_json_path, after_report)

    event5_match = validation["event5_summary"]
    event7_match = validation["event7_summary"]
    before_global = before_report.get("metrics", {}).get("global", {})
    after_global = after_report.get("metrics", {}).get("global", {})
    before_recall = float(before_global.get("recall_abnormal", 0.0) or 0.0)
    after_recall = float(after_global.get("recall_abnormal", 0.0) or 0.0)
    before_f1 = float(before_global.get("f1_abnormal", 0.0) or 0.0)
    after_f1 = float(after_global.get("f1_abnormal", 0.0) or 0.0)
    before_fp = float(before_global.get("false_positives_per_minute", 0.0) or 0.0)
    after_fp = float(after_global.get("false_positives_per_minute", 0.0) or 0.0)

    realistic_enough = bool(event5_match["acceptable_for_training"] and event7_match["acceptable_for_training"])
    improved = bool((after_recall > before_recall) or (after_f1 > before_f1) or (after_fp < before_fp))
    usable = bool(realistic_enough and improved and after_recall > 0.0)

    gaps: list[str] = []
    if not event5_match["acceptable_for_training"]:
        gaps.append("Event 5 RAW-vs-SIM match is still weak on one or more dropout metrics.")
    if not event7_match["acceptable_for_training"]:
        gaps.append("Event 7 RAW-vs-SIM match is still weak on one or more corruption metrics.")
    if after_recall <= before_recall:
        gaps.append("RAW holdout recall did not improve after retraining with RAW-informed cyber scenarios.")
    if after_f1 <= before_f1:
        gaps.append("RAW holdout F1 did not improve relative to baseline.")
    if after_fp > before_fp:
        gaps.append("RAW false positives per minute increased after retraining.")
    if not gaps:
        gaps.append("No major transfer blockers detected in this run.")

    transfer_report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_gap_addressed": {
            "sim_to_raw_failure_before": {
                "recall_abnormal": before_recall,
                "f1_abnormal": before_f1,
                "false_positives_per_minute": before_fp,
            },
            "raw_informed_changes_applied": {
                "event5_modeled_separately": True,
                "event7_modeled_separately": True,
                "shared_noise_model": "student_t_per_channel",
                "global_local_state_model": True,
                "targeted_families_generated": generated["counts_by_family"],
            },
        },
        "event5_match_quality": event5_match,
        "event7_match_quality": event7_match,
        "detector_raw_holdout_before": before_global,
        "detector_raw_holdout_after": after_global,
        "execution": {
            "before_holdout_command": before_cmd,
            "train_command": train_run,
            "after_holdout_command": holdout_after_run,
        },
        "verdict": {
            "cyber_layer_realistic_enough_for_training": realistic_enough,
            "raw_transfer_improved": improved,
            "usable_for_detector_training": usable,
            "main_remaining_gaps": gaps,
            "next_actions": [
                "Increase RAW Event 7 support and recalibrate mode persistence if Event 7 match stays below target.",
                "Tune Event 5 partial/full state transitions per PMU if dropout burst mismatch remains high.",
                "Run another synthetic detector cycle and re-check RAW holdout before classifier/localizer integration.",
            ],
        },
    }
    transfer_json_path = report_dir / "transfer_readiness_report.json"
    _write_json(transfer_json_path, transfer_report)
    _write_md(
        report_dir / "transfer_readiness_report.md",
        [
            "# Transfer Readiness Report",
            "",
            f"- cyber_layer_realistic_enough_for_training: `{transfer_report['verdict']['cyber_layer_realistic_enough_for_training']}`",
            f"- raw_transfer_improved: `{transfer_report['verdict']['raw_transfer_improved']}`",
            f"- usable_for_detector_training: `{transfer_report['verdict']['usable_for_detector_training']}`",
            f"- event5_match_pass_rate: `{event5_match['pass_rate']:.4f}`",
            f"- event7_match_pass_rate: `{event7_match['pass_rate']:.4f}`",
            f"- raw_before_recall: `{before_recall:.4f}`",
            f"- raw_after_recall: `{after_recall:.4f}`",
            f"- raw_before_f1: `{before_f1:.4f}`",
            f"- raw_after_f1: `{after_f1:.4f}`",
            f"- raw_before_fp_min: `{before_fp:.4f}`",
            f"- raw_after_fp_min: `{after_fp:.4f}`",
            "",
            "## Main Remaining Gaps",
            *[f"- {item}" for item in gaps],
            "",
        ],
    )

    generation_patch_report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "counts_by_family": generated["counts_by_family"],
        "split_files": generated["manifest"]["split_files"],
        "total_generated": int(generated["manifest"]["total_scenarios"]),
        "new_scenarios": generated["manifest"]["scenarios"],
        "event5_count_target": int(event5_count),
        "event7_count_target": int(event7_count),
        "mixed_count_target": int(mixed_count),
    }
    _write_json(output_root / "scenario_generation_patch_report.json", generation_patch_report)
    _write_md(
        output_root / "scenario_generation_patch_report.md",
        [
            "# Scenario Generation Patch Report",
            "",
            f"- total_generated: `{generation_patch_report['total_generated']}`",
            *[
                f"- {family}: `{count}`"
                for family, count in sorted(generation_patch_report["counts_by_family"].items())
            ],
            "",
        ],
    )

    return {
        "parameters": parameters,
        "generated": generated,
        "validation": validation,
        "before_holdout": before_report,
        "after_holdout": after_report,
        "transfer_report_path": str(transfer_json_path),
        "scenario_generation_patch_report_path": str(output_root / "scenario_generation_patch_report.json"),
    }

