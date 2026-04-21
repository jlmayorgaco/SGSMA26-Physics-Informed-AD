"""M9 scenario generation orchestration."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.domain.topology import canonical_bus_name
from src.simulation.m9.calibration import compare_raw_vs_sim, extract_reference_statistics, write_json, read_json
from src.simulation.m9.constants import DEFAULT_MVA_BASE, DEFAULT_RATED_FREQUENCY_HZ, PMU_BUSES_OFFICIAL
from src.simulation.m9.cyber import apply_cyber_events, build_cyber_frame_metadata, combine_event_label
from src.simulation.m9.exports import (
    build_all_bus_measurements,
    build_full_state_target,
    build_observed_pmu_frames,
    ensure_scenario_dirs,
    write_labels,
    write_metadata,
    write_parquet_if_available,
    write_plots,
    write_pmu_csvs,
    write_summary_markdown,
)
from src.simulation.m9.physical import generate_physical_truth, truth_to_dataframe
from src.simulation.m9.templates import get_template, list_templates


def _scenario_root(output_root: str | Path, scenario_id: str) -> Path:
    root = Path(output_root)
    if root.name.upper().startswith("SIM") or root.name.upper() == scenario_id.upper():
        return root
    return root / scenario_id


def _write_configs(paths: dict[str, Path], template: dict[str, Any], noise_model: dict[str, Any], physical_metadata: dict[str, Any]) -> dict[str, str]:
    physical_path = paths["config"] / "physical_config.json"
    cyber_path = paths["config"] / "cyber_config.json"
    noise_path = paths["config"] / "noise_config.json"
    write_json(physical_path, {"physical_events": template.get("physical_events", []), "physical_metadata": physical_metadata})
    write_json(cyber_path, {"cyber_events": template.get("cyber_events", [])})
    write_json(noise_path, noise_model)
    return {"physical_config": str(physical_path), "cyber_config": str(cyber_path), "noise_config": str(noise_path)}


def _update_registry(scenarios_root: Path, scenario_id: str, manifest_path: Path, template_name: str) -> None:
    registry_path = scenarios_root / "scenario_registry.json"
    if registry_path.exists():
        try:
            registry = read_json(registry_path)
        except Exception:
            registry = {"scenarios": []}
    else:
        registry = {"scenarios": []}
    scenarios = [s for s in registry.get("scenarios", []) if s.get("scenario_id") != scenario_id]
    scenarios.append({"scenario_id": scenario_id, "scenario_template": template_name, "manifest_path": str(manifest_path), "updated_at_utc": datetime.now(timezone.utc).isoformat()})
    registry["scenarios"] = sorted(scenarios, key=lambda x: x.get("scenario_id", ""))
    write_json(registry_path, registry)


def _combined_truth_df(truth, pmu_buses: set[str], cyber_events: list[dict[str, Any]]) -> pd.DataFrame:
    cyber_labels, cyber_types, _, _ = build_cyber_frame_metadata(truth.timestamps, cyber_events)
    df = truth_to_dataframe(truth, pmu_buses, cyber_type_by_frame=cyber_types)
    combined = [combine_event_label(int(p), int(c), str(ct)) for p, c, ct in zip(truth.event_by_frame, cyber_labels, cyber_types)]
    by_ts = dict(zip([float(x) for x in truth.timestamps], combined))
    df["EVENT"] = df["TIMESTAMP"].map(by_ts).astype(int)
    return df


def generate_scenario(
    raw_path: str | Path = "data/metadata/IEEE_39_Bus_Power_System.raw",
    pmu_location_path: str | Path = "data/metadata/PMUbus_ Location.txt",
    reference_pmu_dir: str | Path = "data/RAW0001",
    output_root: str | Path = "data/scenarios",
    scenario_template: str | dict[str, Any] = "TEMPLATE_OFFICIAL_STYLE_MULTI_EVENT",
    scenario_id: str = "SIM0001",
    seed: int = 12345,
    fps: float = 30.0,
    duration_s: float | None = None,
    use_andes: bool = False,
    noise_profile: str = "raw0001_empirical",
    cyber_profile: str = "template",
    physical_profile: str = "template",
    reference_stats: dict[str, Any] | None = None,
    save_json: bool = True,
    save_csv: bool = True,
    save_plots: bool = True,
) -> dict[str, Any]:
    template = get_template(scenario_template) if isinstance(scenario_template, str) else dict(scenario_template)
    scenario_dir = _scenario_root(output_root, scenario_id)
    paths = ensure_scenario_dirs(scenario_dir)
    pmu_buses = [canonical_bus_name(b) for b in PMU_BUSES_OFFICIAL]
    pmu_bus_set = set(pmu_buses)

    if reference_stats is None:
        reference_stats = extract_reference_statistics(reference_pmu_dir)

    truth = generate_physical_truth(
        raw_path=raw_path,
        pmu_location_path=pmu_location_path,
        template=template,
        seed=seed,
        fps=fps,
        duration_s=duration_s,
        use_andes=use_andes,
    )
    pmu_frames = build_observed_pmu_frames(truth, raw_path, reference_stats, pmu_buses=pmu_buses, seed=seed)
    pmu_frames = apply_cyber_events(pmu_frames, template.get("cyber_events", []), truth.event_by_frame, seed=seed)

    output_files: dict[str, Any] = {}
    if save_csv:
        output_files["pmu_csvs"] = write_pmu_csvs(pmu_frames, paths["pmu"])

    truth_df = _combined_truth_df(truth, pmu_bus_set, template.get("cyber_events", []))
    all_meas = build_all_bus_measurements(truth, raw_path, pmu_bus_set)
    # Keep all-bus measurement labels aligned with the coarse official event semantics.
    event_by_ts = truth_df.drop_duplicates(subset=["TIMESTAMP"])[["TIMESTAMP", "EVENT"]].set_index("TIMESTAMP")["EVENT"]
    all_meas["EVENT"] = all_meas["TIMESTAMP"].map(event_by_ts).fillna(0).astype(int)
    target = build_full_state_target(truth, pmu_bus_set)
    # Keep full-state target labels aligned with the official coarse event label used in all_bus_truth.
    target["EVENT"] = truth_df["EVENT"].to_numpy(dtype=int)
    truth_csv = paths["all_buses"] / "all_bus_truth.csv"
    meas_csv = paths["all_buses"] / "all_bus_measurements.csv"
    target_csv = paths["all_buses"] / "full_state_target.csv"
    truth_df.to_csv(truth_csv, index=False)
    all_meas.to_csv(meas_csv, index=False)
    target.to_csv(target_csv, index=False)
    output_files["all_bus_truth_csv"] = str(truth_csv)
    output_files["all_bus_measurements_csv"] = str(meas_csv)
    output_files["full_state_target_csv"] = str(target_csv)
    if write_parquet_if_available(truth_df, paths["all_buses"] / "all_bus_truth.parquet"):
        output_files["all_bus_truth_parquet"] = str(paths["all_buses"] / "all_bus_truth.parquet")
    if write_parquet_if_available(target, paths["all_buses"] / "full_state_target.parquet"):
        output_files["full_state_target_parquet"] = str(paths["all_buses"] / "full_state_target.parquet")

    labels_files = write_labels(paths, truth, template.get("physical_events", []), template.get("cyber_events", []))
    output_files["labels"] = labels_files

    raw_vs_sim = compare_raw_vs_sim(reference_stats, paths["pmu"])
    metadata_files = write_metadata(paths, pmu_buses, truth, reference_stats, raw_vs_sim=raw_vs_sim)
    output_files["metadata"] = metadata_files

    noise_model = {
        "profile": noise_profile,
        "reference": str(reference_pmu_dir),
        "calibrated_from_raw0001": not bool(reference_stats.get("fallback_used", False)),
        "seed": seed,
        "per_channel_empirical_mean_std": True,
    }
    config_files = _write_configs(paths, template, noise_model, truth.metadata)
    output_files["config"] = config_files

    concurrent_events = []
    for phys in template.get("physical_events", []):
        for cyber in template.get("cyber_events", []):
            if float(phys.get("start_time_s", 0.0)) <= float(cyber.get("end_time_s", 0.0)) and float(cyber.get("start_time_s", 0.0)) <= float(phys.get("end_time_s", 0.0)):
                concurrent_events.append({"physical_event_id": phys.get("event_id"), "cyber_event_id": cyber.get("event_id"), "event_label": 6 if int(cyber.get("event_label", 0)) in {5, 6} else 8})

    manifest = {
        "scenario_id": scenario_id,
        "scenario_template": template["name"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "base_system": "IEEE39",
        "mva_base": DEFAULT_MVA_BASE,
        "rated_frequency_hz": DEFAULT_RATED_FREQUENCY_HZ,
        "fps": float(fps),
        "duration_s": float(duration_s if duration_s is not None else template.get("duration_s", 0.0)),
        "pmu_buses": pmu_buses,
        "physical_events": template.get("physical_events", []),
        "cyber_events": template.get("cyber_events", []),
        "concurrent_events": concurrent_events,
        "noise_model": noise_model,
        "calibration_reference": {"reference_pmu_dir": str(reference_pmu_dir), "reference_source": reference_stats.get("source", "RAW0001"), "fallback_used": reference_stats.get("fallback_used", False)},
        "profiles": {"cyber_profile": cyber_profile, "physical_profile": physical_profile},
        "physical_truth_metadata": truth.metadata,
        "raw_vs_sim_summary": raw_vs_sim.get("distance_summary", {}),
        "output_files": output_files,
    }
    manifest_path = paths["root"] / "scenario_manifest.json"
    write_json(manifest_path, manifest)
    output_files["scenario_manifest"] = str(manifest_path)

    plot_files = write_plots(paths, truth, pmu_frames, raw_vs_sim=raw_vs_sim) if save_plots else {}
    output_files["plots"] = plot_files
    manifest["output_files"] = output_files
    write_json(manifest_path, manifest)
    write_summary_markdown(paths["root"] / "scenario_summary.md", manifest)
    _update_registry(Path(output_root), scenario_id, manifest_path, template["name"])
    return manifest


def generate_template_matrix(
    raw_path: str | Path = "data/metadata/IEEE_39_Bus_Power_System.raw",
    pmu_location_path: str | Path = "data/metadata/PMUbus_ Location.txt",
    reference_pmu_dir: str | Path = "data/RAW0001",
    output_root: str | Path = "data/scenarios",
    seed: int = 12345,
    fps: float = 30.0,
    duration_s: float | None = None,
    use_andes: bool = False,
    save_plots: bool = True,
) -> list[dict[str, Any]]:
    reference_stats = extract_reference_statistics(reference_pmu_dir)
    template_order = [
        "TEMPLATE_OFFICIAL_STYLE_MULTI_EVENT",
        "TEMPLATE_EVENT0_NORMAL",
        "TEMPLATE_EVENT1_FAULT",
        "TEMPLATE_EVENT2_LINE_OUTAGE",
        "TEMPLATE_EVENT3_GENERATION_CHANGE",
        "TEMPLATE_EVENT4_LOAD_CHANGE",
        "TEMPLATE_EVENT5_MISSING_ONLY",
        "TEMPLATE_EVENT6_CONCURRENT_MISSING_PLUS_PHYSICAL",
        "TEMPLATE_EVENT7_BAD_DATA",
        "TEMPLATE_EVENT8_UNKNOWN_COMPOSITE",
    ]
    manifests: list[dict[str, Any]] = []
    for idx, template_name in enumerate(template_order, start=1):
        scenario_id = "SIM0001" if idx == 1 else f"SIM{idx:04d}"
        manifests.append(
            generate_scenario(
                raw_path=raw_path,
                pmu_location_path=pmu_location_path,
                reference_pmu_dir=reference_pmu_dir,
                output_root=output_root,
                scenario_template=template_name,
                scenario_id=scenario_id,
                seed=seed + idx,
                fps=fps,
                duration_s=duration_s,
                use_andes=use_andes and idx == 1,
                reference_stats=reference_stats,
                save_plots=save_plots,
            )
        )
    return manifests
