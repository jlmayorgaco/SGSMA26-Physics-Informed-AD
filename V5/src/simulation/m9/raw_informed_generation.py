"""Targeted RAW-informed scenario generation for Event 5 and Event 7."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.simulation.m9.generator import generate_scenario
from src.simulation.m9.raw_informed_cyber_layer import (
    CyberIntervalSpec,
    RawInformedCyberLayer,
    apply_layer_to_scenario_dir,
)


@dataclass(slots=True)
class RawInformedScenarioSpec:
    scenario_id: str
    split: str
    family: str
    variant: str
    template_name: str
    event_coarse: int
    difficulty_level: str
    target_pmus: list[str]
    interval: CyberIntervalSpec
    template: dict[str, Any]


def _normal_template(name: str, duration_s: float = 12.0) -> dict[str, Any]:
    return {
        "name": name,
        "description": "RAW-informed cyber template with baseline physical system and external cyber process overlay.",
        "duration_s": float(duration_s),
        "physical_events": [],
        "cyber_events": [],
    }


def _mixed_template(name: str, *, variant: str, target_bus: str, duration_s: float = 12.0) -> dict[str, Any]:
    if variant == "event5_overlap":
        physical = {
            "event_id": "P001",
            "event_label": 3,
            "event_type": "generation_change",
            "subtype": "generation_drop",
            "start_time_s": 4.0,
            "end_time_s": 7.0,
            "duration_s": 3.0,
            "severity": 0.18,
            "shape": "ramp",
            "target_bus": target_bus,
            "target_line": None,
            "andes_params": {"p_change_fraction": -0.07},
        }
    else:
        physical = {
            "event_id": "P001",
            "event_label": 4,
            "event_type": "load_change",
            "subtype": "load_drop",
            "start_time_s": 4.0,
            "end_time_s": 7.2,
            "duration_s": 3.2,
            "severity": 0.16,
            "shape": "ramp",
            "target_bus": target_bus,
            "target_line": None,
            "andes_params": {"p_change_fraction": -0.06},
        }
    return {
        "name": name,
        "description": "RAW-informed concurrent physical + cyber overlay.",
        "duration_s": float(duration_s),
        "physical_events": [physical],
        "cyber_events": [],
    }


def _split_schedule(total: int, *, val_count: int, test_count: int) -> list[str]:
    train_count = total - val_count - test_count
    if train_count <= 0:
        raise ValueError("Invalid split counts: no training scenarios remain")
    return ["train"] * train_count + ["val"] * val_count + ["test"] * test_count


def build_raw_informed_specs(
    *,
    event5_count: int,
    event7_count: int,
    mixed_count: int,
    seed: int,
) -> list[RawInformedScenarioSpec]:
    rng = np.random.default_rng(seed)
    pmus = ["BUS39", "BUS29", "BUS10", "BUS22", "BUS19", "BUS2", "BUS5", "BUS6"]
    split_event5 = _split_schedule(event5_count, val_count=max(5, event5_count // 6), test_count=max(5, event5_count // 6))
    split_event7 = _split_schedule(event7_count, val_count=max(5, event7_count // 6), test_count=max(5, event7_count // 6))
    split_mixed = _split_schedule(mixed_count, val_count=max(6, mixed_count // 7), test_count=max(6, mixed_count // 7))

    specs: list[RawInformedScenarioSpec] = []
    scenario_counter = 1

    event5_variants = [
        "full_dropout_bursty",
        "partial_dropout",
        "bursty_missing",
        "long_full_dropout",
    ]
    for idx in range(event5_count):
        bus = pmus[idx % len(pmus)]
        variant = event5_variants[idx % len(event5_variants)]
        start = float(rng.uniform(3.5, 5.2))
        if variant == "long_full_dropout":
            duration = float(rng.uniform(2.2, 3.4))
            subtype = "full_dropout_long"
        elif variant == "partial_dropout":
            duration = float(rng.uniform(1.4, 2.5))
            subtype = "partial_dropout"
        elif variant == "bursty_missing":
            duration = float(rng.uniform(1.8, 3.0))
            subtype = "bursty_dropout"
        else:
            duration = float(rng.uniform(1.1, 2.0))
            subtype = "full_dropout_bursty"
        scenario_id = f"SIMM2{scenario_counter:04d}"
        scenario_counter += 1
        template_name = f"RAW_INFORMED_EVENT5_{variant.upper()}_{bus}"
        interval = CyberIntervalSpec(
            kind="event5",
            start_time_s=start,
            end_time_s=start + duration,
            target_pmus=[bus],
            subtype=subtype,
            event_label_if_no_physical=5,
            event_label_if_physical=6,
        )
        specs.append(
            RawInformedScenarioSpec(
                scenario_id=scenario_id,
                split=split_event5[idx],
                family="RAW_INFORMED_EVENT5",
                variant=variant,
                template_name=template_name,
                event_coarse=5,
                difficulty_level="hard" if variant != "full_dropout_bursty" else "medium",
                target_pmus=[bus],
                interval=interval,
                template=_normal_template(template_name),
            )
        )

    event7_variants = [
        "spike_corruption",
        "bias_drift",
        "stuck_after_missing",
        "replay_like",
    ]
    for idx in range(event7_count):
        bus = pmus[(idx * 2 + 1) % len(pmus)]
        variant = event7_variants[idx % len(event7_variants)]
        start = float(rng.uniform(3.8, 5.0))
        duration = float(rng.uniform(1.6, 3.0))
        scenario_id = f"SIMM2{scenario_counter:04d}"
        scenario_counter += 1
        template_name = f"RAW_INFORMED_EVENT7_{variant.upper()}_{bus}"
        interval = CyberIntervalSpec(
            kind="event7",
            start_time_s=start,
            end_time_s=start + duration,
            target_pmus=[bus],
            subtype=variant,
            event_label_if_no_physical=7,
            event_label_if_physical=8,
        )
        specs.append(
            RawInformedScenarioSpec(
                scenario_id=scenario_id,
                split=split_event7[idx],
                family="RAW_INFORMED_EVENT7",
                variant=variant,
                template_name=template_name,
                event_coarse=7,
                difficulty_level="hard",
                target_pmus=[bus],
                interval=interval,
                template=_normal_template(template_name),
            )
        )

    mixed_variants = ["event5_overlap", "event7_overlap"]
    for idx in range(mixed_count):
        variant = mixed_variants[idx % len(mixed_variants)]
        bus = pmus[(idx * 3 + 2) % len(pmus)]
        target_bus = ["BUS2", "BUS7", "BUS19", "BUS39"][idx % 4]
        start = float(rng.uniform(4.0, 5.0))
        duration = float(rng.uniform(1.7, 2.8))
        scenario_id = f"SIMM2{scenario_counter:04d}"
        scenario_counter += 1
        template_name = f"RAW_INFORMED_MIXED_CYBER_{variant.upper()}_{target_bus}_{bus}"
        if variant == "event5_overlap":
            interval = CyberIntervalSpec(
                kind="event5",
                start_time_s=start,
                end_time_s=start + duration,
                target_pmus=[bus],
                subtype="event5_overlap",
                event_label_if_no_physical=5,
                event_label_if_physical=6,
            )
            event_coarse = 6
        else:
            interval = CyberIntervalSpec(
                kind="event7",
                start_time_s=start,
                end_time_s=start + duration,
                target_pmus=[bus],
                subtype="event7_overlap",
                event_label_if_no_physical=7,
                event_label_if_physical=8,
            )
            event_coarse = 8
        specs.append(
            RawInformedScenarioSpec(
                scenario_id=scenario_id,
                split=split_mixed[idx],
                family="RAW_INFORMED_MIXED_CYBER",
                variant=variant,
                template_name=template_name,
                event_coarse=event_coarse,
                difficulty_level="hard",
                target_pmus=[bus],
                interval=interval,
                template=_mixed_template(template_name, variant=variant, target_bus=target_bus),
            )
        )
    return specs


def _scenario_dir(workspace_root: Path, scenario_id: str) -> Path:
    return workspace_root / "data" / "scenarios" / scenario_id


def _relative_scenario_path(workspace_root: Path, scenario_id: str) -> str:
    path = _scenario_dir(workspace_root, scenario_id)
    return str(path.relative_to(workspace_root))


def generate_raw_informed_dataset(
    *,
    workspace_root: Path,
    output_root: Path,
    raw_path: Path,
    pmu_location_path: Path,
    reference_pmu_dir: Path,
    layer: RawInformedCyberLayer,
    event5_count: int,
    event7_count: int,
    mixed_count: int,
    seed: int,
) -> dict[str, Any]:
    specs = build_raw_informed_specs(
        event5_count=event5_count,
        event7_count=event7_count,
        mixed_count=mixed_count,
        seed=seed,
    )
    scenarios_root = workspace_root / "data" / "scenarios"
    scenarios_root.mkdir(parents=True, exist_ok=True)
    metadata_dir = output_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    latent_dir = output_root / "metadata" / "latent_states"
    latent_dir.mkdir(parents=True, exist_ok=True)

    split_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    counts_by_family: dict[str, int] = {}
    for index, spec in enumerate(specs):
        scenario_seed = int(seed + index)
        generate_scenario(
            raw_path=raw_path,
            pmu_location_path=pmu_location_path,
            reference_pmu_dir=reference_pmu_dir,
            output_root=scenarios_root,
            scenario_template=spec.template,
            scenario_id=spec.scenario_id,
            seed=scenario_seed,
            use_andes=False,
            save_plots=False,
        )
        scenario_dir = _scenario_dir(workspace_root, spec.scenario_id)
        latent_path = latent_dir / f"{spec.scenario_id}_cyber_latent_trace.csv"
        latent = apply_layer_to_scenario_dir(
            scenario_dir=scenario_dir,
            layer=layer,
            intervals=[spec.interval],
            latent_output_path=latent_path,
        )
        manifest_path = scenario_dir / "scenario_manifest.json"
        if manifest_path.exists():
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            payload = {"scenario_id": spec.scenario_id}
        payload["scenario_template"] = spec.template_name
        payload["targeted_family"] = spec.family
        payload["targeted_variant"] = spec.variant
        payload["event_coarse"] = int(spec.event_coarse)
        payload["split"] = spec.split
        payload["raw_informed_cyber"] = {
            "enabled": True,
            "family": spec.family,
            "variant": spec.variant,
            "target_pmus": spec.target_pmus,
            "interval": asdict(spec.interval),
            "latent_trace_path": str(latent_path.resolve()),
            "latent_rows": int(len(latent)),
        }
        manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        row = {
            "scenario_id": spec.scenario_id,
            "scenario_dir": _relative_scenario_path(workspace_root, spec.scenario_id),
            "template_name": spec.template_name,
            "event_coarse": int(spec.event_coarse),
            "difficulty_level": spec.difficulty_level,
            "scenario_family": f"{spec.family}::{spec.variant}",
            "seed_family": f"{spec.family}::{','.join(spec.target_pmus)}::{spec.variant}",
            "targeted_family": spec.family,
            "targeted_variant": spec.variant,
            "split": spec.split,
        }
        split_rows.append(row)
        manifest_rows.append(
            {
                **row,
                "scenario_seed": scenario_seed,
                "target_pmus": ";".join(spec.target_pmus),
                "interval_start_s": float(spec.interval.start_time_s),
                "interval_end_s": float(spec.interval.end_time_s),
                "cyber_kind": spec.interval.kind,
            }
        )
        counts_by_family[spec.family] = counts_by_family.get(spec.family, 0) + 1

    split_df = pd.DataFrame(split_rows)
    train_df = split_df.loc[split_df["split"] == "train"].drop(columns=["split"]).reset_index(drop=True)
    val_df = split_df.loc[split_df["split"] == "val"].drop(columns=["split"]).reset_index(drop=True)
    test_df = split_df.loc[split_df["split"] == "test"].drop(columns=["split"]).reset_index(drop=True)
    train_csv = metadata_dir / "m2_train_scenarios.csv"
    val_csv = metadata_dir / "m2_val_scenarios.csv"
    test_csv = metadata_dir / "m2_test_scenarios.csv"
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    generated_manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "counts_by_family": counts_by_family,
        "total_scenarios": int(len(split_df)),
        "split_counts": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "scenarios": manifest_rows,
        "split_files": {
            "train": str(train_csv),
            "val": str(val_csv),
            "test": str(test_csv),
        },
    }
    generated_manifest_path = output_root / "generated_scenarios_manifest.json"
    generated_manifest_path.write_text(json.dumps(generated_manifest, indent=2), encoding="utf-8")
    return {
        "specs": specs,
        "counts_by_family": counts_by_family,
        "train_csv": train_csv,
        "val_csv": val_csv,
        "test_csv": test_csv,
        "manifest_path": generated_manifest_path,
        "manifest": generated_manifest,
    }

