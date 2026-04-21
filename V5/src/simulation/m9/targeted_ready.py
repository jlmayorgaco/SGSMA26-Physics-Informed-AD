from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd

from src.simulation.m9.constants import PMU_BUSES_OFFICIAL, PMU_MEASUREMENT_SUFFIXES
from src.simulation.m9.generator import generate_scenario


@dataclass(slots=True)
class TargetedScenarioSpec:
    scenario_id: str
    split: str
    targeted_family: str
    targeted_variant: str
    template_name: str
    event_coarse: int
    difficulty_level: str
    reason: str
    scenario_group: str
    target_bus: str = ""
    target_channels: tuple[str, ...] = ()
    template: dict[str, Any] | None = None


def _physical_event(
    *,
    event_id: str,
    label: int,
    event_type: str,
    subtype: str,
    start: float,
    duration: float,
    severity: float,
    target_bus: str | None = None,
    target_line: str | None = None,
    shape: str = "step",
    andes_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "event_label": int(label),
        "event_type": event_type,
        "subtype": subtype,
        "start_time_s": float(start),
        "end_time_s": float(start + duration),
        "duration_s": float(duration),
        "severity": float(severity),
        "shape": shape,
        "target_bus": target_bus,
        "target_line": target_line,
        "andes_params": andes_params or {},
    }


def _cyber_event(
    *,
    event_id: str,
    label: int,
    event_type: str,
    subtype: str,
    start: float,
    duration: float,
    target_pmus: list[str],
    target_channels: list[str] | None = None,
    **params: Any,
) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "event_label": int(label),
        "event_type": event_type,
        "subtype": subtype,
        "start_time_s": float(start),
        "end_time_s": float(start + duration),
        "duration_s": float(duration),
        "target_pmus": list(target_pmus),
        "target_channels": list(target_channels or ["ALL"]),
        "params": params,
    }


def _normal_template(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "description": "Targeted normal hard negative with benign near-threshold PMU behavior.",
        "duration_s": 12.0,
        "physical_events": [],
        "cyber_events": [],
    }


def _missing_template(name: str, *, bus: str, subtype: str, start: float, duration: float, channels: list[str] | None = None, **params: Any) -> dict[str, Any]:
    return {
        "name": name,
        "description": "Targeted cyber-heavy missing-data scenario.",
        "duration_s": 12.0,
        "physical_events": [],
        "cyber_events": [
            _cyber_event(
                event_id="C001",
                label=5,
                event_type="missing_data",
                subtype=subtype,
                start=start,
                duration=duration,
                target_pmus=[bus],
                target_channels=channels,
                **params,
            )
        ],
    }


def _bad_data_template(
    name: str,
    *,
    bus: str,
    subtype: str,
    start: float,
    duration: float,
    channels: list[str],
    extra_event: dict[str, Any] | None = None,
    **params: Any,
) -> dict[str, Any]:
    cyber_events = [
        _cyber_event(
            event_id="C001",
            label=7,
            event_type="bad_data",
            subtype=subtype,
            start=start,
            duration=duration,
            target_pmus=[bus],
            target_channels=channels,
            **params,
        )
    ]
    if extra_event is not None:
        cyber_events.append(extra_event)
    return {
        "name": name,
        "description": "Targeted cyber-heavy bad-data scenario.",
        "duration_s": 12.0,
        "physical_events": [],
        "cyber_events": cyber_events,
    }


def _concurrent_template(
    name: str,
    *,
    physical_event: dict[str, Any],
    cyber_events: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "name": name,
        "description": "Targeted concurrent scenario with subtle physical dynamics plus cyber overlap.",
        "duration_s": 12.0,
        "physical_events": [physical_event],
        "cyber_events": cyber_events,
    }


def _scenario_dir(workspace_root: Path, scenario_id: str) -> Path:
    return workspace_root / "data" / "scenarios" / scenario_id


def _measurement_cols_for_bus(df: pd.DataFrame, bus: str) -> list[str]:
    return [f"{bus}_{suffix}" for suffix in PMU_MEASUREMENT_SUFFIXES if f"{bus}_{suffix}" in df.columns]


def _apply_normal_hard_negative_profile(scenario_dir: Path, *, variant: str, target_bus: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    token = int("".join(ch for ch in target_bus if ch.isdigit()))
    path = scenario_dir / "pmu" / f"Bus{token}_Competition_Data_sim.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    ts = pd.to_numeric(df["TIMESTAMP"], errors="coerce").to_numpy(dtype=float)
    cols = _measurement_cols_for_bus(df, target_bus)
    if not cols:
        return
    center = 6.0 + float(rng.uniform(-0.8, 0.8))
    width = 1.8 + float(rng.uniform(0.2, 1.0))
    local_mask = np.abs(ts - center) <= width
    envelope = np.clip(1.0 - np.abs(ts - center) / max(width, 1e-6), 0.0, 1.0)
    for col in cols:
        x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(x)
        if not finite.any():
            continue
        scale = max(float(np.nanstd(x[finite])), 1e-6)
        perturb = np.zeros_like(x, dtype=float)
        if col.endswith("_Freq"):
            freq_amp = 0.0018 if variant == "freq_wobble" else 0.0012
            perturb += envelope * np.sin(2.0 * np.pi * (ts - center) / 1.6) * freq_amp
        elif col.endswith("_ROCOF"):
            rocof_amp = 0.010 if variant == "freq_wobble" else 0.006
            perturb += envelope * np.cos(2.0 * np.pi * (ts - center) / 1.4) * rocof_amp
        elif col.endswith("ANG"):
            ang_amp = 1.2 if variant == "angle_drift" else 0.6
            perturb += envelope * np.sin(2.0 * np.pi * (ts - center) / 2.2) * ang_amp
            if variant == "angle_drift":
                perturb += np.linspace(0.0, 0.9, len(ts))
        else:
            amp_scale = 0.020 if variant == "ringdown" else 0.012
            perturb += envelope * np.exp(-np.maximum(ts - center, 0.0)) * np.sin(2.0 * np.pi * (ts - center) / 1.9) * scale * amp_scale
            if variant == "gentle_drift":
                perturb += np.linspace(0.0, scale * 0.010, len(ts))
        jitter = rng.normal(0.0, scale * 0.0025, size=len(ts))
        x[local_mask] = x[local_mask] + perturb[local_mask] + jitter[local_mask]
        df[col] = x
    df["DATA_PRESENT"] = 1
    if "Event" in df.columns:
        df["Event"] = 0
    df.to_csv(path, index=False)


def _split_counts(total: int, *, val_count: int, test_count: int) -> tuple[int, int, int]:
    train_count = total - val_count - test_count
    if train_count < 1:
        raise ValueError("targeted split counts leave no train scenarios")
    return train_count, val_count, test_count


def _build_targeted_specs() -> list[TargetedScenarioSpec]:
    specs: list[TargetedScenarioSpec] = []
    scenario_index = 1
    buses = ["BUS29", "BUS10", "BUS19", "BUS22", "BUS39", "BUS2", "BUS5", "BUS6"]

    family_plan = [
        ("NORMAL_HARD_NEGATIVES", 60, 9, 9),
        ("CYBER_HEAVY_MISSING", 30, 5, 5),
        ("CYBER_HEAVY_BAD_DATA", 30, 5, 5),
        ("CONCURRENT_EXTRA", 40, 6, 6),
    ]

    split_order: dict[str, list[str]] = {}
    for family, total, val_count, test_count in family_plan:
        train_count, _, _ = _split_counts(total, val_count=val_count, test_count=test_count)
        split_order[family] = ["train"] * train_count + ["val"] * val_count + ["test"] * test_count

    normal_variants = ["ringdown", "gentle_drift", "angle_drift", "freq_wobble"]
    for i in range(60):
        scenario_id = f"SIMR{scenario_index:04d}"
        scenario_index += 1
        variant = normal_variants[i % len(normal_variants)]
        bus = buses[i % len(buses)]
        split = split_order["NORMAL_HARD_NEGATIVES"][i]
        template_name = f"TARGET_EVENT0_NORMAL_HARD_NEGATIVE_{variant.upper()}_{bus}"
        specs.append(
            TargetedScenarioSpec(
                scenario_id=scenario_id,
                split=split,
                targeted_family="NORMAL_HARD_NEGATIVES",
                targeted_variant=variant,
                template_name=template_name,
                event_coarse=0,
                difficulty_level="hard" if variant != "ringdown" else "medium",
                reason="Benign drift/oscillation near detector thresholds should remain normal.",
                scenario_group=f"NORMAL::{variant}::{bus}",
                target_bus=bus,
                template=_normal_template(template_name),
            )
        )

    missing_variants = [
        ("full_dropout", None, 4.0, 0.9, {}),
        ("burst_dropout", None, 4.2, 1.8, {}),
        ("periodic_dropout", ["VA_MAG", "VB_MAG", "VC_MAG"], 4.0, 2.4, {"period_frames": 5, "width_frames": 2}),
        ("full_dropout", ["IA_MAG", "IB_MAG"], 5.0, 2.2, {}),
        ("periodic_dropout", None, 3.8, 3.0, {"period_frames": 7, "width_frames": 3}),
    ]
    for i in range(30):
        scenario_id = f"SIMR{scenario_index:04d}"
        scenario_index += 1
        variant, channels, start, duration, params = missing_variants[i % len(missing_variants)]
        bus = buses[(i * 2) % len(buses)]
        split = split_order["CYBER_HEAVY_MISSING"][i]
        template_name = f"TARGET_EVENT5_MISSING_{variant.upper()}_{bus}"
        specs.append(
            TargetedScenarioSpec(
                scenario_id=scenario_id,
                split=split,
                targeted_family="CYBER_HEAVY_MISSING",
                targeted_variant=variant,
                template_name=template_name,
                event_coarse=5,
                difficulty_level="hard" if variant != "full_dropout" else "medium",
                reason="Expose missing-only, bursty, and partial-dropout cyber behavior across different PMUs/channels.",
                scenario_group=f"MISSING::{variant}::{bus}::{','.join(channels or ['ALL'])}",
                target_bus=bus,
                target_channels=tuple(channels or ("ALL",)),
                template=_missing_template(template_name, bus=bus, subtype=variant, start=start, duration=duration, channels=channels, **params),
            )
        )

    bad_variants = [
        ("spike", ["VA_MAG", "IA_MAG"], {"amplitude": 5.0}),
        ("bias", ["VA_ANG"], {"bias": 2.8}),
        ("drift", ["VB_MAG", "VC_MAG"], {"drift": 0.05}),
        ("replay_window", ["IA_MAG", "IB_MAG"], {"lag_frames": 6}),
        ("fixed_delay", ["VA_MAG", "VB_MAG"], {"delay_frames": 3}),
        ("timestamp_jitter", ["ALL"], {"std_s": 0.005}),
    ]
    for i in range(30):
        scenario_id = f"SIMR{scenario_index:04d}"
        scenario_index += 1
        variant, channels, params = bad_variants[i % len(bad_variants)]
        bus = buses[(i * 3 + 1) % len(buses)]
        split = split_order["CYBER_HEAVY_BAD_DATA"][i]
        template_name = f"TARGET_EVENT7_BAD_DATA_{variant.upper()}_{bus}"
        extra = None
        if variant == "replay_window":
            extra = _cyber_event(
                event_id="C002",
                label=7,
                event_type="bad_data",
                subtype="stuck_at_last_value",
                start=5.0,
                duration=1.4,
                target_pmus=[bus],
                target_channels=channels,
            )
        specs.append(
            TargetedScenarioSpec(
                scenario_id=scenario_id,
                split=split,
                targeted_family="CYBER_HEAVY_BAD_DATA",
                targeted_variant=variant,
                template_name=template_name,
                event_coarse=7,
                difficulty_level="hard",
                reason="Strengthen bad-data sensitivity on subtle spikes, bias/drift, replay, stuck, and timing corruption.",
                scenario_group=f"BAD::{variant}::{bus}::{','.join(channels)}",
                target_bus=bus,
                target_channels=tuple(channels),
                template=_bad_data_template(
                    template_name,
                    bus=bus,
                    subtype=variant,
                    start=4.0,
                    duration=1.8 if variant != "bias" else 2.6,
                    channels=channels,
                    extra_event=extra,
                    **params,
                ),
            )
        )

    concurrent_variants = [
        ("missing_overlap", 6),
        ("subtle_fault_plus_missing", 6),
        ("load_plus_bad_data", 8),
        ("generation_plus_replay", 8),
    ]
    concurrent_buses = ["BUS2", "BUS7", "BUS19", "BUS39"]
    for i in range(40):
        scenario_id = f"SIMR{scenario_index:04d}"
        scenario_index += 1
        variant, label = concurrent_variants[i % len(concurrent_variants)]
        split = split_order["CONCURRENT_EXTRA"][i]
        physical_bus = concurrent_buses[i % len(concurrent_buses)]
        cyber_bus = buses[(i * 2 + 3) % len(buses)]
        if variant == "missing_overlap":
            physical = _physical_event(
                event_id="P001",
                label=6,
                event_type="generation_change",
                subtype="generation_drop",
                start=4.0,
                duration=2.4,
                severity=0.22,
                target_bus=physical_bus,
                shape="ramp",
                andes_params={"p_change_fraction": -0.08},
            )
            cyber_events = [
                _cyber_event(
                    event_id="C001",
                    label=6,
                    event_type="missing_data",
                    subtype="burst_dropout",
                    start=4.2,
                    duration=1.8,
                    target_pmus=[cyber_bus],
                )
            ]
        elif variant == "subtle_fault_plus_missing":
            physical = _physical_event(
                event_id="P001",
                label=6,
                event_type="fault",
                subtype="3LG",
                start=4.0,
                duration=0.16,
                severity=0.20,
                target_bus=physical_bus,
                andes_params={"fault_type": "BusFault", "rf": 0.02, "xf": 0.02},
            )
            cyber_events = [
                _cyber_event(
                    event_id="C001",
                    label=6,
                    event_type="missing_data",
                    subtype="periodic_dropout",
                    start=4.1,
                    duration=1.6,
                    target_pmus=[cyber_bus],
                    period_frames=6,
                    width_frames=2,
                )
            ]
        elif variant == "load_plus_bad_data":
            physical = _physical_event(
                event_id="P001",
                label=8,
                event_type="load_change",
                subtype="load_drop",
                start=4.0,
                duration=2.8,
                severity=0.18,
                target_bus=physical_bus,
                shape="ramp",
                andes_params={"p_change_fraction": -0.07},
            )
            cyber_events = [
                _cyber_event(
                    event_id="C001",
                    label=8,
                    event_type="bad_data",
                    subtype="bias",
                    start=4.15,
                    duration=2.0,
                    target_pmus=[cyber_bus],
                    target_channels=["VA_ANG"],
                    bias=2.5,
                )
            ]
        else:
            physical = _physical_event(
                event_id="P001",
                label=8,
                event_type="generation_change",
                subtype="generation_drop",
                start=4.0,
                duration=2.2,
                severity=0.20,
                target_bus=physical_bus,
                shape="ramp",
                andes_params={"p_change_fraction": -0.09},
            )
            cyber_events = [
                _cyber_event(
                    event_id="C001",
                    label=8,
                    event_type="bad_data",
                    subtype="replay_window",
                    start=4.2,
                    duration=1.8,
                    target_pmus=[cyber_bus],
                    target_channels=["IA_MAG", "IB_MAG"],
                    lag_frames=5,
                )
            ]
        template_name = f"TARGET_EVENT{label}_{variant.upper()}_{physical_bus}_{cyber_bus}"
        specs.append(
            TargetedScenarioSpec(
                scenario_id=scenario_id,
                split=split,
                targeted_family="CONCURRENT_EXTRA",
                targeted_variant=variant,
                template_name=template_name,
                event_coarse=label,
                difficulty_level="hard",
                reason="Protect concurrent detection when weak physical evidence overlaps missing or bad-data corruption.",
                scenario_group=f"CONCURRENT::{variant}::{physical_bus}::{cyber_bus}",
                target_bus=physical_bus,
                template=_concurrent_template(template_name, physical_event=physical, cyber_events=cyber_events),
            )
        )
    return specs


def generate_targeted_ready_dataset(
    *,
    workspace_root: Path,
    raw_path: Path,
    pmu_location_path: Path,
    reference_pmu_dir: Path,
    seed: int = 12345,
) -> dict[str, Any]:
    scenarios_root = workspace_root / "data" / "scenarios"
    scenarios_root.mkdir(parents=True, exist_ok=True)
    specs = _build_targeted_specs()
    records: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    counts_by_family: dict[str, int] = {}

    for index, spec in enumerate(specs):
        scenario_seed = seed + index
        generate_scenario(
            raw_path=raw_path,
            pmu_location_path=pmu_location_path,
            reference_pmu_dir=reference_pmu_dir,
            output_root=scenarios_root,
            scenario_template=spec.template or spec.template_name,
            scenario_id=spec.scenario_id,
            seed=scenario_seed,
            use_andes=False,
            save_plots=False,
        )
        scenario_dir = _scenario_dir(workspace_root, spec.scenario_id)
        if spec.targeted_family == "NORMAL_HARD_NEGATIVES":
            _apply_normal_hard_negative_profile(
                scenario_dir,
                variant=spec.targeted_variant,
                target_bus=spec.target_bus or "BUS29",
                seed=scenario_seed,
            )
        manifest_path = scenario_dir / "scenario_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["targeted_generation"] = {
                "family": spec.targeted_family,
                "variant": spec.targeted_variant,
                "reason": spec.reason,
                "split_assignment": spec.split,
                "scenario_group": spec.scenario_group,
            }
            manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

        record = {
            "scenario_id": spec.scenario_id,
            "scenario_dir": str(Path("data") / "scenarios" / spec.scenario_id),
            "template_name": spec.template_name,
            "event_coarse": int(spec.event_coarse),
            "difficulty_level": spec.difficulty_level,
            "scenario_family": f"{spec.targeted_family}::{spec.targeted_variant}",
            "seed_family": spec.scenario_group,
            "split": spec.split,
            "targeted_family": spec.targeted_family,
            "targeted_variant": spec.targeted_variant,
            "generation_reason": spec.reason,
        }
        records.append(record)
        scenario_rows.append(
            {
                **record,
                "target_bus": spec.target_bus,
                "target_channels": ",".join(spec.target_channels),
                "scenario_group": spec.scenario_group,
            }
        )
        counts_by_family[spec.targeted_family] = counts_by_family.get(spec.targeted_family, 0) + 1

    frame = pd.DataFrame(records)
    return {
        "records": frame,
        "scenarios": pd.DataFrame(scenario_rows),
        "counts_by_family": counts_by_family,
        "specs": [asdict(item) for item in specs],
    }
