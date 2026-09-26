from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

try:
    import _bootstrap  # type: ignore  # noqa: F401
except ModuleNotFoundError:
    from pipelines import _bootstrap  # type: ignore  # noqa: F401

import numpy as np
import pandas as pd

from src.classes import PipelineResult
from src.features.bus_agnostic import (
    aggregate_pmu_features,
    graph_feature_block,
    load_zbus_distances,
    pmu_severity_vector,
    pmu_window_features,
)
from src.helpers.paths import DEFAULT_TOPOLOGY_DIR, WORKBENCH_DIR
from src.utils.io import json_safe, write_json


DEFAULT_SIM_DIR = WORKBENCH_DIR / "simulated" / "sgsma_generated"
DEFAULT_SCENARIO_DIR = WORKBENCH_DIR / "scenarios" / "sgsma_generated"
DEFAULT_OUT = WORKBENCH_DIR / "agnostic_training_set"
ALL_BUSES = tuple(range(1, 40))
LOAD_BUSES = (3, 4, 7, 8, 12, 15, 16, 18, 20, 21, 23, 24, 25, 26, 27, 28, 29)
LABEL_COLUMNS = {
    "sample_id",
    "sim_id",
    "placement_id",
    "observed_pmus",
    "event_label",
    "abnormal_label",
    "location_label",
    "location_type",
}


def _bus_from_path(path: Path) -> int | None:
    match = re.search(r"Bus(\d+)", path.name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _load_all_bus_frames(sim_path: Path) -> dict[int, pd.DataFrame]:
    frames: dict[int, pd.DataFrame] = {}
    for folder in (sim_path / "pmu", sim_path / "nonpmu"):
        if not folder.exists():
            continue
        for path in folder.glob("Bus*.csv"):
            bus = _bus_from_path(path)
            if bus is not None and bus not in frames:
                frames[bus] = pd.read_csv(path).sort_values("TIMESTAMP").reset_index(drop=True)
    return dict(sorted(frames.items()))


def _event_label(scenario: dict[str, Any]) -> int:
    event_types = scenario.get("labels", {}).get("event_types", [])
    if not event_types:
        return 0
    match = re.search(r"event(\d+)", str(event_types[0]))
    return int(match.group(1)) if match else 8


def _location_from_scenario(scenario: dict[str, Any]) -> str:
    labels = scenario.get("labels", {})
    event = _event_label(scenario)
    if event == 0:
        return "none"
    physical = labels.get("physical_targets", [])
    cyber = labels.get("cyber_targets", [])
    if event in {5, 7} and cyber:
        target = cyber[0]
        pmu = target.get("pmu") if isinstance(target, dict) else target
        return f"PMU{int(pmu)}" if pmu is not None else "none"
    if physical:
        target = physical[0]
        if target.get("line_from") is not None and target.get("line_to") is not None:
            a = min(int(target["line_from"]), int(target["line_to"]))
            b = max(int(target["line_from"]), int(target["line_to"]))
            return f"LINE{a}-{b}"
        if target.get("bus") is not None:
            return f"BUS{int(target['bus'])}"
    if cyber:
        target = cyber[0]
        pmu = target.get("pmu") if isinstance(target, dict) else target
        return f"PMU{int(pmu)}" if pmu is not None else "none"
    return "none"


def _location_type(label: str) -> str:
    value = str(label)
    if value.startswith("BUS"):
        return "BUS"
    if value.startswith("LINE"):
        return "LINE"
    if value.startswith("PMU"):
        return "PMU"
    return "NONE"


def _lines(topology_dir: Path = DEFAULT_TOPOLOGY_DIR) -> list[tuple[int, int]]:
    data = pd.read_csv(topology_dir / "branches_physical.csv")
    out = []
    for row in data.itertuples(index=False):
        out.append((min(int(row.from_bus), int(row.to_bus)), max(int(row.from_bus), int(row.to_bus))))
    return list(dict.fromkeys(out))


def _distance_lookup(distances: pd.DataFrame) -> dict[tuple[int, int], float]:
    out: dict[tuple[int, int], float] = {}
    for row in distances.itertuples(index=False):
        a = int(row.from_bus)
        b = int(row.to_bus)
        d = max(float(row.z_eff_abs), 1e-9)
        out[(a, b)] = d
        out[(b, a)] = d
    return out


def _label_buses(label: str, lines: list[tuple[int, int]]) -> list[int]:
    if label.startswith(("BUS", "PMU")):
        digits = "".join(ch for ch in label if ch.isdigit())
        return [int(digits)] if digits else []
    if label.startswith("LINE"):
        text = label.replace("LINE", "")
        if "-" in text:
            a, b = text.split("-", 1)
            return [int(a), int(b)]
    return []


def _candidate_set(event_label: int, observed_pmus: tuple[int, ...], lines: list[tuple[int, int]]) -> list[str]:
    event = int(event_label)
    if event == 0:
        return ["none"]
    if event in {5, 7}:
        return [f"PMU{bus}" for bus in observed_pmus]
    if event == 2:
        return [f"LINE{a}-{b}" for a, b in lines]
    if event == 4:
        return [f"BUS{bus}" for bus in LOAD_BUSES]
    return [f"BUS{bus}" for bus in ALL_BUSES]


def _hard_negative_candidates(
    true_location: str,
    event_label: int,
    observed_pmus: tuple[int, ...],
    lines: list[tuple[int, int]],
    distances: dict[tuple[int, int], float],
    rng: np.random.Generator,
    n_negatives: int,
) -> list[str]:
    candidates = [c for c in _candidate_set(event_label, observed_pmus, lines) if c != true_location]
    if len(candidates) <= n_negatives:
        return candidates
    true_buses = _label_buses(true_location, lines)
    scored = []
    for candidate in candidates:
        cbuses = _label_buses(candidate, lines)
        if true_buses and cbuses:
            score = min(distances.get((a, b), 1.0) for a in true_buses for b in cbuses)
        else:
            score = rng.random() + 1.0
        scored.append((score, candidate))
    scored.sort(key=lambda item: item[0])
    hard = [candidate for _, candidate in scored[: max(1, int(n_negatives * 0.75))]]
    remaining = [candidate for _, candidate in scored if candidate not in set(hard)]
    if remaining and len(hard) < n_negatives:
        hard.extend(list(rng.choice(remaining, size=min(len(remaining), n_negatives - len(hard)), replace=False)))
    return hard[:n_negatives]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0


def _expected_signature(
    candidate: str,
    observed_pmus: tuple[int, ...],
    distances: dict[tuple[int, int], float],
    tau: float,
) -> np.ndarray:
    buses = _label_buses(candidate, [])
    if not buses:
        return np.zeros(len(observed_pmus), dtype=float)
    values = []
    for pmu in observed_pmus:
        d = min(distances.get((pmu, bus), 1.0) for bus in buses)
        values.append(math.exp(-d / max(float(tau), 1e-9)))
    arr = np.asarray(values, dtype=float)
    return arr / max(float(np.linalg.norm(arr)), 1e-12)


def _candidate_features(
    candidate: str,
    event_label: int,
    observed_pmus: tuple[int, ...],
    severity: dict[int, float],
    distances: dict[tuple[int, int], float],
) -> dict[str, float]:
    ctype = _location_type(candidate)
    buses = _label_buses(candidate, [])
    sev = np.asarray([severity.get(bus, 0.0) for bus in observed_pmus], dtype=float)
    sev_norm = sev / max(float(np.linalg.norm(sev)), 1e-12)
    dist_values = []
    for pmu in observed_pmus:
        if buses:
            dist_values.append(min(distances.get((pmu, bus), 1.0) for bus in buses))
    dist = np.asarray(dist_values, dtype=float)
    weighted_distance = float(np.dot(sev, dist) / max(float(np.sum(sev)), 1e-12)) if dist.size else 1.0
    inv_distance = float(np.sum(sev / np.maximum(dist, 1e-9)) / max(float(np.sum(sev)), 1e-12)) if dist.size else 0.0
    scores = []
    residuals = []
    for tau in (0.05, 0.10, 0.20, 0.40, 0.80):
        expected = _expected_signature(candidate, observed_pmus, distances, tau)
        scores.append(_cosine(sev_norm, expected))
        residuals.append(float(np.linalg.norm(sev_norm - expected)))
    strongest_idx = int(np.nanargmax(sev)) if sev.size else 0
    strongest_pmu = observed_pmus[strongest_idx] if observed_pmus else -1
    expected_at_strongest = 0.0
    if buses and strongest_pmu > 0:
        expected_at_strongest = 1.0 / max(min(distances.get((strongest_pmu, bus), 1.0) for bus in buses), 1e-9)
    return {
        "pred_event": float(event_label),
        "event_is_fault": float(event_label == 1),
        "event_is_line": float(event_label == 2),
        "event_is_generation": float(event_label in {3, 6}),
        "event_is_load": float(event_label == 4),
        "event_is_mixed": float(event_label in {6, 8}),
        "candidate_is_bus": float(ctype == "BUS"),
        "candidate_is_line": float(ctype == "LINE"),
        "candidate_is_pmu": float(ctype == "PMU"),
        "candidate_is_observed": float(any(bus in set(observed_pmus) for bus in buses)),
        "candidate_is_load_class": float(any(bus in set(LOAD_BUSES) for bus in buses)),
        "candidate_is_generator_class": float(any(30 <= bus <= 39 for bus in buses)),
        "min_distance_to_observed": float(np.nanmin(dist)) if dist.size else 1.0,
        "mean_distance_to_observed": float(np.nanmean(dist)) if dist.size else 1.0,
        "max_distance_to_observed": float(np.nanmax(dist)) if dist.size else 1.0,
        "weighted_distance_to_severity": weighted_distance,
        "inverse_distance_score": inv_distance,
        "diffusion_cosine_max": float(np.nanmax(scores)) if scores else 0.0,
        "diffusion_cosine_mean": float(np.nanmean(scores)) if scores else 0.0,
        "diffusion_residual_min": float(np.nanmin(residuals)) if residuals else 1.0,
        "expected_at_strongest_pmu": float(expected_at_strongest),
        "strongest_pmu_energy": float(np.nanmax(sev)) if sev.size else 0.0,
    }


def _placement(
    available_buses: list[int],
    true_location: str,
    rng: np.random.Generator,
    min_pmus: int,
    max_pmus: int,
) -> tuple[int, ...]:
    k = int(rng.integers(min_pmus, min(max_pmus, len(available_buses)) + 1))
    chosen = set(int(v) for v in rng.choice(available_buses, size=k, replace=False))
    true_buses = [bus for bus in _label_buses(true_location, []) if bus in available_buses]
    if true_buses and rng.random() < 0.35:
        chosen.add(int(rng.choice(true_buses)))
    while len(chosen) > k:
        removable = [bus for bus in chosen if bus not in set(true_buses)]
        chosen.remove(int(rng.choice(removable or list(chosen))))
    return tuple(sorted(chosen))


def _window_row(
    sim_id: str,
    placement_id: int,
    observed: tuple[int, ...],
    event_label: int,
    location: str,
    bus_features: dict[int, dict[str, float]],
    distances_df: pd.DataFrame,
    lines: list[tuple[int, int]],
) -> dict[str, Any]:
    selected = {bus: bus_features[bus] for bus in observed}
    row: dict[str, Any] = {
        "sample_id": f"{sim_id}_P{placement_id:02d}",
        "sim_id": sim_id,
        "placement_id": int(placement_id),
        "observed_pmus": " ".join(str(bus) for bus in observed),
        "event_label": int(event_label),
        "abnormal_label": int(event_label != 0),
        "location_label": location,
        "location_type": _location_type(location),
    }
    row.update(aggregate_pmu_features(selected))
    graph = graph_feature_block(selected, distances_df, candidate_buses=list(ALL_BUSES), lines=lines)
    for key, value in graph.items():
        if re.search(r"BUS\d+|LINE\d+-\d+", key):
            continue
        row[key] = value
    return row


def build_dataset(
    sim_dir: Path = DEFAULT_SIM_DIR,
    scenario_dir: Path = DEFAULT_SCENARIO_DIR,
    topology_dir: Path = DEFAULT_TOPOLOGY_DIR,
    out_dir: Path = DEFAULT_OUT,
    max_scenarios: int | None = None,
    start_index: int = 0,
    placements_per_scenario: int = 4,
    min_pmus: int = 4,
    max_pmus: int = 12,
    negatives_per_positive: int = 12,
    random_state: int = 20260504,
) -> PipelineResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(random_state))
    distances_df = load_zbus_distances(topology_dir)
    distances = _distance_lookup(distances_df)
    lines = _lines(topology_dir)
    scenario_paths = sorted(scenario_dir.glob("SIM*.json"))
    if start_index:
        scenario_paths = scenario_paths[int(start_index) :]
    if max_scenarios is not None:
        scenario_paths = scenario_paths[: int(max_scenarios)]
    window_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for scenario_path in scenario_paths:
        scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
        sim_id = str(scenario.get("sim_id") or scenario_path.stem)
        frames = _load_all_bus_frames(sim_dir / sim_id)
        if not frames:
            continue
        event_label = _event_label(scenario)
        location = _location_from_scenario(scenario)
        available = sorted(frames)
        bus_features = {bus: pmu_window_features(frame, bus) for bus, frame in frames.items()}
        for placement_id in range(int(placements_per_scenario)):
            observed = _placement(available, location, rng, min_pmus, max_pmus)
            sample = _window_row(sim_id, placement_id, observed, event_label, location, bus_features, distances_df, lines)
            window_rows.append(sample)
            if event_label == 0 or location == "none":
                continue
            severity = pmu_severity_vector({bus: bus_features[bus] for bus in observed})
            candidates = [location] + _hard_negative_candidates(
                location,
                event_label,
                observed,
                lines,
                distances,
                rng,
                negatives_per_positive,
            )
            for candidate in candidates:
                row = {
                    "sample_id": sample["sample_id"],
                    "sim_id": sim_id,
                    "event_label": int(event_label),
                    "location_label": location,
                    "candidate_label": candidate,
                    "target": int(candidate == location),
                    "observed_pmus": sample["observed_pmus"],
                }
                row.update(_candidate_features(candidate, event_label, observed, severity, distances))
                candidate_rows.append(row)
    windows = pd.DataFrame(window_rows)
    candidates = pd.DataFrame(candidate_rows)
    windows_csv = out_dir / "agnostic_windows.csv"
    candidates_csv = out_dir / "agnostic_candidate_rows.csv"
    windows.to_csv(windows_csv, index=False)
    candidates.to_csv(candidates_csv, index=False)
    feature_cols = [col for col in windows.columns if col not in LABEL_COLUMNS and windows[col].dtype.kind in "bifc"]
    candidate_feature_cols = [
        col
        for col in candidates.columns
        if col not in {"sample_id", "sim_id", "event_label", "location_label", "candidate_label", "target", "observed_pmus"}
        and candidates[col].dtype.kind in "bifc"
    ]
    report = {
        "n_windows": int(len(windows)),
        "n_candidate_rows": int(len(candidates)),
        "placements_per_scenario": int(placements_per_scenario),
        "pmu_range": [int(min_pmus), int(max_pmus)],
        "window_feature_count": int(len(feature_cols)),
        "candidate_feature_count": int(len(candidate_feature_cols)),
        "contains_bus_specific_window_features": bool(any(re.search(r"BUS\d+|LINE\d+-\d+", col) for col in feature_cols)),
        "contains_bus_specific_candidate_features": bool(any(re.search(r"BUS\d+|LINE\d+-\d+", col) for col in candidate_feature_cols)),
    }
    report_path = out_dir / "p12_report.json"
    write_json(report_path, report)
    result = PipelineResult(
        name="p12_build_agnostic_training_set",
        status="completed",
        outputs={
            "windows": str(windows_csv.resolve()),
            "candidate_rows": str(candidates_csv.resolve()),
            "report": str(report_path.resolve()),
        },
        metrics=report,
        notes=[
            "Features used for model training are placement agnostic; bus/location labels are retained only as targets and candidate identifiers.",
            "PMU placements are randomly resampled per scenario from all available simulated bus streams.",
        ],
    )
    write_json(out_dir / "p12_pipeline_result.json", result.to_dict())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build bus-agnostic placement-augmented SGSMA training tables.")
    parser.add_argument("--sim-dir", type=Path, default=DEFAULT_SIM_DIR)
    parser.add_argument("--scenario-dir", type=Path, default=DEFAULT_SCENARIO_DIR)
    parser.add_argument("--topology-dir", type=Path, default=DEFAULT_TOPOLOGY_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--placements-per-scenario", type=int, default=4)
    parser.add_argument("--min-pmus", type=int, default=4)
    parser.add_argument("--max-pmus", type=int, default=12)
    parser.add_argument("--negatives-per-positive", type=int, default=12)
    parser.add_argument("--random-state", type=int, default=20260504)
    args = parser.parse_args()
    result = build_dataset(
        sim_dir=args.sim_dir,
        scenario_dir=args.scenario_dir,
        topology_dir=args.topology_dir,
        out_dir=args.out_dir,
        max_scenarios=args.max_scenarios,
        start_index=args.start_index,
        placements_per_scenario=args.placements_per_scenario,
        min_pmus=args.min_pmus,
        max_pmus=args.max_pmus,
        negatives_per_positive=args.negatives_per_positive,
        random_state=args.random_state,
    )
    print(json.dumps(json_safe(result.to_dict()), indent=2))


if __name__ == "__main__":
    main()
