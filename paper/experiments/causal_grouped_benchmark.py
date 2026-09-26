"""Reproducible causal grouped-split benchmark for PI-HED revisions.

The module is self-contained by design.  It imports the calibrated IEEE-39
scenario generator, but it neither loads nor modifies the frozen competition
models.  All generated files live below ``paper/evidence/revised``.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import pickle
import platform
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

# Direct execution sets sys.path[0] to paper/experiments rather than the
# repository root.  Add the root before importing the project package.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src.simulation.transient_scenarios import (
    DEFAULT_PMU_BUSES,
    MeasurementCalibration,
    ScenarioDataset,
    ScenarioSpec,
    calibrate_from_event0,
    simulate_scenario,
)


DEFAULT_OUTPUT = REPO_ROOT / "paper" / "evidence" / "revised" / "causal_benchmark"
TOPOLOGY_DIR = REPO_ROOT / "data" / "topology" / "ieee39"
EVENT_NAMES = {
    0: "normal",
    1: "fault",
    2: "line_outage",
    3: "generation_change",
    4: "load_change",
    5: "missing_data",
    6: "physical_plus_missing",
    7: "bad_data",
    8: "unknown_mixed",
}
PHYSICAL_EVENTS = (1, 2, 3, 4, 6, 8)
INTEGRITY_EVENTS = (5, 6, 7, 8)
GENERATOR_DEVICE_TO_REPORT_BUS = {
    30: 2,
    31: 6,
    32: 10,
    33: 19,
    34: 20,
    35: 22,
    36: 23,
    37: 25,
    38: 29,
    39: 39,
}
GENERATOR_REPORT_TO_DEVICE_BUS = {
    report: device for device, report in GENERATOR_DEVICE_TO_REPORT_BUS.items()
}
LOAD_REPORT_BUSES = (3, 4, 7, 8, 12, 15, 16, 18, 20, 21, 23, 24, 25, 26, 27, 28, 29, 39)


def _canonical_bus(value: str | int) -> str:
    text = str(value).upper().replace("BUS", "")
    return f"BUS{int(text)}"


def _canonical_line(value: str) -> str:
    text = str(value).upper().replace("LINE", "")
    left, right = (int(part) for part in text.split("-", 1))
    a, b = sorted((left, right))
    return f"LINE{a}-{b}"


@dataclass(frozen=True)
class CandidateUniverses:
    buses: tuple[str, ...]
    lines: tuple[str, ...]
    generator_buses: tuple[str, ...]
    load_buses: tuple[str, ...]
    pmu_buses: tuple[str, ...]

    def physical(self, event_type: int) -> tuple[str, ...]:
        return {
            1: self.buses,
            2: self.lines,
            3: self.generator_buses,
            4: self.load_buses,
            6: self.generator_buses,
            8: self.generator_buses,
        }.get(int(event_type), ())

    def integrity(self, event_type: int) -> tuple[str, ...]:
        return self.pmu_buses if int(event_type) in INTEGRITY_EVENTS else ()

    def to_jsonable(self) -> dict[str, list[str]]:
        return {key: list(value) for key, value in asdict(self).items()}


def load_candidate_universes() -> CandidateUniverses:
    """Load device universes and exclude transformers from line candidates."""

    branches = pd.read_csv(TOPOLOGY_DIR / "branches_physical.csv")
    is_line = np.isclose(branches["tap"].to_numpy(float), 1.0) & np.isclose(
        branches["shift_deg"].to_numpy(float), 0.0
    )
    line_rows = branches.loc[is_line]
    lines = tuple(
        sorted(
            (_canonical_line(f"{int(row.from_bus)}-{int(row.to_bus)}") for row in line_rows.itertuples()),
            key=lambda token: tuple(int(x) for x in token[4:].split("-")),
        )
    )
    if len(lines) != 34 or int((~is_line).sum()) != 12:
        raise RuntimeError(
            f"IEEE-39 candidate audit failed: {len(lines)} lines and {(~is_line).sum()} transformers"
        )

    import andes

    system = andes.load(andes.get_case("ieee39/ieee39_full.xlsx"), setup=False, no_output=True)
    buses = tuple(_canonical_bus(bus) for bus in sorted(set(int(x) for x in system.Bus.idx.v)))
    generator_devices = sorted(set(int(x) for x in system.GENROU.bus.v))
    if set(generator_devices) != set(GENERATOR_DEVICE_TO_REPORT_BUS):
        raise RuntimeError("ANDES generator terminals differ from the audited reporting map")
    generators = tuple(
        _canonical_bus(bus) for bus in sorted(GENERATOR_DEVICE_TO_REPORT_BUS.values())
    )
    andes_load_buses = set(int(bus) for bus in system.PQ.bus.v)
    if not set(LOAD_REPORT_BUSES).issubset(andes_load_buses):
        raise RuntimeError("The audited nonzero-load universe is not simulatable by the ANDES case")
    # The ANDES workbook contains an additional nonzero proxy at BUS31.  The
    # supplied competition RAW record has zero demand there, so the task-facing
    # candidate universe follows the RAW system definition and excludes BUS31.
    loads = tuple(_canonical_bus(bus) for bus in LOAD_REPORT_BUSES)
    pmus = tuple(_canonical_bus(bus) for bus in DEFAULT_PMU_BUSES)
    return CandidateUniverses(buses, lines, generators, loads, pmus)


@dataclass(frozen=True)
class ScenarioRecord:
    spec: ScenarioSpec
    target_key: str
    replicate: int


def _paired(values_a: Sequence[str], values_b: Sequence[str]) -> list[tuple[str, str]]:
    count = max(len(values_a), len(values_b))
    return [(values_a[i % len(values_a)], values_b[i % len(values_b)]) for i in range(count)]


def target_keys(universes: CandidateUniverses, smoke: bool = False) -> dict[int, list[tuple[str | None, str | None]]]:
    """Return predeclared physical/integrity targets for each event."""

    keys: dict[int, list[tuple[str | None, str | None]]] = {
        0: [(None, None)],
        1: [(x, None) for x in universes.buses],
        2: [(x, None) for x in universes.lines],
        3: [(x, None) for x in universes.generator_buses],
        4: [(x, None) for x in universes.load_buses],
        5: [(None, x) for x in universes.pmu_buses],
        6: list(_paired(universes.generator_buses, universes.pmu_buses)),
        7: [(None, x) for x in universes.pmu_buses],
        8: list(_paired(universes.generator_buses, universes.pmu_buses)),
    }
    if smoke:
        return {event: values[:1] for event, values in keys.items()}
    return keys


def build_scenario_records(
    universes: CandidateUniverses,
    *,
    replicas_per_target: int = 5,
    smoke: bool = False,
    base_seed: int = 202_608,
) -> list[ScenarioRecord]:
    """Build independent scenarios; splitting occurs later by replica."""

    if replicas_per_target < 5:
        raise ValueError("Primary 60/20/20 target-covered protocol requires at least five replicas")
    records: list[ScenarioRecord] = []
    keys = target_keys(universes, smoke=smoke)
    for event_type, pairs in keys.items():
        for target_index, (physical, integrity) in enumerate(pairs):
            target_key = f"e{event_type}:{physical or 'NONE'}:{integrity or 'NONE'}"
            for replica in range(replicas_per_target):
                severity = 0.06 + 0.03 * ((replica + target_index) % 4)
                operating_scale = 0.97 + 0.015 * ((2 * replica + target_index) % 5)
                target: str | int | None = physical
                if physical and physical.startswith("BUS"):
                    report_bus = int(physical[3:])
                    target = (
                        GENERATOR_REPORT_TO_DEVICE_BUS[report_bus]
                        if event_type in {3, 6, 8}
                        else report_bus
                    )
                elif physical and physical.startswith("LINE"):
                    target = physical[4:]
                cyber_target = int(integrity[3:]) if integrity else None
                duration = 0.20 if event_type == 1 else 0.70
                scenario_seed = base_seed + 100_003 * event_type + 101 * target_index + replica
                safe_target = (physical or integrity or "normal").replace("-", "_")
                spec = ScenarioSpec(
                    scenario_id=f"e{event_type}_{safe_target}_r{replica:02d}",
                    event_type=event_type,
                    target=target,
                    cyber_target=cyber_target,
                    severity=severity,
                    event_start_s=2.0,
                    event_duration_s=duration,
                    simulation_end_s=5.0,
                    tstep_s=1.0 / 30.0,
                    operating_scale=operating_scale,
                    seed=scenario_seed,
                )
                records.append(ScenarioRecord(spec, target_key, replica))
    ids = [record.spec.scenario_id for record in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Scenario identifiers are not unique")
    return records


def assign_primary_splits(
    records: Sequence[ScenarioRecord], split_seed: int = 17
) -> dict[str, str]:
    """Assign 60/20/20 within every target key, never within a scenario."""

    grouped: dict[str, list[ScenarioRecord]] = defaultdict(list)
    for record in records:
        grouped[record.target_key].append(record)
    assignments: dict[str, str] = {}
    for key, members in sorted(grouped.items()):
        if len(members) < 5:
            raise ValueError(f"Target {key} has fewer than five independent replicas")
        digest = int(hashlib.sha256(f"{split_seed}:{key}".encode()).hexdigest()[:16], 16)
        rng = np.random.default_rng(digest)
        order = rng.permutation(len(members))
        n_train = max(3, int(math.floor(0.6 * len(members))))
        n_val = max(1, int(math.floor(0.2 * len(members))))
        if n_train + n_val >= len(members):
            n_train = len(members) - 2
            n_val = 1
        split_by_pos = {
            **{int(pos): "train" for pos in order[:n_train]},
            **{int(pos): "validation" for pos in order[n_train : n_train + n_val]},
            **{int(pos): "test" for pos in order[n_train + n_val :]},
        }
        for pos, member in enumerate(members):
            assignments[member.spec.scenario_id] = split_by_pos[pos]
    audit_primary_splits(records, assignments)
    return assignments


def audit_primary_splits(records: Sequence[ScenarioRecord], assignments: Mapping[str, str]) -> None:
    expected = {record.spec.scenario_id for record in records}
    if set(assignments) != expected:
        raise AssertionError("Split map does not contain each scenario exactly once")
    by_target: dict[str, set[str]] = defaultdict(set)
    for record in records:
        by_target[record.target_key].add(assignments[record.spec.scenario_id])
    missing = [key for key, splits in by_target.items() if splits != {"train", "validation", "test"}]
    if missing:
        raise AssertionError(f"Targets missing from at least one primary split: {missing[:5]}")


def assign_leave_target_out(
    records: Sequence[ScenarioRecord], split_seed: int = 17
) -> dict[str, str]:
    """Secondary stress split with target keys disjoint across partitions."""

    event_keys: dict[int, list[str]] = defaultdict(list)
    for record in records:
        if record.target_key not in event_keys[record.spec.event_type]:
            event_keys[record.spec.event_type].append(record.target_key)
    key_split: dict[str, str] = {}
    for event_type, keys in sorted(event_keys.items()):
        if event_type == 0 or len(keys) < 3:
            key_split.update({key: "train" for key in keys})
            continue
        rng = np.random.default_rng(split_seed + 1009 * event_type)
        ordered = [keys[i] for i in rng.permutation(len(keys))]
        n_test = max(1, round(0.2 * len(keys)))
        n_val = max(1, round(0.2 * len(keys)))
        for key in ordered[:n_test]:
            key_split[key] = "test"
        for key in ordered[n_test : n_test + n_val]:
            key_split[key] = "validation"
        for key in ordered[n_test + n_val :]:
            key_split[key] = "train"
    assignments = {record.spec.scenario_id: key_split[record.target_key] for record in records}
    # A normal scenario has no target to hold out.  Preserve normal exposure in
    # all partitions so FAR and the detector threshold remain defined.
    normals = [record for record in records if record.spec.event_type == 0]
    normal_primary = assign_primary_splits(normals, split_seed=split_seed)
    assignments.update(normal_primary)
    return assignments


@dataclass
class FeatureTable:
    X: np.ndarray
    pmu_scores: np.ndarray
    event: np.ndarray
    physical: np.ndarray
    integrity: np.ndarray
    scenario_id: np.ndarray
    time_s: np.ndarray
    dt_s: np.ndarray
    feature_names: tuple[str, ...]

    def subset(self, mask: np.ndarray) -> "FeatureTable":
        return FeatureTable(
            X=self.X[mask],
            pmu_scores=self.pmu_scores[mask],
            event=self.event[mask],
            physical=self.physical[mask],
            integrity=self.integrity[mask],
            scenario_id=self.scenario_id[mask],
            time_s=self.time_s[mask],
            dt_s=self.dt_s[mask],
            feature_names=self.feature_names,
        )


def _safe_window_stats(values: np.ndarray, current: np.ndarray, duration_s: float) -> np.ndarray:
    """Return strictly backward-looking statistics for one signal matrix."""

    if values.ndim != 2:
        raise ValueError("values must have shape (time, pmu)")
    finite = np.isfinite(values)
    count = finite.sum(axis=0)
    total = np.nansum(values, axis=0)
    mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    centered = np.where(finite, values - mean, 0.0)
    variance = np.divide(
        np.sum(centered * centered, axis=0), count, out=np.zeros_like(total), where=count > 0
    )
    std = np.sqrt(np.maximum(variance, 0.0))
    first = np.zeros(values.shape[1], dtype=float)
    last = np.zeros(values.shape[1], dtype=float)
    for col in range(values.shape[1]):
        valid = values[:, col][finite[:, col]]
        if len(valid):
            first[col], last[col] = valid[0], valid[-1]
        else:
            first[col] = last[col] = 0.0
    filled_current = np.where(np.isfinite(current), current, mean)
    delta = filled_current - first
    slope = (last - first) / max(float(duration_s), 1e-9)
    return np.column_stack([filled_current, delta, mean, std, slope])


def _unwrap_preserving_missing(values: np.ndarray) -> np.ndarray:
    output = np.asarray(values, dtype=float).copy()
    for col in range(output.shape[1]):
        finite = np.isfinite(output[:, col])
        if np.any(finite):
            output[finite, col] = np.unwrap(output[finite, col])
    return output


def extract_causal_features(
    dataset: ScenarioDataset,
    *,
    window_s: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Extract trailing features; index ``i`` never reads a sample after ``i``."""

    if not 1.0 <= window_s <= 5.0:
        raise ValueError("window_s must be between 1 and 5 seconds")
    time_axis = np.asarray(dataset.time_s, dtype=float)
    if len(time_axis) < 2:
        raise ValueError("Scenario has fewer than two timestamps")
    dt = float(np.median(np.diff(time_axis)))
    samples = max(2, int(round(window_s / max(dt, 1e-9))) + 1)
    signals = (
        ("v", np.asarray(dataset.observed_voltage_pu, dtype=float)),
        ("a", _unwrap_preserving_missing(np.asarray(dataset.observed_angle_rad, dtype=float))),
        ("f", np.asarray(dataset.observed_frequency_hz, dtype=float)),
    )
    mask = np.asarray(dataset.observed_mask, dtype=bool)
    n_rows, n_pmus = signals[0][1].shape
    if n_pmus != len(dataset.pmu_buses):
        raise ValueError("Observed signal width does not match PMU list")

    rows: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    for idx in range(n_rows):
        start = max(0, idx - samples + 1)
        duration = max(time_axis[idx] - time_axis[start], dt)
        blocks: list[np.ndarray] = []
        score_terms: list[np.ndarray] = []
        for _, matrix in signals:
            block = _safe_window_stats(matrix[start : idx + 1], matrix[idx], duration)
            blocks.append(block)
            scale = np.maximum(block[:, 3], 1e-6)
            score_terms.append(np.abs(block[:, 1]) / scale)
        missing_fraction = 1.0 - mask[start : idx + 1].mean(axis=0)
        pmu_score = np.log1p(np.mean(score_terms, axis=0)) + 3.0 * missing_fraction
        # PMU-major layout keeps each device's causal measurements adjacent.
        row = np.column_stack([blocks[0], blocks[1], blocks[2], missing_fraction, pmu_score]).ravel()
        rows.append(row)
        scores.append(pmu_score)

    stat_names = ("current", "delta", "mean", "std", "slope")
    names: list[str] = []
    for bus in dataset.pmu_buses:
        for signal_name in ("v", "a", "f"):
            names.extend(f"bus{int(bus)}_{signal_name}_{stat}" for stat in stat_names)
        names.extend((f"bus{int(bus)}_missing_fraction", f"bus{int(bus)}_disturbance_score"))
    return np.asarray(rows, dtype=np.float32), np.asarray(scores, dtype=np.float32), tuple(names)


def _physical_label(dataset: ScenarioDataset) -> str:
    event_type = int(dataset.spec.event_type)
    if event_type not in PHYSICAL_EVENTS:
        return ""
    value = str(dataset.metadata.get("physical_target") or "")
    if event_type == 2:
        return _canonical_line(value)
    if event_type == 8 and "+" in value:
        # Existing simulator also perturbs a random load.  Keep the declared
        # generator target as the primary single-label physical output.
        value = value.split("+", 1)[0]
    if event_type in {3, 6, 8}:
        device_bus = int(value.upper().replace("BUS", ""))
        return _canonical_bus(GENERATOR_DEVICE_TO_REPORT_BUS[device_bus])
    return _canonical_bus(value)


def scenario_to_table(dataset: ScenarioDataset, window_s: float) -> FeatureTable:
    X, scores, names = extract_causal_features(dataset, window_s=window_s)
    event = np.asarray(dataset.event_label, dtype=int)
    physical_token = _physical_label(dataset)
    integrity_token = (
        _canonical_bus(dataset.spec.cyber_target)
        if dataset.spec.event_type in INTEGRITY_EVENTS and dataset.spec.cyber_target is not None
        else ""
    )
    physical = np.where(event != 0, physical_token, "").astype(object)
    integrity = np.where(event != 0, integrity_token, "").astype(object)
    dt = np.diff(dataset.time_s, prepend=dataset.time_s[0])
    if len(dt) > 1:
        dt[0] = float(np.median(dt[1:]))
    return FeatureTable(
        X=X,
        pmu_scores=scores,
        event=event,
        physical=physical,
        integrity=integrity,
        scenario_id=np.full(len(event), dataset.spec.scenario_id, dtype=object),
        time_s=np.asarray(dataset.time_s, dtype=float),
        dt_s=np.asarray(dt, dtype=float),
        feature_names=names,
    )


def concatenate_tables(tables: Sequence[FeatureTable]) -> FeatureTable:
    if not tables:
        raise ValueError("No scenario tables supplied")
    names = tables[0].feature_names
    if any(table.feature_names != names for table in tables):
        raise ValueError("Feature schemas differ across scenarios")
    return FeatureTable(
        X=np.concatenate([table.X for table in tables]),
        pmu_scores=np.concatenate([table.pmu_scores for table in tables]),
        event=np.concatenate([table.event for table in tables]),
        physical=np.concatenate([table.physical for table in tables]),
        integrity=np.concatenate([table.integrity for table in tables]),
        scenario_id=np.concatenate([table.scenario_id for table in tables]),
        time_s=np.concatenate([table.time_s for table in tables]),
        dt_s=np.concatenate([table.dt_s for table in tables]),
        feature_names=names,
    )


def _load_or_simulate(
    record: ScenarioRecord,
    calibration: MeasurementCalibration,
    cache_root: Path,
    force: bool,
) -> ScenarioDataset:
    directory = cache_root / record.spec.scenario_id
    if not force and (directory / "trajectory.npz").exists() and (directory / "scenario.json").exists():
        return ScenarioDataset.load(directory)
    print(
        f"[ANDES] {record.spec.scenario_id} event={record.spec.event_type} "
        f"target={record.spec.target} cyber={record.spec.cyber_target}",
        flush=True,
    )
    dataset = simulate_scenario(record.spec, calibration, output_dir=None)
    dataset.save(directory)
    return dataset


class ElectricalDistance:
    def __init__(self, path: Path = TOPOLOGY_DIR / "zbus_effective_distance_full.csv") -> None:
        frame = pd.read_csv(path)
        self._distance = {
            (int(row.from_bus), int(row.to_bus)): float(row.z_eff_abs) for row in frame.itertuples()
        }
        values = np.asarray(list(self._distance.values()), dtype=float)
        positive = values[values > 0]
        self.scale = float(np.median(positive)) if len(positive) else 1.0

    @staticmethod
    def endpoints(token: str) -> tuple[int, ...]:
        if token.startswith("BUS"):
            return (int(token[3:]),)
        if token.startswith("LINE"):
            return tuple(int(x) for x in token[4:].split("-", 1))
        raise ValueError(f"Unknown location token {token!r}")

    def bus(self, left: int, right: int) -> float:
        return self._distance.get((left, right), self._distance.get((right, left), float("nan")))

    def tokens(self, left: str, right: str) -> float:
        a, b = self.endpoints(left), self.endpoints(right)
        forward = [min(self.bus(x, y) for y in b) for x in a]
        backward = [min(self.bus(y, x) for x in a) for y in b]
        return float(0.5 * (np.mean(forward) + np.mean(backward)))

    def candidate_to_pmus(self, token: str, pmus: Sequence[int]) -> np.ndarray:
        endpoints = self.endpoints(token)
        return np.asarray([min(self.bus(endpoint, int(pmu)) for endpoint in endpoints) for pmu in pmus])


def topology_candidate_matrix(
    X: np.ndarray,
    pmu_scores: np.ndarray,
    candidates: Sequence[str],
    distance: ElectricalDistance,
    pmus: Sequence[int] = DEFAULT_PMU_BUSES,
) -> np.ndarray:
    """Build candidate-conditioned ranker rows for one candidate per sample."""

    if len(X) != len(candidates) or len(X) != len(pmu_scores):
        raise ValueError("X, scores, and candidate vectors must have equal length")
    static: list[np.ndarray] = []
    interaction: list[np.ndarray] = []
    for score, candidate in zip(pmu_scores, candidates):
        d = distance.candidate_to_pmus(str(candidate), pmus) / max(distance.scale, 1e-12)
        affinity = np.exp(-np.clip(d, 0.0, 50.0))
        endpoints = distance.endpoints(str(candidate))
        kind = np.array([float(str(candidate).startswith("BUS")), float(str(candidate).startswith("LINE"))])
        endpoint_features = np.array(
            [min(endpoints) / 39.0, max(endpoints) / 39.0, len(endpoints) / 2.0]
        )
        static.append(np.concatenate([kind, endpoint_features, d, [d.min(), d.mean(), d.max()]]))
        weighted = score * affinity
        interaction.append(np.concatenate([weighted, [weighted.sum(), weighted.max()]]))
    return np.column_stack([X, np.asarray(static, dtype=np.float32), np.asarray(interaction, dtype=np.float32)])


def _make_classifier(n_trees: int, seed: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=int(n_trees),
        max_depth=14,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=int(seed),
        n_jobs=-1,
    )


def _fit_classifier(X: np.ndarray, y: np.ndarray, n_trees: int, seed: int):
    labels = np.unique(y)
    if len(labels) == 0:
        raise ValueError("Cannot fit a classifier without rows")
    if len(labels) == 1:
        return DummyClassifier(strategy="constant", constant=labels[0]).fit(X, y)
    return _make_classifier(n_trees, seed).fit(X, y)


def _rank_classes(model, X: np.ndarray, k: int = 3) -> list[list[str]]:
    probabilities = model.predict_proba(X)
    if probabilities.ndim == 1:
        probabilities = probabilities[:, None]
    classes = np.asarray(model.classes_).astype(str)
    width = min(int(k), len(classes))
    order = np.argsort(-probabilities, axis=1, kind="stable")[:, :width]
    return [[str(classes[index]) for index in row] for row in order]


def _tree_budget(total: int, heads: Sequence[int]) -> dict[int, int]:
    if total < len(heads):
        raise ValueError(f"Tree budget {total} is smaller than {len(heads)} typed heads")
    base, remainder = divmod(int(total), len(heads))
    return {event: base + (position < remainder) for position, event in enumerate(heads)}


def _select_validation_threshold(score: np.ndarray, truth: np.ndarray) -> float:
    candidates = np.linspace(0.1, 0.9, 33)
    values = [f1_score(truth, score >= threshold, zero_division=0) for threshold in candidates]
    best = max(values)
    return float(candidates[values.index(best)])


def _choose_detector_threshold(model, validation: FeatureTable) -> float:
    probabilities = model.predict_proba(validation.X)
    classes = list(model.classes_)
    if 1 not in classes:
        return 0.5
    score = probabilities[:, classes.index(1)]
    return _select_validation_threshold(score, validation.event != 0)


def _flat_event_scores(model, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray(model.predict_proba(X), dtype=float)
    if probabilities.ndim == 1:
        probabilities = probabilities[:, None]
    classes = np.asarray(model.classes_, dtype=int)
    normal_positions = np.flatnonzero(classes == 0)
    normal_probability = (
        probabilities[:, int(normal_positions[0])]
        if len(normal_positions)
        else np.zeros(len(probabilities), dtype=float)
    )
    abnormal_positions = np.flatnonzero(classes != 0)
    labels = np.zeros(len(probabilities), dtype=int)
    if len(abnormal_positions):
        local = np.argmax(probabilities[:, abnormal_positions], axis=1)
        labels = classes[abnormal_positions[local]]
    return 1.0 - normal_probability, labels


def _choose_flat_threshold(model, validation: FeatureTable) -> float:
    score, _ = _flat_event_scores(model, validation.X)
    return _select_validation_threshold(score, validation.event != 0)


def _event_augmented(X: np.ndarray, event: np.ndarray) -> np.ndarray:
    one_hot = np.zeros((len(X), 9), dtype=np.float32)
    one_hot[np.arange(len(X)), np.clip(np.asarray(event, dtype=int), 0, 8)] = 1.0
    return np.column_stack([X, one_hot])


@dataclass
class Prediction:
    event: np.ndarray
    physical_top: list[list[str]]
    integrity_top: list[list[str]]
    predict_seconds: float


class FlatBaseline:
    name = "flat"

    def __init__(self, tree_budget: int, seed: int) -> None:
        self.tree_budget = int(tree_budget)
        self.seed = int(seed)
        self.models: dict[str, Any] = {}
        self.detector_threshold = 0.5
        self.fit_seconds = 0.0

    def fit(self, train: FeatureTable, validation: FeatureTable) -> "FlatBaseline":
        started = time.perf_counter()
        self.models["event"] = _fit_classifier(train.X, train.event, self.tree_budget, self.seed)
        self.detector_threshold = _choose_flat_threshold(self.models["event"], validation)
        physical = train.physical != ""
        integrity = train.integrity != ""
        self.models["physical"] = _fit_classifier(
            train.X[physical], train.physical[physical], self.tree_budget, self.seed + 1
        )
        self.models["integrity"] = _fit_classifier(
            train.X[integrity], train.integrity[integrity], self.tree_budget, self.seed + 2
        )
        self.fit_seconds = time.perf_counter() - started
        return self

    def predict(self, table: FeatureTable) -> Prediction:
        started = time.perf_counter()
        abnormal_score, abnormal_label = _flat_event_scores(self.models["event"], table.X)
        event = np.where(abnormal_score >= self.detector_threshold, abnormal_label, 0).astype(int)
        physical_all = _rank_classes(self.models["physical"], table.X, 3)
        integrity_all = _rank_classes(self.models["integrity"], table.X, 3)
        physical = [row if int(label) in PHYSICAL_EVENTS else [] for row, label in zip(physical_all, event)]
        integrity = [row if int(label) in INTEGRITY_EVENTS else [] for row, label in zip(integrity_all, event)]
        return Prediction(event, physical, integrity, time.perf_counter() - started)


class TypedBaseline:
    name = "typed"

    def __init__(self, tree_budget: int, seed: int) -> None:
        self.tree_budget = int(tree_budget)
        self.seed = int(seed)
        self.models: dict[str, Any] = {}
        self.physical_heads: dict[int, Any] = {}
        self.integrity_heads: dict[int, Any] = {}
        self.detector_threshold = 0.5
        self.fit_seconds = 0.0

    def fit(self, train: FeatureTable, validation: FeatureTable) -> "TypedBaseline":
        started = time.perf_counter()
        detector_trees = self.tree_budget // 2
        event_trees = self.tree_budget - detector_trees
        self.models["detector"] = _fit_classifier(
            train.X, (train.event != 0).astype(int), detector_trees, self.seed
        )
        abnormal = train.event != 0
        self.models["event"] = _fit_classifier(
            train.X[abnormal], train.event[abnormal], event_trees, self.seed + 1
        )
        self.detector_threshold = _choose_detector_threshold(self.models["detector"], validation)

        physical_budget = _tree_budget(self.tree_budget, PHYSICAL_EVENTS)
        for event_type in PHYSICAL_EVENTS:
            mask = (train.event == event_type) & (train.physical != "")
            self.physical_heads[event_type] = _fit_classifier(
                train.X[mask], train.physical[mask], physical_budget[event_type], self.seed + 100 + event_type
            )
        integrity_budget = _tree_budget(self.tree_budget, INTEGRITY_EVENTS)
        for event_type in INTEGRITY_EVENTS:
            mask = (train.event == event_type) & (train.integrity != "")
            self.integrity_heads[event_type] = _fit_classifier(
                train.X[mask], train.integrity[mask], integrity_budget[event_type], self.seed + 200 + event_type
            )
        self.fit_seconds = time.perf_counter() - started
        return self

    def predict_event(self, X: np.ndarray) -> np.ndarray:
        detector = self.models["detector"]
        probabilities = detector.predict_proba(X)
        classes = list(detector.classes_)
        if 1 in classes:
            abnormal_score = probabilities[:, classes.index(1)]
        else:
            abnormal_score = np.zeros(len(X), dtype=float)
        event = np.zeros(len(X), dtype=int)
        routed = abnormal_score >= self.detector_threshold
        if np.any(routed):
            event[routed] = self.models["event"].predict(X[routed]).astype(int)
        return event

    def predict(self, table: FeatureTable) -> Prediction:
        started = time.perf_counter()
        event = self.predict_event(table.X)
        physical: list[list[str]] = [[] for _ in range(len(table.X))]
        integrity: list[list[str]] = [[] for _ in range(len(table.X))]
        for event_type, model in self.physical_heads.items():
            indices = np.flatnonzero(event == event_type)
            if len(indices):
                ranked = _rank_classes(model, table.X[indices], 3)
                for idx, row in zip(indices, ranked):
                    physical[int(idx)] = row
        for event_type, model in self.integrity_heads.items():
            indices = np.flatnonzero(event == event_type)
            if len(indices):
                ranked = _rank_classes(model, table.X[indices], 3)
                for idx, row in zip(indices, ranked):
                    integrity[int(idx)] = row
        return Prediction(event, physical, integrity, time.perf_counter() - started)


def _fit_candidate_ranker(
    table: FeatureTable,
    role: str,
    universes: CandidateUniverses,
    distance: ElectricalDistance,
    n_trees: int,
    seed: int,
    negatives_per_positive: int = 8,
    candidate_override: Mapping[tuple[str, int], Sequence[str]] | None = None,
):
    truth = table.physical if role == "physical" else table.integrity
    eligible = np.flatnonzero(truth != "")
    rng = np.random.default_rng(seed)
    rows: list[int] = []
    candidates: list[str] = []
    labels: list[int] = []
    for idx in eligible:
        event_type = int(table.event[idx])
        default_universe = (
            universes.physical(event_type) if role == "physical" else universes.integrity(event_type)
        )
        universe = tuple((candidate_override or {}).get((role, event_type), default_universe))
        correct = str(truth[idx])
        if correct not in universe:
            raise AssertionError(f"Truth {correct} is outside event-{event_type} {role} universe")
        rows.append(int(idx))
        candidates.append(correct)
        labels.append(1)
        negatives = [candidate for candidate in universe if candidate != correct]
        if len(negatives) > negatives_per_positive:
            selected = rng.choice(len(negatives), size=negatives_per_positive, replace=False)
            negatives = [negatives[int(pos)] for pos in selected]
        rows.extend([int(idx)] * len(negatives))
        candidates.extend(negatives)
        labels.extend([0] * len(negatives))
    selected_X = _event_augmented(table.X[rows], table.event[rows])
    matrix = topology_candidate_matrix(
        selected_X, table.pmu_scores[rows], candidates, distance
    )
    return _fit_classifier(matrix, np.asarray(labels, dtype=int), n_trees, seed)


class TypedTopologyBaseline(TypedBaseline):
    name = "typed_topology"

    def __init__(
        self,
        tree_budget: int,
        seed: int,
        universes: CandidateUniverses,
        distance: ElectricalDistance,
        training_candidates: Mapping[tuple[str, int], Sequence[str]] | None = None,
    ) -> None:
        super().__init__(tree_budget, seed)
        self.universes = universes
        self.distance = distance
        self.training_candidates = training_candidates
        self.rankers: dict[str, Any] = {}

    def fit(self, train: FeatureTable, validation: FeatureTable) -> "TypedTopologyBaseline":
        # Fit only the shared typed detector/event stages here.  Location tree
        # budgets are then assigned to one candidate ranker per output role.
        started = time.perf_counter()
        detector_trees = self.tree_budget // 2
        event_trees = self.tree_budget - detector_trees
        self.models["detector"] = _fit_classifier(
            train.X, (train.event != 0).astype(int), detector_trees, self.seed
        )
        abnormal = train.event != 0
        self.models["event"] = _fit_classifier(
            train.X[abnormal], train.event[abnormal], event_trees, self.seed + 1
        )
        self.detector_threshold = _choose_detector_threshold(self.models["detector"], validation)
        self.rankers["physical"] = _fit_candidate_ranker(
            train,
            "physical",
            self.universes,
            self.distance,
            self.tree_budget,
            self.seed + 300,
            candidate_override=self.training_candidates,
        )
        self.rankers["integrity"] = _fit_candidate_ranker(
            train,
            "integrity",
            self.universes,
            self.distance,
            self.tree_budget,
            self.seed + 400,
            candidate_override=self.training_candidates,
        )
        self.fit_seconds = time.perf_counter() - started
        return self

    def _rank_role(
        self, table: FeatureTable, event: np.ndarray, role: str
    ) -> list[list[str]]:
        output: list[list[str]] = [[] for _ in range(len(table.X))]
        ranker = self.rankers[role]
        valid_events = PHYSICAL_EVENTS if role == "physical" else INTEGRITY_EVENTS
        for event_type in valid_events:
            indices = np.flatnonzero(event == event_type)
            if not len(indices):
                continue
            universe = (
                self.universes.physical(event_type)
                if role == "physical"
                else self.universes.integrity(event_type)
            )
            expanded_rows = np.repeat(indices, len(universe))
            candidates = list(universe) * len(indices)
            augmented = _event_augmented(table.X[expanded_rows], event[expanded_rows])
            matrix = topology_candidate_matrix(
                augmented, table.pmu_scores[expanded_rows], candidates, self.distance
            )
            probabilities = ranker.predict_proba(matrix)
            classes = list(ranker.classes_)
            positive = probabilities[:, classes.index(1)] if 1 in classes else np.zeros(len(matrix))
            scores = positive.reshape(len(indices), len(universe))
            top_width = min(3, len(universe))
            ordering = np.argsort(-scores, axis=1, kind="stable")[:, :top_width]
            for idx, order in zip(indices, ordering):
                output[int(idx)] = [universe[int(pos)] for pos in order]
        return output

    def predict(self, table: FeatureTable) -> Prediction:
        started = time.perf_counter()
        event = self.predict_event(table.X)
        physical = self._rank_role(table, event, "physical")
        integrity = self._rank_role(table, event, "integrity")
        return Prediction(event, physical, integrity, time.perf_counter() - started)


def _false_alarm_metrics(table: FeatureTable, prediction: Prediction) -> dict[str, float]:
    false_samples = (table.event == 0) & (prediction.event != 0)
    normal = table.event == 0
    normal_minutes = float(np.sum(table.dt_s[normal]) / 60.0)
    episodes = 0
    for scenario in np.unique(table.scenario_id):
        indices = np.flatnonzero(table.scenario_id == scenario)
        order = indices[np.argsort(table.time_s[indices], kind="stable")]
        active = false_samples[order]
        truth_normal = normal[order]
        previous = False
        for is_false, is_normal in zip(active, truth_normal):
            current = bool(is_false and is_normal)
            if current and not previous:
                episodes += 1
            previous = current
    return {
        "normal_minutes": normal_minutes,
        "false_positive_samples": int(false_samples.sum()),
        "false_positive_fraction": float(false_samples.sum() / max(normal.sum(), 1)),
        "false_alarm_episodes": int(episodes),
        "false_alarm_episodes_per_min": float(episodes / max(normal_minutes, 1e-12)),
    }


def _detection_delay_metrics(table: FeatureTable, prediction: Prediction) -> dict[str, float]:
    delays: list[float] = []
    total = 0
    detected = 0
    for scenario in np.unique(table.scenario_id):
        indices = np.flatnonzero(table.scenario_id == scenario)
        order = indices[np.argsort(table.time_s[indices], kind="stable")]
        abnormal = table.event[order] != 0
        if not np.any(abnormal):
            continue
        total += 1
        onset = float(table.time_s[order][np.flatnonzero(abnormal)[0]])
        eligible = abnormal & (prediction.event[order] != 0)
        if np.any(eligible):
            alarm = float(table.time_s[order][np.flatnonzero(eligible)[0]])
            delays.append(max(0.0, alarm - onset))
            detected += 1
    return {
        "event_scenarios": int(total),
        "detected_event_scenarios": int(detected),
        "scenario_detection_rate": float(detected / max(total, 1)),
        "delay_mean_s": float(np.mean(delays)) if delays else float("nan"),
        "delay_median_s": float(np.median(delays)) if delays else float("nan"),
        "delay_p95_s": float(np.percentile(delays, 95)) if delays else float("nan"),
    }


def _localization_metrics(
    truth: np.ndarray,
    ranked: Sequence[Sequence[str]],
    distance: ElectricalDistance,
    prefix: str,
) -> dict[str, float]:
    eligible = np.flatnonzero(truth != "")
    if not len(eligible):
        return {
            f"{prefix}_n": 0,
            f"{prefix}_top1": float("nan"),
            f"{prefix}_top3": float("nan"),
            f"{prefix}_prediction_coverage": float("nan"),
            f"{prefix}_electrical_distance_conditional": float("nan"),
            f"{prefix}_electrical_distance_penalized": float("nan"),
        }
    top1, top3, covered = 0, 0, 0
    conditional: list[float] = []
    all_distances = np.asarray(list(distance._distance.values()), dtype=float)
    penalty = float(np.nanmax(all_distances))
    penalized: list[float] = []
    for idx in eligible:
        expected = str(truth[idx])
        candidates = list(ranked[int(idx)])
        if candidates:
            covered += 1
            top1 += int(candidates[0] == expected)
            top3 += int(expected in candidates[:3])
            value = distance.tokens(expected, candidates[0])
            conditional.append(value)
            penalized.append(value)
        else:
            penalized.append(penalty)
    count = len(eligible)
    return {
        f"{prefix}_n": int(count),
        f"{prefix}_top1": float(top1 / count),
        f"{prefix}_top3": float(top3 / count),
        f"{prefix}_prediction_coverage": float(covered / count),
        f"{prefix}_electrical_distance_conditional": (
            float(np.mean(conditional)) if conditional else float("nan")
        ),
        f"{prefix}_electrical_distance_penalized": float(np.mean(penalized)),
    }


def evaluate_predictions(
    table: FeatureTable,
    prediction: Prediction,
    distance: ElectricalDistance,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    true_detection = table.event != 0
    pred_detection = prediction.event != 0
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_detection, pred_detection, average="binary", zero_division=0
    )
    abnormal = true_detection
    event_macro_all = f1_score(
        table.event, prediction.event, labels=list(range(9)), average="macro", zero_division=0
    )
    event_weighted_all = f1_score(
        table.event, prediction.event, labels=list(range(9)), average="weighted", zero_division=0
    )
    event_macro = f1_score(
        table.event[abnormal],
        prediction.event[abnormal],
        labels=list(range(1, 9)),
        average="macro",
        zero_division=0,
    )
    event_weighted = f1_score(
        table.event[abnormal],
        prediction.event[abnormal],
        labels=list(range(1, 9)),
        average="weighted",
        zero_division=0,
    )
    labels = list(range(9))
    confusion = confusion_matrix(table.event, prediction.event, labels=labels)
    detailed = classification_report(
        table.event,
        prediction.event,
        labels=labels,
        target_names=[EVENT_NAMES[label] for label in labels],
        zero_division=0,
        output_dict=True,
    )
    metrics: dict[str, Any] = {
        "detection_precision": float(precision),
        "detection_recall": float(recall),
        "detection_f1": float(f1),
        "event_macro_f1_all_labels": float(event_macro_all),
        "event_weighted_f1_all_labels": float(event_weighted_all),
        "event_macro_f1_abnormal": float(event_macro),
        "event_weighted_f1_abnormal": float(event_weighted),
        **_false_alarm_metrics(table, prediction),
        **_detection_delay_metrics(table, prediction),
        **_localization_metrics(table.physical, prediction.physical_top, distance, "physical"),
        **_localization_metrics(table.integrity, prediction.integrity_top, distance, "integrity"),
    }
    return metrics, detailed, confusion


def model_complexity(model: Any) -> dict[str, float]:
    classifiers: list[Any] = []

    def collect(value: Any) -> None:
        if isinstance(value, (ExtraTreesClassifier, DummyClassifier)):
            classifiers.append(value)
        elif isinstance(value, Mapping):
            for item in value.values():
                collect(item)

    collect(getattr(model, "models", {}))
    collect(getattr(model, "physical_heads", {}))
    collect(getattr(model, "integrity_heads", {}))
    collect(getattr(model, "rankers", {}))
    trees = 0
    nodes = 0
    depths: list[int] = []
    for classifier in classifiers:
        for estimator in getattr(classifier, "estimators_", []):
            trees += 1
            nodes += int(estimator.tree_.node_count)
            depths.append(int(estimator.tree_.max_depth))
    payload = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
    return {
        "fitted_classifiers": int(len(classifiers)),
        "fitted_trees": int(trees),
        "fitted_nodes": int(nodes),
        "depth_mean": float(np.mean(depths)) if depths else 0.0,
        "depth_max": int(max(depths)) if depths else 0,
        "serialized_bytes": int(len(payload)),
    }


def prediction_frame(table: FeatureTable, prediction: Prediction, model_name: str, seed: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": model_name,
            "model_seed": seed,
            "scenario_id": table.scenario_id,
            "time_s": table.time_s,
            "true_event": table.event,
            "pred_event": prediction.event,
            "true_physical": table.physical,
            "physical_top3": ["|".join(row[:3]) for row in prediction.physical_top],
            "true_integrity": table.integrity,
            "integrity_top3": ["|".join(row[:3]) for row in prediction.integrity_top],
        }
    )


def _restricted_training_candidates(table: FeatureTable) -> dict[tuple[str, int], tuple[str, ...]]:
    mapping: dict[tuple[str, int], tuple[str, ...]] = {}
    for event_type in range(1, 9):
        event_mask = table.event == event_type
        physical = tuple(sorted(set(str(x) for x in table.physical[event_mask] if str(x))))
        integrity = tuple(sorted(set(str(x) for x in table.integrity[event_mask] if str(x))))
        if physical:
            mapping[("physical", event_type)] = physical
        if integrity:
            mapping[("integrity", event_type)] = integrity
    return mapping


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _aggregate_metric_rows(rows: pd.DataFrame) -> pd.DataFrame:
    identity = ["protocol", "model"]
    numeric = [
        column
        for column in rows.columns
        if column not in identity + ["model_seed"] and pd.api.types.is_numeric_dtype(rows[column])
    ]
    grouped = rows.groupby(identity, as_index=False)[numeric].agg(["mean", "std"])
    grouped.columns = [
        "_".join(part for part in column if part) if isinstance(column, tuple) else column
        for column in grouped.columns
    ]
    return grouped


def run_protocol(
    *,
    protocol_name: str,
    tables_by_scenario: Mapping[str, FeatureTable],
    assignments: Mapping[str, str],
    universes: CandidateUniverses,
    distance: ElectricalDistance,
    output_dir: Path,
    model_seeds: Sequence[int],
    tree_budget: int,
    topology_only: bool = False,
    save_models: bool = False,
) -> pd.DataFrame:
    split_tables: dict[str, FeatureTable] = {}
    for split in ("train", "validation", "test"):
        selected = [
            table for scenario_id, table in tables_by_scenario.items() if assignments[scenario_id] == split
        ]
        split_tables[split] = concatenate_tables(selected)
    train, validation, test = (
        split_tables["train"],
        split_tables["validation"],
        split_tables["test"],
    )

    protocol_dir = output_dir / protocol_name
    protocol_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for seed in model_seeds:
        if topology_only:
            training_candidates = _restricted_training_candidates(train)
            models: list[Any] = [
                TypedTopologyBaseline(
                    tree_budget,
                    seed,
                    universes,
                    distance,
                    training_candidates=training_candidates,
                )
            ]
        else:
            models = [
                FlatBaseline(tree_budget, seed),
                TypedBaseline(tree_budget, seed),
                TypedTopologyBaseline(tree_budget, seed, universes, distance),
            ]
        for model in models:
            print(f"[FIT] protocol={protocol_name} model={model.name} seed={seed}", flush=True)
            model.fit(train, validation)
            prediction = model.predict(test)
            metrics, detailed, confusion = evaluate_predictions(test, prediction, distance)
            duration_minutes = float(np.sum(test.dt_s) / 60.0)
            row = {
                "protocol": protocol_name,
                "model": model.name,
                "model_seed": int(seed),
                "detector_threshold": float(getattr(model, "detector_threshold", float("nan"))),
                "fit_seconds": float(model.fit_seconds),
                "predict_seconds": float(prediction.predict_seconds),
                "predict_seconds_per_input_minute": float(
                    prediction.predict_seconds / max(duration_minutes, 1e-12)
                ),
                **metrics,
                **model_complexity(model),
            }
            metric_rows.append(row)
            slug = f"{model.name}_seed{seed}"
            _json_dump(protocol_dir / f"event_report_{slug}.json", detailed)
            pd.DataFrame(
                confusion,
                index=[EVENT_NAMES[i] for i in range(9)],
                columns=[EVENT_NAMES[i] for i in range(9)],
            ).to_csv(protocol_dir / f"confusion_{slug}.csv", index_label="true_event")
            prediction_frames.append(prediction_frame(test, prediction, model.name, seed))
            if save_models:
                model_dir = protocol_dir / "models"
                model_dir.mkdir(exist_ok=True)
                joblib.dump(model, model_dir / f"{slug}.joblib", compress=3)

    rows = pd.DataFrame(metric_rows)
    rows.to_csv(protocol_dir / "metrics_by_seed.csv", index=False)
    _aggregate_metric_rows(rows).to_csv(protocol_dir / "metrics_summary.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        protocol_dir / "test_predictions.csv", index=False
    )
    return rows


def _scenario_manifest(
    records: Sequence[ScenarioRecord],
    assignments: Mapping[str, str],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scenario_id": record.spec.scenario_id,
                "scenario_seed": record.spec.seed,
                "split": assignments[record.spec.scenario_id],
                "target_key": record.target_key,
                "replicate": record.replicate,
                "event_type": record.spec.event_type,
                "physical_target_requested": record.spec.target,
                "integrity_target_requested": record.spec.cyber_target,
                "severity": record.spec.severity,
                "operating_scale": record.spec.operating_scale,
                "event_start_s": record.spec.event_start_s,
                "event_duration_s": record.spec.event_duration_s,
                "simulation_end_s": record.spec.simulation_end_s,
            }
            for record in records
        ]
    )


def _load_calibration(path: Path | None, output_dir: Path) -> MeasurementCalibration:
    if path is not None:
        calibration = MeasurementCalibration.from_json(path)
    else:
        cached = REPO_ROOT / "output" / "transient_state_estimation_benchmark" / "measurement_calibration.json"
        if cached.exists():
            calibration = MeasurementCalibration.from_json(cached)
        else:
            calibration = calibrate_from_event0(REPO_ROOT / "data" / "RAW0001")
    calibration.to_json(output_dir / "measurement_calibration.json")
    return calibration


def _synthetic_dataset() -> ScenarioDataset:
    time_axis = np.arange(0.0, 6.0, 0.1)
    pmus = np.asarray(DEFAULT_PMU_BUSES, dtype=int)
    buses = np.arange(1, 40, dtype=int)
    n_time, n_pmu = len(time_axis), len(pmus)
    observed_v = np.ones((n_time, n_pmu), dtype=float)
    observed_a = np.zeros((n_time, n_pmu), dtype=float)
    observed_f = np.full((n_time, n_pmu), 60.0, dtype=float)
    return ScenarioDataset(
        spec=ScenarioSpec("synthetic", 0, event_start_s=2.0, simulation_end_s=6.0, tstep_s=0.1),
        bus_ids=buses,
        pmu_buses=pmus,
        time_s=time_axis,
        voltage_pu=np.ones((n_time, len(buses))),
        angle_rad=np.zeros((n_time, len(buses))),
        frequency_hz=np.full((n_time, len(buses)), 60.0),
        rocof_hz_s=np.zeros((n_time, len(buses))),
        observed_voltage_pu=observed_v,
        observed_angle_rad=observed_a,
        observed_frequency_hz=observed_f,
        observed_mask=np.ones((n_time, n_pmu), dtype=bool),
        event_label=np.zeros(n_time, dtype=int),
        ybus=np.eye(len(buses), dtype=complex),
    )


def run_self_checks() -> dict[str, Any]:
    universes = load_candidate_universes()
    if len(universes.lines) != 34:
        raise AssertionError("Transmission-line candidate count changed")
    transformer_tokens = {
        "LINE2-30",
        "LINE6-31",
        "LINE10-32",
        "LINE11-12",
        "LINE12-13",
        "LINE19-20",
        "LINE19-33",
        "LINE20-34",
        "LINE22-35",
        "LINE23-36",
        "LINE25-37",
        "LINE29-38",
    }
    if transformer_tokens.intersection(universes.lines):
        raise AssertionError("A transformer entered the line-outage universe")
    records = build_scenario_records(universes, replicas_per_target=5, smoke=True)
    assignments = assign_primary_splits(records, split_seed=17)
    audit_primary_splits(records, assignments)

    original = _synthetic_dataset()
    changed = _synthetic_dataset()
    cutoff = 35
    changed.observed_voltage_pu[cutoff:, 0] = 7.0
    before, _, names = extract_causal_features(original, window_s=3.0)
    after, _, changed_names = extract_causal_features(changed, window_s=3.0)
    np.testing.assert_array_equal(before[:cutoff], after[:cutoff])
    if names != changed_names:
        raise AssertionError("Causal mutation changed the feature schema")
    return {
        "status": "passed",
        "transmission_lines": len(universes.lines),
        "transformers_excluded": 12,
        "smoke_scenarios": len(records),
        "causal_prefix_rows_verified": cutoff,
        "feature_count": len(names),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--window-s", type=float, default=3.0)
    parser.add_argument("--replicas-per-target", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument("--base-seed", type=int, default=202_608)
    parser.add_argument("--model-seeds", default="11,29,47")
    parser.add_argument("--tree-budget", type=int, default=120)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--stress-leave-target-out", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_check:
        result = run_self_checks()
        print(json.dumps(result, indent=2))
        return 0
    if not 1.0 <= args.window_s <= 5.0:
        raise SystemExit("--window-s must be between 1 and 5")
    if args.stress_leave_target_out and args.smoke:
        raise SystemExit("Leave-target-out stress requires the target-complete run, not --smoke")
    model_seeds = tuple(int(token.strip()) for token in args.model_seeds.split(",") if token.strip())
    if not model_seeds:
        raise SystemExit("At least one --model-seeds value is required")

    output_dir = args.output.resolve() / ("smoke" if args.smoke else "target_complete")
    output_dir.mkdir(parents=True, exist_ok=True)
    check = run_self_checks()
    _json_dump(output_dir / "self_check.json", check)
    universes = load_candidate_universes()
    records = build_scenario_records(
        universes,
        replicas_per_target=args.replicas_per_target,
        smoke=args.smoke,
        base_seed=args.base_seed,
    )
    assignments = assign_primary_splits(records, split_seed=args.split_seed)
    manifest = _scenario_manifest(records, assignments)
    manifest.to_csv(output_dir / "scenario_manifest_primary.csv", index=False)
    calibration = _load_calibration(args.calibration, output_dir)

    protocol = {
        "status": "engineering_smoke_not_paper_evidence" if args.smoke else "target_complete_primary",
        "causal_window_s": args.window_s,
        "sample_rate_hz": calibration.sample_rate_hz,
        "split": "within-target scenario-group 60/20/20",
        "split_seed": args.split_seed,
        "replicas_per_target": args.replicas_per_target,
        "model_seeds": list(model_seeds),
        "tree_budget_per_task": args.tree_budget,
        "threshold_selection": {
            "metric": "validation abnormal-class F1",
            "candidate_grid": [float(value) for value in np.linspace(0.1, 0.9, 33)],
            "flat_score": "1 - P(event=0) from the multiclass event model",
            "typed_score": "P(abnormal) from the binary detector",
        },
        "false_alarm_rate_denominator": "normal-labeled exposure",
        "runtime_scope": "model prediction on precomputed causal features",
        "calibration_label_use": "RAW0001 rows with Event == 0",
        "candidate_universes": universes.to_jsonable(),
        "scenario_count": len(records),
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "input_hashes": {
            "branches_physical.csv": _sha256(TOPOLOGY_DIR / "branches_physical.csv"),
            "zbus_effective_distance_full.csv": _sha256(
                TOPOLOGY_DIR / "zbus_effective_distance_full.csv"
            ),
            "measurement_calibration.json": _sha256(output_dir / "measurement_calibration.json"),
        },
        "event8_localization_note": (
            "The existing simulator creates generation, load, and integrity perturbations. "
            "The declared generator is the primary physical target; the additional random load "
            "remains in scenario metadata and is not scored as a single-label target."
        ),
    }
    _json_dump(output_dir / "protocol.json", protocol)

    cache_root = output_dir / "scenarios"
    tables: dict[str, FeatureTable] = {}
    quality_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for record in records:
        dataset = _load_or_simulate(record, calibration, cache_root, args.force)
        time_delta = np.diff(dataset.time_s)
        quality_rows.append(
            {
                "scenario_id": record.spec.scenario_id,
                "event_type": record.spec.event_type,
                "sample_count": len(dataset.time_s),
                "final_time_s": float(dataset.time_s[-1]),
                "requested_horizon_s": record.spec.simulation_end_s,
                "reached_horizon": bool(
                    dataset.time_s[-1] >= record.spec.simulation_end_s - 2.0 * record.spec.tstep_s
                ),
                "andes_returned_false_at_horizon": bool(
                    dataset.metadata.get("andes_returned_false_at_horizon", False)
                ),
                "median_dt_s": float(np.median(time_delta)),
                "min_dt_s": float(np.min(time_delta)),
                "max_dt_s": float(np.max(time_delta)),
                "finite_ground_truth_fraction": float(
                    np.mean(
                        np.isfinite(dataset.voltage_pu)
                        & np.isfinite(dataset.angle_rad)
                        & np.isfinite(dataset.frequency_hz)
                    )
                ),
                "observed_fraction": float(np.mean(dataset.observed_mask)),
                "abnormal_samples": int(np.sum(dataset.event_label != 0)),
            }
        )
        table = scenario_to_table(dataset, args.window_s)
        # Fail before training if any simulator target escaped the declared universe.
        physical_truth = {str(x) for x in table.physical if str(x)}
        integrity_truth = {str(x) for x in table.integrity if str(x)}
        expected_physical = set(universes.physical(record.spec.event_type))
        expected_integrity = set(universes.integrity(record.spec.event_type))
        if not physical_truth.issubset(expected_physical):
            raise AssertionError(
                f"{record.spec.scenario_id}: physical truth outside universe: "
                f"{physical_truth - expected_physical}"
            )
        if not integrity_truth.issubset(expected_integrity):
            raise AssertionError(
                f"{record.spec.scenario_id}: integrity truth outside universe: "
                f"{integrity_truth - expected_integrity}"
            )
        tables[record.spec.scenario_id] = table
    simulation_seconds = time.perf_counter() - started
    quality = pd.DataFrame(quality_rows)
    quality.to_csv(output_dir / "scenario_quality.csv", index=False)
    if not bool(quality["reached_horizon"].all()):
        failed = quality.loc[~quality["reached_horizon"], "scenario_id"].tolist()
        raise RuntimeError(f"Scenarios failed before the requested horizon: {failed[:5]}")
    if float(quality["finite_ground_truth_fraction"].min()) < 1.0:
        raise RuntimeError("At least one scenario has non-finite full-state ground truth")

    distance = ElectricalDistance()
    rows = [
        run_protocol(
            protocol_name="primary",
            tables_by_scenario=tables,
            assignments=assignments,
            universes=universes,
            distance=distance,
            output_dir=output_dir,
            model_seeds=model_seeds,
            tree_budget=args.tree_budget,
            save_models=args.save_models,
        )
    ]
    if args.stress_leave_target_out:
        stress = assign_leave_target_out(records, split_seed=args.split_seed)
        _scenario_manifest(records, stress).to_csv(
            output_dir / "scenario_manifest_leave_target_out.csv", index=False
        )
        rows.append(
            run_protocol(
                protocol_name="leave_target_out_stress",
                tables_by_scenario=tables,
                assignments=stress,
                universes=universes,
                distance=distance,
                output_dir=output_dir,
                model_seeds=model_seeds,
                tree_budget=args.tree_budget,
                topology_only=True,
                save_models=args.save_models,
            )
        )
    all_rows = pd.concat(rows, ignore_index=True)
    all_rows.to_csv(output_dir / "all_metrics_by_seed.csv", index=False)
    _aggregate_metric_rows(all_rows).to_csv(output_dir / "all_metrics_summary.csv", index=False)
    _json_dump(
        output_dir / "run_complete.json",
        {
            "status": "complete",
            "paper_evidence": not args.smoke,
            "scenario_count": len(records),
            "simulation_and_feature_seconds": simulation_seconds,
            "andes_false_return_at_completed_horizon_count": int(
                quality["andes_returned_false_at_horizon"].sum()
            ),
            "completed_unix_time": time.time(),
        },
    )
    print(f"Complete: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
