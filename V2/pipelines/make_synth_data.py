"""Generate synthetic IEEE 39 PMU training data for SGSMA 2026 V2.

The generator intentionally does not sample the hackathon raw PMU files. It
uses IEEE 39 metadata/topology plus randomized physics-informed perturbations
to create training scenarios with the same per-bus CSV schema as the provided
raw data. Each scenario also gets JSON metadata and review plots so an
electrical engineer can inspect whether the voltage/current/frequency response
looks credible before the data is used for ML.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import re
import shutil
import sys
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.plotters import (  # noqa: E402
    plot_current_phases,
    plot_frequency_rocof,
    plot_ieee39_diagram,
    plot_voltage_phases,
    write_engineering_report,
)


PMU_BUSES = [2, 5, 6, 10, 19, 22, 29, 39]
GENERATOR_BUSES = [2, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
LOAD_OR_NETWORK_BUSES = [bus for bus in range(1, 30)]

IEEE39_BRANCHES: list[tuple[int, int]] = [
    (1, 2),
    (1, 39),
    (2, 3),
    (2, 25),
    (2, 30),
    (3, 4),
    (3, 18),
    (4, 5),
    (4, 14),
    (5, 6),
    (5, 8),
    (6, 7),
    (6, 11),
    (6, 31),
    (7, 8),
    (8, 9),
    (9, 39),
    (10, 11),
    (10, 13),
    (10, 32),
    (11, 12),
    (12, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (16, 17),
    (16, 19),
    (16, 21),
    (16, 24),
    (17, 18),
    (17, 27),
    (19, 20),
    (19, 33),
    (19, 39),
    (20, 34),
    (21, 22),
    (22, 23),
    (22, 35),
    (23, 24),
    (23, 36),
    (25, 26),
    (25, 37),
    (26, 27),
    (26, 28),
    (26, 29),
    (28, 29),
    (29, 38),
]
BRANCH_REACTANCE_BY_EDGE: dict[tuple[int, int], float] = {}

EVENT_LABELS = {
    0: "normal",
    1: "fault",
    2: "line_outage",
    3: "generation_change",
    4: "load_change",
    5: "pmu_missing_data",
    6: "pmu_missing_data_plus_physical_event",
    7: "bad_data",
    8: "unknown_or_other",
}

LABEL_PRIORITY = {
    0: 0,
    8: 10,
    4: 20,
    3: 30,
    2: 40,
    1: 50,
    5: 60,
    6: 70,
    7: 80,
}

PHYSICAL_KINDS = {"fault", "line_outage", "generation_change", "load_change"}
PHASE_OFFSETS = {"A": 0.0, "B": -120.0, "C": 120.0}
MEASUREMENT_SUFFIXES = [
    "VA_ANG",
    "VA_MAG",
    "VB_ANG",
    "VB_MAG",
    "VC_ANG",
    "VC_MAG",
    "IA_ANG",
    "IA_MAG",
    "IB_ANG",
    "IB_MAG",
    "IC_ANG",
    "IC_MAG",
    "Freq",
    "ROCOF",
]


@dataclass(frozen=True)
class BusMetadata:
    """Static bus values used to synthesize nominal PMU channels."""

    bus: int
    base_kv: float
    vm_pu: float
    theta_deg: float
    base_current_a: float

    @property
    def line_neutral_voltage_v(self) -> float:
        return self.base_kv * 1000.0 * self.vm_pu / math.sqrt(3.0)


@dataclass
class EventSpec:
    """A synthetic event plus enough metadata for review and labeling."""

    event_id: str
    kind: str
    label: int
    start_sec: float
    end_sec: float
    nodes: list[int] = field(default_factory=list)
    line: list[int] | None = None
    pmu_bus: int | None = None
    severity: float = 0.0
    params: dict[str, Any] = field(default_factory=dict)
    affected_pmu_buses: list[int] = field(default_factory=list)
    affected_buses: list[int] = field(default_factory=list)
    weights_by_pmu_bus: dict[str, float] = field(default_factory=dict)
    engineering_note: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "pipelines" / "make_synth_data.json",
        help="Path to the synthetic data configuration JSON.",
    )
    parser.add_argument("--n-scenarios", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--duration-sec", type=float, default=None)
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG/markdown review artifacts.")
    parser.add_argument(
        "--plot-all",
        action="store_true",
        help="Write review plots/reports for every generated scenario.",
    )
    parser.add_argument(
        "--plot-first-n",
        type=int,
        default=None,
        help="Write review plots/reports for only the first N generated scenarios.",
    )
    parser.add_argument(
        "--plot-buses",
        default=None,
        help='Review buses to plot: "pmu", "all", or comma-separated bus numbers.',
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("allow_raw_data_sampling", False):
        raise ValueError(
            "Synthetic V2 data must not sample hackathon raw PMU files. "
            "Set allow_raw_data_sampling=false in make_synth_data.json."
        )
    return config


def resolve_project_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def wrap_degrees(values: np.ndarray | float) -> np.ndarray | float:
    return ((np.asarray(values) + 180.0) % 360.0) - 180.0


def bus_columns(bus: int) -> list[str]:
    prefix = f"BUS{bus}"
    return ["TIMESTAMP"] + [f"{prefix}_{suffix}" for suffix in MEASUREMENT_SUFFIXES] + [
        "DATA_PRESENT",
        "Event",
    ]


def measurement_columns(bus: int) -> list[str]:
    prefix = f"BUS{bus}"
    return [f"{prefix}_{suffix}" for suffix in MEASUREMENT_SUFFIXES]


def magnitude_columns(bus: int) -> list[str]:
    prefix = f"BUS{bus}"
    return [
        f"{prefix}_VA_MAG",
        f"{prefix}_VB_MAG",
        f"{prefix}_VC_MAG",
        f"{prefix}_IA_MAG",
        f"{prefix}_IB_MAG",
        f"{prefix}_IC_MAG",
    ]


def parse_pmu_metadata(metadata_dir: Path, config: dict[str, Any]) -> dict[int, BusMetadata]:
    """Parse bus base kV, per-unit voltage, and angle from PMUbus_ Location.txt."""

    metadata_file = metadata_dir / "PMUbus_ Location.txt"
    configured_currents = {int(k): float(v) for k, v in config.get("base_current_a", {}).items()}
    parsed: dict[int, tuple[float, float, float]] = {}
    if metadata_file.exists():
        for line in metadata_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip().startswith("BUS"):
                continue
            parts = line.split()
            match = re.search(r"BUS(\d+)", parts[0])
            if not match or len(parts) < 5:
                continue
            bus = int(match.group(1))
            try:
                base_kv = float(parts[1])
                vm_pu = float(parts[3])
                theta_deg = float(parts[4])
            except ValueError:
                continue
            parsed[bus] = (base_kv, vm_pu, theta_deg)

    metadata: dict[int, BusMetadata] = {}
    for bus in sorted(set(range(1, 40)).union(config.get("pmu_buses", PMU_BUSES))):
        base_kv, vm_pu, theta_deg = parsed.get(bus, (345.0, 1.0, 0.0))
        metadata[bus] = BusMetadata(
            bus=bus,
            base_kv=base_kv,
            vm_pu=vm_pu,
            theta_deg=theta_deg,
            base_current_a=configured_currents.get(bus, 500.0),
        )
    return metadata


def resolve_output_buses(config: dict[str, Any]) -> list[int]:
    """Return the bus set that should be exported as CSV/plots."""

    output_buses = config.get("output_buses", config.get("pmu_buses", PMU_BUSES))
    if isinstance(output_buses, str) and output_buses.lower() == "all":
        return list(range(1, 40))
    return sorted({int(bus) for bus in output_buses})


def parse_plot_bus_option(value: str, pmu_buses: list[int]) -> str | list[int]:
    """Parse a CLI plot-bus selector into the review config shape."""

    token = str(value).strip().lower()
    if token == "all":
        return "all"
    if token in {"pmu", "pmus"}:
        return [int(bus) for bus in pmu_buses]
    buses = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if not buses:
        raise ValueError("--plot-buses must be 'pmu', 'all', or comma-separated bus numbers.")
    invalid = [bus for bus in buses if bus < 1 or bus > 39]
    if invalid:
        raise ValueError(f"--plot-buses contains invalid IEEE-39 bus numbers: {invalid}")
    return sorted(set(buses))


def parse_raw_branch_model(metadata_dir: Path) -> tuple[list[tuple[int, int]], dict[tuple[int, int], float], str]:
    """Parse IEEE bus connectivity and branch reactance from the provided PSS/E RAW file."""

    raw_file = metadata_dir / "IEEE 39 Bus Power System.raw"
    if not raw_file.exists():
        return IEEE39_BRANCHES, {}, "hardcoded_ieee39_topology"

    lines = raw_file.read_text(encoding="utf-8", errors="ignore").splitlines()
    internal_to_bus: dict[int, int] = {}
    for line in lines[3:]:
        if line.strip().startswith("0 /end bus"):
            break
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        match = re.search(r"BUS(\d+)", parts[1])
        if not match:
            continue
        internal_to_bus[int(parts[0])] = int(match.group(1))

    try:
        branch_start = next(index for index, line in enumerate(lines) if "starting branch section" in line) + 1
    except StopIteration:
        return IEEE39_BRANCHES, {}, "hardcoded_ieee39_topology"

    branches: list[tuple[int, int]] = []
    reactance: dict[tuple[int, int], float] = {}
    for line in lines[branch_start:]:
        if line.strip().startswith("0"):
            break
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            left_internal = int(parts[0])
            right_internal = int(parts[1])
            x_pu = abs(float(parts[4]))
        except ValueError:
            continue
        if left_internal not in internal_to_bus or right_internal not in internal_to_bus:
            continue
        left = internal_to_bus[left_internal]
        right = internal_to_bus[right_internal]
        edge = tuple(sorted((left, right)))
        branches.append((left, right))
        reactance[edge] = max(x_pu, 1e-4)

    if len(branches) < 35:
        return IEEE39_BRANCHES, {}, "hardcoded_ieee39_topology"
    return branches, reactance, "psse_raw_branch_reactance"


def inspect_andes(metadata_dir: Path) -> dict[str, Any]:
    """Return ANDES availability without making dataset generation depend on it."""

    info: dict[str, Any] = {
        "available": False,
        "case_path": None,
        "engine_used": "ieee39_topology_surrogate",
        "note": (
            "The synthetic generator is ANDES-aware and records the IEEE 39 case "
            "when available, but uses a fast topology/swing-response surrogate by default."
        ),
    }
    try:
        import andes  # type: ignore

        andes_root = Path(andes.__file__).resolve().parent
        installed_case = andes_root / "cases" / "ieee39" / "ieee39_full.xlsx"
        metadata_case = metadata_dir / "IEEE 39 Bus Power System.raw"
        case_path = installed_case if installed_case.exists() else metadata_case
        info.update(
            {
                "available": True,
                "version": getattr(andes, "__version__", "unknown"),
                "case_path": str(case_path),
                "engine_used": "ieee39_topology_surrogate_with_andes_case_reference",
            }
        )
    except Exception as exc:  # pragma: no cover - optional dependency
        info["import_error"] = str(exc)
    return info


def build_neighbors(branches: list[tuple[int, int]]) -> dict[int, list[int]]:
    neighbors = {bus: [] for bus in range(1, 40)}
    for left, right in branches:
        neighbors[left].append(right)
        neighbors[right].append(left)
    return neighbors


def shortest_distances(source_nodes: list[int], branches: list[tuple[int, int]]) -> dict[int, int]:
    neighbors = build_neighbors(branches)
    distance = {bus: 10_000 for bus in range(1, 40)}
    queue: deque[int] = deque()
    for source in source_nodes:
        if source in distance:
            distance[source] = 0
            queue.append(source)
    while queue:
        bus = queue.popleft()
        for neighbor in neighbors[bus]:
            if distance[neighbor] > distance[bus] + 1:
                distance[neighbor] = distance[bus] + 1
                queue.append(neighbor)
    return distance


def reactance_distances(
    source_nodes: list[int],
    branches: list[tuple[int, int]],
    reactance_by_edge: dict[tuple[int, int], float],
) -> dict[int, float]:
    neighbors: dict[int, list[tuple[int, float]]] = {bus: [] for bus in range(1, 40)}
    for left, right in branches:
        x_pu = reactance_by_edge.get(tuple(sorted((left, right))), 1.0)
        neighbors[left].append((right, x_pu))
        neighbors[right].append((left, x_pu))

    distance = {bus: float("inf") for bus in range(1, 40)}
    heap: list[tuple[float, int]] = []
    for source in source_nodes:
        if source in distance:
            distance[source] = 0.0
            heapq.heappush(heap, (0.0, source))
    while heap:
        dist, bus = heapq.heappop(heap)
        if dist > distance[bus]:
            continue
        for neighbor, x_pu in neighbors[bus]:
            candidate = dist + x_pu
            if candidate < distance[neighbor]:
                distance[neighbor] = candidate
                heapq.heappush(heap, (candidate, neighbor))
    return distance


def electrical_weights(
    source_nodes: list[int],
    pmu_buses: list[int],
    config: dict[str, Any],
) -> dict[int, float]:
    tau = float(config.get("topology", {}).get("electrical_distance_tau", 2.6))
    if BRANCH_REACTANCE_BY_EDGE:
        distances_x = reactance_distances(source_nodes, IEEE39_BRANCHES, BRANCH_REACTANCE_BY_EDGE)
        median_x = float(np.median(list(BRANCH_REACTANCE_BY_EDGE.values())))
        distances = {
            bus: (distances_x[bus] / max(median_x, 1e-4)) if np.isfinite(distances_x[bus]) else 10_000
            for bus in range(1, 40)
        }
    else:
        distances = shortest_distances(source_nodes, IEEE39_BRANCHES)
    raw_weights = {
        bus: math.exp(-distances[bus] / max(tau, 1e-6)) if distances[bus] < 10_000 else 0.0 for bus in pmu_buses
    }
    max_weight = max(raw_weights.values()) if raw_weights else 1.0
    if max_weight <= 0:
        return {bus: 0.0 for bus in pmu_buses}
    return {bus: float(weight / max_weight) for bus, weight in raw_weights.items()}


def annotate_event(
    event: EventSpec,
    output_buses: list[int],
    pmu_buses: list[int],
    config: dict[str, Any],
) -> EventSpec:
    """Attach topology weights and affected-bus lists to an event."""

    bus_local_event = event.kind in {"pmu_dropout", "bad_data", "cyber_physical_overlap"} and event.pmu_bus
    if bus_local_event:
        affected = [int(event.pmu_bus)]
        weights = {bus: 1.0 if bus == int(event.pmu_bus) else 0.0 for bus in output_buses}
    else:
        if event.line:
            source_nodes = [int(event.line[0]), int(event.line[1])]
        elif event.nodes:
            source_nodes = [int(node) for node in event.nodes]
        elif event.pmu_bus:
            source_nodes = [int(event.pmu_bus)]
        else:
            source_nodes = output_buses
        weights = electrical_weights(source_nodes, output_buses, config)
        threshold = float(config.get("topology", {}).get("minimum_observable_weight", 0.035))
        affected = [bus for bus, weight in sorted(weights.items(), key=lambda item: -item[1]) if weight >= threshold]
    event.affected_buses = affected or output_buses[:]
    event.affected_pmu_buses = [bus for bus in event.affected_buses if bus in set(pmu_buses)]
    event.weights_by_pmu_bus = {str(bus): round(weight, 4) for bus, weight in sorted(weights.items())}
    return event


def choose_weighted(mapping: dict[str, float], rng: np.random.Generator) -> str:
    items = list(mapping.keys())
    weights = np.array([float(mapping[item]) for item in items], dtype=float)
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError(f"Invalid event weights: {mapping}")
    return str(rng.choice(items, p=weights / weights.sum()))


def uniform_range(values: list[float], rng: np.random.Generator) -> float:
    if len(values) != 2:
        raise ValueError(f"Expected [low, high] range, got {values}")
    return float(rng.uniform(float(values[0]), float(values[1])))


def event_time(
    kind: str,
    config: dict[str, Any],
    rng: np.random.Generator,
    duration_override: float | None = None,
    start_override: float | None = None,
) -> tuple[float, float]:
    total_duration = float(config["duration_sec"])
    min_start = float(config.get("pre_event_min_sec", 1.0))
    if duration_override is None:
        duration = uniform_range(config["event_types"][kind]["duration_sec"], rng)
    else:
        duration = float(duration_override)
    duration = min(duration, max(0.1, total_duration - min_start - 0.1))
    if start_override is None:
        latest_start = max(min_start, total_duration - duration - 0.15)
        start = min_start if latest_start <= min_start else float(rng.uniform(min_start, latest_start))
    else:
        start = float(start_override)
    end = min(total_duration, start + duration)
    return round(start, 4), round(end, 4)


def sample_single_event(
    kind: str,
    config: dict[str, Any],
    rng: np.random.Generator,
    pmu_buses: list[int],
    start_override: float | None = None,
    duration_override: float | None = None,
) -> EventSpec:
    event_cfg = config["event_types"][kind]
    start, end = event_time(kind, config, rng, duration_override, start_override)
    severity = uniform_range(event_cfg.get("severity", [0.2, 0.7]), rng)

    if kind == "fault":
        node = int(rng.integers(1, 40))
        mode = str(rng.choice(event_cfg["fault_modes"]))
        return EventSpec(
            event_id="",
            kind="fault",
            label=1,
            start_sec=start,
            end_sec=end,
            nodes=[node],
            severity=severity,
            params={"fault_mode": mode},
            engineering_note=(
                f"{mode} short-circuit at bus {node}; voltage sag and current surge decay "
                "with electrical distance from the faulted bus."
            ),
        )

    if kind == "line_outage":
        line = list(IEEE39_BRANCHES[int(rng.integers(0, len(IEEE39_BRANCHES)))])
        return EventSpec(
            event_id="",
            kind="line_outage",
            label=2,
            start_sec=start,
            end_sec=end,
            line=line,
            severity=severity,
            params={"operation": "open_line"},
            engineering_note=(
                f"Line {line[0]}-{line[1]} is opened; nearby buses show voltage angle step "
                "and current redistribution."
            ),
        )

    if kind == "generation_change":
        node = int(rng.choice(GENERATOR_BUSES))
        mode = str(rng.choice(event_cfg["modes"]))
        return EventSpec(
            event_id="",
            kind="generation_change",
            label=3,
            start_sec=start,
            end_sec=end,
            nodes=[node],
            severity=severity,
            params={"mode": mode},
            engineering_note=(
                f"Generator bus {node} experiences {mode}; the surrogate applies a damped "
                "swing-like frequency transient and voltage setpoint change."
            ),
        )

    if kind == "load_change":
        node = int(rng.choice(LOAD_OR_NETWORK_BUSES))
        mode = str(rng.choice(event_cfg["modes"]))
        return EventSpec(
            event_id="",
            kind="load_change",
            label=4,
            start_sec=start,
            end_sec=end,
            nodes=[node],
            severity=severity,
            params={"mode": mode},
            engineering_note=(
                f"Load at bus {node} has a {mode}; current demand changes and voltages "
                "move according to electrical distance."
            ),
        )

    if kind == "pmu_dropout":
        pmu_bus = int(rng.choice(pmu_buses))
        return EventSpec(
            event_id="",
            kind="pmu_dropout",
            label=5,
            start_sec=start,
            end_sec=end,
            pmu_bus=pmu_bus,
            severity=1.0,
            params={"missing_channels": "all"},
            engineering_note=f"All PMU channels at bus {pmu_bus} are missing; DATA_PRESENT=0.",
        )

    if kind == "bad_data":
        pmu_bus = int(rng.choice(pmu_buses))
        mode = str(rng.choice(event_cfg["modes"]))
        return EventSpec(
            event_id="",
            kind="bad_data",
            label=7,
            start_sec=start,
            end_sec=end,
            pmu_bus=pmu_bus,
            severity=severity,
            params={"mode": mode},
            engineering_note=f"Telemetry at bus {pmu_bus} is corrupted by synthetic {mode}.",
        )

    raise ValueError(f"Unsupported event kind: {kind}")


def sample_cyber_physical(
    config: dict[str, Any],
    rng: np.random.Generator,
    pmu_buses: list[int],
    physical_kind_override: str | None = None,
) -> list[EventSpec]:
    dropout = sample_single_event("pmu_dropout", config, rng, pmu_buses)
    duration = dropout.end_sec - dropout.start_sec
    physical_kind = physical_kind_override or str(rng.choice(config["event_types"]["cyber_physical"]["physical_modes"]))
    start = dropout.start_sec + float(rng.uniform(0.15, max(0.2, duration * 0.65)))
    physical_duration = min(
        float(config["duration_sec"]) - start - 0.05,
        uniform_range(config["event_types"][physical_kind]["duration_sec"], rng),
    )
    physical = sample_single_event(
        physical_kind,
        config,
        rng,
        pmu_buses,
        start_override=start,
        duration_override=max(0.12, physical_duration),
    )
    overlap_start = max(dropout.start_sec, physical.start_sec)
    overlap_end = min(dropout.end_sec, physical.end_sec)
    overlap = EventSpec(
        event_id="",
        kind="cyber_physical_overlap",
        label=6,
        start_sec=round(overlap_start, 4),
        end_sec=round(overlap_end, 4),
        nodes=physical.nodes,
        line=physical.line,
        pmu_bus=dropout.pmu_bus,
        severity=max(dropout.severity, physical.severity),
        params={"physical_kind": physical_kind},
        engineering_note=(
            "Label 6 is assigned only on the missing PMU bus while the telemetry "
            "dropout overlaps a physical event."
        ),
    )
    if overlap.end_sec <= overlap.start_sec:
        return [dropout, physical]
    return [dropout, physical, overlap]


def sample_multi_event(
    config: dict[str, Any],
    rng: np.random.Generator,
    pmu_buses: list[int],
) -> list[EventSpec]:
    max_events = int(config["event_types"]["multi_event"].get("max_events", 3))
    count = int(rng.integers(2, max_events + 1))
    # Use two longer physical events first so the label-8 ambiguous interval is
    # guaranteed enough samples for training. Short faults remain covered by the
    # dedicated label-1 sampler.
    physical_choices = ["line_outage", "generation_change", "load_change"]
    optional_choices = ["pmu_dropout", "bad_data"]
    selected = list(rng.choice(physical_choices, size=2, replace=False))
    if count > 2:
        selected.extend(list(rng.choice(optional_choices, size=count - 2, replace=False)))
    base_start = float(rng.uniform(float(config.get("pre_event_min_sec", 1.0)), float(config["duration_sec"]) - 3.0))
    events: list[EventSpec] = []
    for index, kind in enumerate(selected):
        jitter = (0.18 * index) if index < 2 else float(rng.uniform(-0.25, 0.65)) + 0.1 * index
        start = max(float(config.get("pre_event_min_sec", 1.0)), base_start + jitter)
        duration = uniform_range(config["event_types"][kind]["duration_sec"], rng)
        if index < 2:
            duration = max(duration, 2.0)
        events.append(sample_single_event(kind, config, rng, pmu_buses, start, duration))
    return events


def add_ambiguous_overlap_annotations(events: list[EventSpec], config: dict[str, Any]) -> list[EventSpec]:
    annotations: list[EventSpec] = []
    physical = [event for event in events if event.kind in PHYSICAL_KINDS]
    min_overlap = 1.0 / float(config["fps"])
    for left_index, left in enumerate(physical):
        for right in physical[left_index + 1 :]:
            start = max(left.start_sec, right.start_sec)
            end = min(left.end_sec, right.end_sec)
            if end - start < min_overlap:
                continue
            nodes = sorted({*left.nodes, *right.nodes})
            if left.line:
                nodes.extend(int(node) for node in left.line)
            if right.line:
                nodes.extend(int(node) for node in right.line)
            annotations.append(
                EventSpec(
                    event_id="",
                    kind="ambiguous_overlap",
                    label=8,
                    start_sec=round(start, 4),
                    end_sec=round(end, 4),
                    nodes=sorted(set(nodes)),
                    severity=max(left.severity, right.severity),
                    params={"overlap_kinds": [left.kind, right.kind]},
                    engineering_note=(
                        "Two physical signatures overlap; per the guide, this interval is "
                        "labeled 8 when it does not cleanly match a single event class."
                    ),
                )
            )
    return events + annotations[:3]


def forced_kind_from_spec(forced_spec: str | dict[str, Any] | None) -> str | None:
    if forced_spec is None:
        return None
    if isinstance(forced_spec, str):
        return forced_spec
    return str(forced_spec["kind"])


def apply_forced_options(events: list[EventSpec], forced_spec: str | dict[str, Any] | None) -> list[EventSpec]:
    if not isinstance(forced_spec, dict):
        return events

    for event in events:
        if event.kind == "pmu_dropout" and "pmu_bus" in forced_spec:
            event.pmu_bus = int(forced_spec["pmu_bus"])
            event.engineering_note = f"All PMU channels at bus {event.pmu_bus} are missing; DATA_PRESENT=0."
        if event.kind == "bad_data" and "pmu_bus" in forced_spec:
            event.pmu_bus = int(forced_spec["pmu_bus"])
            event.engineering_note = f"Telemetry at bus {event.pmu_bus} is corrupted by synthetic {event.params.get('mode')}."
        if event.kind == "cyber_physical_overlap" and "pmu_bus" in forced_spec:
            event.pmu_bus = int(forced_spec["pmu_bus"])
        if event.kind == "line_outage" and "line" in forced_spec:
            event.line = [int(forced_spec["line"][0]), int(forced_spec["line"][1])]
            event.nodes = []
            event.engineering_note = (
                f"Line {event.line[0]}-{event.line[1]} is opened; nearby buses show voltage angle step "
                "and current redistribution."
            )
        if event.kind in {"fault", "generation_change", "load_change"}:
            nodes = forced_spec.get(f"{event.kind}_nodes", forced_spec.get("physical_nodes", forced_spec.get("nodes")))
            if nodes:
                event.nodes = [int(node) for node in nodes]
        if event.kind == "fault" and "fault_mode" in forced_spec:
            event.params["fault_mode"] = str(forced_spec["fault_mode"])
        if event.kind in {"generation_change", "load_change", "bad_data"} and "mode" in forced_spec:
            event.params["mode"] = str(forced_spec["mode"])

    return events


def sample_scenario_events(
    scenario_id: str,
    config: dict[str, Any],
    rng: np.random.Generator,
    pmu_buses: list[int],
    output_buses: list[int],
    forced_spec: str | dict[str, Any] | None = None,
) -> list[EventSpec]:
    kind = forced_kind_from_spec(forced_spec) or choose_weighted(config["event_mix"], rng)
    if kind == "cyber_physical":
        physical_override = forced_spec.get("physical_kind") if isinstance(forced_spec, dict) else None
        events = sample_cyber_physical(config, rng, pmu_buses, physical_kind_override=physical_override)
    elif kind == "multi_event":
        events = sample_multi_event(config, rng, pmu_buses)
    else:
        events = [sample_single_event(kind, config, rng, pmu_buses)]
    events = apply_forced_options(events, forced_spec)
    events = add_ambiguous_overlap_annotations(events, config)

    annotated: list[EventSpec] = []
    for index, event in enumerate(events, start=1):
        event.event_id = f"{scenario_id}_E{index:02d}"
        annotated.append(annotate_event(event, output_buses, pmu_buses, config))
    return annotated


def make_time_grid(config: dict[str, Any]) -> np.ndarray:
    fps = float(config["fps"])
    n_samples = int(round(float(config["duration_sec"]) * fps)) + 1
    return np.round(np.arange(n_samples, dtype=float) / fps, 3)


def create_base_frames(
    metadata: dict[int, BusMetadata],
    pmu_buses: list[int],
    t: np.ndarray,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> dict[int, pd.DataFrame]:
    noise = config["noise"]
    nominal_frequency = float(config.get("nominal_frequency_hz", 60.0))
    common_freq = nominal_frequency + rng.normal(0.0, float(noise["frequency_std_hz"]), len(t))
    common_freq += 0.0015 * np.sin(2.0 * np.pi * 0.17 * t + rng.uniform(0.0, 2.0 * np.pi))
    scenario_angle_offset = float(rng.uniform(-4.0, 4.0))
    frames: dict[int, pd.DataFrame] = {}

    for bus in pmu_buses:
        meta = metadata[bus]
        prefix = f"BUS{bus}"
        slow_voltage = 0.0008 * np.sin(2.0 * np.pi * 0.09 * t + rng.uniform(0.0, 2.0 * np.pi))
        slow_current = 0.0030 * np.sin(2.0 * np.pi * 0.07 * t + rng.uniform(0.0, 2.0 * np.pi))
        base_angle = meta.theta_deg - 60.0 + scenario_angle_offset
        angle_wobble = 0.02 * np.sin(2.0 * np.pi * 0.13 * t + rng.uniform(0.0, 2.0 * np.pi))

        df = pd.DataFrame({"TIMESTAMP": t})
        for phase, offset in PHASE_OFFSETS.items():
            v_mag_noise = rng.normal(0.0, float(noise["voltage_mag_std_frac"]), len(t))
            i_mag_noise = rng.normal(0.0, float(noise["current_mag_std_frac"]), len(t))
            angle_noise = rng.normal(0.0, float(noise["angle_std_deg"]), len(t))
            current_angle_noise = rng.normal(0.0, float(noise["angle_std_deg"]) * 1.8, len(t))
            v_angle = wrap_degrees(base_angle + offset + angle_wobble + angle_noise)
            i_angle = wrap_degrees(v_angle + 158.0 + current_angle_noise)
            phase_balance = 1.0 + rng.normal(0.0, 0.0008)
            df[f"{prefix}_V{phase}_ANG"] = v_angle
            df[f"{prefix}_V{phase}_MAG"] = meta.line_neutral_voltage_v * phase_balance * (
                1.0 + slow_voltage + v_mag_noise
            )
            df[f"{prefix}_I{phase}_ANG"] = i_angle
            df[f"{prefix}_I{phase}_MAG"] = meta.base_current_a * (1.0 + slow_current + i_mag_noise)

        df[f"{prefix}_Freq"] = common_freq + rng.normal(0.0, float(noise["frequency_std_hz"]) * 0.25, len(t))
        df[f"{prefix}_ROCOF"] = 0.0
        df["DATA_PRESENT"] = 1
        df["Event"] = 0
        frames[bus] = df[bus_columns(bus)]
    return frames


def smooth_step(rel_time: np.ndarray, ramp_sec: float) -> np.ndarray:
    x = np.clip(rel_time / max(ramp_sec, 1e-6), 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * x)


def event_mask(t: np.ndarray, event: EventSpec) -> np.ndarray:
    return (t >= event.start_sec) & (t <= event.end_sec)


def phase_fault_factors(mode: str, phase: str) -> tuple[float, float]:
    if mode == "three_phase":
        return 1.0, 1.0
    if mode.endswith(phase):
        return 1.0, 1.15
    if mode.startswith("line_line") and phase in mode[-2:]:
        return 0.85, 1.0
    return 0.22, 0.28


def apply_frequency_delta(df: pd.DataFrame, bus: int, mask: np.ndarray, delta: np.ndarray) -> None:
    col = f"BUS{bus}_Freq"
    df.loc[mask, col] = df.loc[mask, col].to_numpy(dtype=float) + delta


def apply_fault(frames: dict[int, pd.DataFrame], event: EventSpec, t: np.ndarray) -> None:
    mode = str(event.params.get("fault_mode", "three_phase"))
    mask = event_mask(t, event)
    if not np.any(mask):
        return
    rel = t[mask] - event.start_sec
    duration = max(event.end_sec - event.start_sec, 1e-3)
    envelope = 0.65 + 0.35 * np.exp(-rel / max(duration / 2.5, 0.04))
    ring = np.sin(2.0 * np.pi * 7.5 * rel) * np.exp(-rel / 0.35)

    for bus, df in frames.items():
        weight = float(event.weights_by_pmu_bus.get(str(bus), 0.0))
        if weight <= 0:
            continue
        prefix = f"BUS{bus}"
        for phase in ("A", "B", "C"):
            v_factor, i_factor = phase_fault_factors(mode, phase)
            sag = np.clip(event.severity * weight * v_factor * envelope, 0.0, 0.92)
            surge = 1.0 + 4.2 * event.severity * weight * i_factor * envelope
            v_mag = f"{prefix}_V{phase}_MAG"
            i_mag = f"{prefix}_I{phase}_MAG"
            v_ang = f"{prefix}_V{phase}_ANG"
            i_ang = f"{prefix}_I{phase}_ANG"
            df.loc[mask, v_mag] = df.loc[mask, v_mag].to_numpy(dtype=float) * (1.0 - sag)
            df.loc[mask, i_mag] = df.loc[mask, i_mag].to_numpy(dtype=float) * surge
            df.loc[mask, v_ang] = wrap_degrees(
                df.loc[mask, v_ang].to_numpy(dtype=float) - 8.0 * event.severity * weight * ring
            )
            df.loc[mask, i_ang] = wrap_degrees(
                df.loc[mask, i_ang].to_numpy(dtype=float) + 16.0 * event.severity * weight * ring
            )
        freq_delta = -0.045 * event.severity * (0.45 + 0.55 * weight) * envelope
        apply_frequency_delta(df, bus, mask, freq_delta)


def apply_line_outage(frames: dict[int, pd.DataFrame], event: EventSpec, t: np.ndarray) -> None:
    mask = event_mask(t, event)
    if not np.any(mask) or not event.line:
        return
    rel = t[mask] - event.start_sec
    step = smooth_step(rel, 0.45)
    oscillation = np.sin(2.0 * np.pi * 1.8 * rel) * np.exp(-rel / 2.5)
    midpoint = 0.5 * (event.line[0] + event.line[1])

    for bus, df in frames.items():
        weight = float(event.weights_by_pmu_bus.get(str(bus), 0.0))
        side = -1.0 if bus <= midpoint else 1.0
        prefix = f"BUS{bus}"
        voltage_scale = 1.0 + side * 0.035 * event.severity * weight * step
        voltage_scale += 0.010 * event.severity * weight * oscillation
        current_scale = 1.0 - side * 0.24 * event.severity * weight * step
        current_scale += 0.030 * event.severity * weight * oscillation
        angle_delta = side * 3.2 * event.severity * weight * step + 0.8 * event.severity * weight * oscillation
        for phase in ("A", "B", "C"):
            df.loc[mask, f"{prefix}_V{phase}_MAG"] *= voltage_scale
            df.loc[mask, f"{prefix}_I{phase}_MAG"] *= current_scale
            df.loc[mask, f"{prefix}_V{phase}_ANG"] = wrap_degrees(
                df.loc[mask, f"{prefix}_V{phase}_ANG"].to_numpy(dtype=float) + angle_delta
            )
            df.loc[mask, f"{prefix}_I{phase}_ANG"] = wrap_degrees(
                df.loc[mask, f"{prefix}_I{phase}_ANG"].to_numpy(dtype=float) + angle_delta * 0.75
            )
        apply_frequency_delta(df, bus, mask, -0.010 * event.severity * weight * oscillation)


def apply_generation_change(frames: dict[int, pd.DataFrame], event: EventSpec, t: np.ndarray) -> None:
    mask = event_mask(t, event)
    if not np.any(mask):
        return
    mode = str(event.params.get("mode", "trip"))
    sign = 1.0 if mode == "ramp_up" else -1.0
    rel = t[mask] - event.start_sec
    step = smooth_step(rel, 0.8)
    swing = np.exp(-rel / 2.2) * np.sin(2.0 * np.pi * 0.85 * rel)

    for bus, df in frames.items():
        weight = float(event.weights_by_pmu_bus.get(str(bus), 0.0))
        global_weight = 0.55 + 0.45 * weight
        prefix = f"BUS{bus}"
        v_scale = 1.0 + sign * 0.018 * event.severity * weight * step
        i_scale = 1.0 - sign * 0.11 * event.severity * weight * step
        angle_delta = sign * (1.2 * event.severity * weight * step + 0.35 * event.severity * swing)
        for phase in ("A", "B", "C"):
            df.loc[mask, f"{prefix}_V{phase}_MAG"] *= v_scale
            df.loc[mask, f"{prefix}_I{phase}_MAG"] *= i_scale
            df.loc[mask, f"{prefix}_V{phase}_ANG"] = wrap_degrees(
                df.loc[mask, f"{prefix}_V{phase}_ANG"].to_numpy(dtype=float) + angle_delta
            )
            df.loc[mask, f"{prefix}_I{phase}_ANG"] = wrap_degrees(
                df.loc[mask, f"{prefix}_I{phase}_ANG"].to_numpy(dtype=float) + angle_delta
            )
        freq_delta = sign * event.severity * global_weight * (0.14 * np.exp(-rel / 2.8) + 0.015 * swing)
        apply_frequency_delta(df, bus, mask, freq_delta)


def apply_load_change(frames: dict[int, pd.DataFrame], event: EventSpec, t: np.ndarray) -> None:
    mask = event_mask(t, event)
    if not np.any(mask):
        return
    mode = str(event.params.get("mode", "load_increase"))
    sign = 1.0 if mode == "load_increase" else -1.0
    rel = t[mask] - event.start_sec
    step = smooth_step(rel, 0.55)
    small_ring = np.exp(-rel / 1.7) * np.sin(2.0 * np.pi * 1.2 * rel)

    for bus, df in frames.items():
        weight = float(event.weights_by_pmu_bus.get(str(bus), 0.0))
        prefix = f"BUS{bus}"
        v_scale = 1.0 - sign * 0.045 * event.severity * weight * step
        i_scale = 1.0 + sign * 0.32 * event.severity * weight * step
        angle_delta = -sign * 1.8 * event.severity * weight * step + 0.4 * event.severity * weight * small_ring
        for phase in ("A", "B", "C"):
            df.loc[mask, f"{prefix}_V{phase}_MAG"] *= v_scale
            df.loc[mask, f"{prefix}_I{phase}_MAG"] *= i_scale
            df.loc[mask, f"{prefix}_V{phase}_ANG"] = wrap_degrees(
                df.loc[mask, f"{prefix}_V{phase}_ANG"].to_numpy(dtype=float) + angle_delta
            )
            df.loc[mask, f"{prefix}_I{phase}_ANG"] = wrap_degrees(
                df.loc[mask, f"{prefix}_I{phase}_ANG"].to_numpy(dtype=float) + angle_delta * 0.8
            )
        freq_delta = -sign * event.severity * (0.5 + 0.5 * weight) * (0.045 * np.exp(-rel / 2.0))
        apply_frequency_delta(df, bus, mask, freq_delta)


def apply_pmu_dropout(frames: dict[int, pd.DataFrame], event: EventSpec, t: np.ndarray) -> None:
    if event.pmu_bus is None or event.pmu_bus not in frames:
        return
    mask = event_mask(t, event)
    df = frames[event.pmu_bus]
    df.loc[mask, measurement_columns(event.pmu_bus)] = np.nan
    df.loc[mask, "DATA_PRESENT"] = 0


def apply_bad_data(
    frames: dict[int, pd.DataFrame],
    event: EventSpec,
    t: np.ndarray,
    rng: np.random.Generator,
) -> None:
    if event.pmu_bus is None or event.pmu_bus not in frames:
        return
    mask = event_mask(t, event)
    if not np.any(mask):
        return
    bus = event.pmu_bus
    df = frames[bus]
    prefix = f"BUS{bus}"
    mode = str(event.params.get("mode", "voltage_scale"))
    rel = t[mask] - event.start_sec
    step = smooth_step(rel, 0.08)
    if mode == "voltage_scale":
        phase = str(rng.choice(["A", "B", "C"]))
        col = f"{prefix}_V{phase}_MAG"
        df.loc[mask, col] = df.loc[mask, col].to_numpy(dtype=float) * (1.0 + event.severity * step)
        event.params["channel"] = col
    elif mode == "current_spike":
        phase = str(rng.choice(["A", "B", "C"]))
        col = f"{prefix}_I{phase}_MAG"
        spike = 1.0 + event.severity * (1.0 + 2.0 * np.exp(-rel / 0.18))
        df.loc[mask, col] = df.loc[mask, col].to_numpy(dtype=float) * spike
        event.params["channel"] = col
    elif mode == "angle_bias":
        phase = str(rng.choice(["A", "B", "C"]))
        col = f"{prefix}_V{phase}_ANG"
        bias = event.severity * 25.0 * step
        df.loc[mask, col] = wrap_degrees(df.loc[mask, col].to_numpy(dtype=float) + bias)
        event.params["channel"] = col
    else:
        col = f"{prefix}_Freq"
        df.loc[mask, col] = df.loc[mask, col].to_numpy(dtype=float) + event.severity * 0.25 * step
        event.params["channel"] = col


def update_label_array(labels: np.ndarray, mask: np.ndarray, label: int) -> None:
    current = labels[mask]
    if current.size == 0:
        return
    current_priority = np.array([LABEL_PRIORITY[int(value)] for value in current], dtype=int)
    take = current_priority < LABEL_PRIORITY[label]
    if np.any(take):
        updated = current.copy()
        updated[take] = label
        labels[mask] = updated


def apply_events(
    frames: dict[int, pd.DataFrame],
    events: list[EventSpec],
    t: np.ndarray,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> dict[int, np.ndarray]:
    pmu_buses = sorted(frames)
    labels = {bus: np.zeros(len(t), dtype=int) for bus in pmu_buses}
    physical_mask = np.zeros(len(t), dtype=bool)
    physical_count = np.zeros(len(t), dtype=int)
    dropout_masks: list[tuple[int, np.ndarray]] = []

    for event in events:
        mask = event_mask(t, event)
        if event.kind == "fault":
            apply_fault(frames, event, t)
            physical_mask |= mask
            physical_count[mask] += 1
            for bus in pmu_buses:
                update_label_array(labels[bus], mask, event.label)
        elif event.kind == "line_outage":
            apply_line_outage(frames, event, t)
            physical_mask |= mask
            physical_count[mask] += 1
            for bus in pmu_buses:
                update_label_array(labels[bus], mask, event.label)
        elif event.kind == "generation_change":
            apply_generation_change(frames, event, t)
            physical_mask |= mask
            physical_count[mask] += 1
            for bus in pmu_buses:
                update_label_array(labels[bus], mask, event.label)
        elif event.kind == "load_change":
            apply_load_change(frames, event, t)
            physical_mask |= mask
            physical_count[mask] += 1
            for bus in pmu_buses:
                update_label_array(labels[bus], mask, event.label)
        elif event.kind == "pmu_dropout":
            apply_pmu_dropout(frames, event, t)
            if event.pmu_bus is not None and event.pmu_bus in labels:
                update_label_array(labels[event.pmu_bus], mask, event.label)
                dropout_masks.append((event.pmu_bus, mask))
        elif event.kind == "bad_data":
            apply_bad_data(frames, event, t, rng)
            if event.pmu_bus is not None and event.pmu_bus in labels:
                update_label_array(labels[event.pmu_bus], mask, event.label)
        elif event.kind in {"cyber_physical_overlap", "ambiguous_overlap"}:
            continue
        else:
            for bus in pmu_buses:
                update_label_array(labels[bus], mask, 8)

    ambiguous_physical = physical_count >= 2
    if np.any(ambiguous_physical):
        for bus in pmu_buses:
            current = labels[bus]
            replace = ambiguous_physical & np.isin(current, [1, 2, 3, 4])
            current[replace] = 8
            labels[bus] = current

    for bus, dropout_mask in dropout_masks:
        update_label_array(labels[bus], dropout_mask & physical_mask, 6)

    for bus, df in frames.items():
        df["Event"] = labels[bus]

    inject_background_nan(frames, config, rng)
    recompute_rocof(frames, config, rng)
    return labels


def inject_background_nan(
    frames: dict[int, pd.DataFrame],
    config: dict[str, Any],
    rng: np.random.Generator,
) -> None:
    probability = float(config.get("noise", {}).get("nan_probability_normal", 0.0))
    if probability <= 0:
        return
    for bus, df in frames.items():
        mask = rng.random(len(df)) < probability
        if not np.any(mask):
            continue
        df.loc[mask, measurement_columns(bus)] = np.nan
        df.loc[mask, "DATA_PRESENT"] = 0


def recompute_rocof(
    frames: dict[int, pd.DataFrame],
    config: dict[str, Any],
    rng: np.random.Generator,
) -> None:
    fps = float(config["fps"])
    noise_std = float(config.get("noise", {}).get("rocof_std_hz_per_s", 0.02))
    for bus, df in frames.items():
        prefix = f"BUS{bus}"
        freq = pd.Series(df[f"{prefix}_Freq"], dtype=float)
        filled = freq.interpolate(limit_direction="both").to_numpy(dtype=float)
        rocof = np.gradient(filled, 1.0 / fps) + rng.normal(0.0, noise_std, len(df))
        missing = df["DATA_PRESENT"].to_numpy(dtype=int) == 0
        rocof[missing] = np.nan
        df[f"{prefix}_ROCOF"] = rocof


def clip_physical_ranges(frames: dict[int, pd.DataFrame]) -> None:
    for bus, df in frames.items():
        for col in magnitude_columns(bus):
            df[col] = df[col].clip(lower=0.0)
        df["DATA_PRESENT"] = df["DATA_PRESENT"].fillna(0).astype(int)
        df["Event"] = df["Event"].fillna(0).astype(int)


def bus_output_filename(bus: int, pmu_buses: list[int], config: dict[str, Any]) -> str:
    style = str(config.get("filename_style", "pmu_prefix")).lower()
    if style == "raw":
        return f"Bus{bus}_Competition_Data_nanmask.csv"
    prefix = "PMU" if bus in set(pmu_buses) else "NON_PMU"
    return f"{prefix}_Bus{bus}_Competition_Data_nanmask.csv"


def write_bus_csvs(
    frames: dict[int, pd.DataFrame],
    scenario_dir: Path,
    pmu_buses: list[int],
    config: dict[str, Any],
) -> dict[int, str]:
    csv_dir = scenario_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    files: dict[int, str] = {}
    for bus, df in sorted(frames.items()):
        output = csv_dir / bus_output_filename(bus, pmu_buses, config)
        df[bus_columns(bus)].to_csv(output, index=False, float_format="%.6f")
        files[bus] = output.relative_to(scenario_dir).as_posix()
    return files


def prepare_scenario_dir(output_dir: Path, scenario_id: str) -> Path:
    """Create a clean scenario directory without touching paths outside output_dir."""

    output_root = output_dir.resolve()
    scenario_dir = (output_dir / scenario_id).resolve()
    if output_root != scenario_dir and output_root not in scenario_dir.parents:
        raise ValueError(f"Refusing to clean scenario outside output_dir: {scenario_dir}")
    if scenario_dir.exists():
        shutil.rmtree(scenario_dir)
    scenario_dir.mkdir(parents=True, exist_ok=True)
    return scenario_dir


def label_counts(frames: dict[int, pd.DataFrame]) -> dict[str, int]:
    counts = {str(label): 0 for label in EVENT_LABELS}
    for df in frames.values():
        values = df["Event"].value_counts()
        for label, count in values.items():
            counts[str(int(label))] = counts.get(str(int(label)), 0) + int(count)
    return counts


def validate_generated_dataset(output_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate generated CSV/plot artifacts and write a reviewer-friendly summary."""

    errors: list[str] = []
    allowed_labels = set(EVENT_LABELS)
    filename_style = str(manifest.get("config", {}).get("filename_style", "pmu_prefix")).lower()
    output_buses = [int(bus) for bus in manifest["output_buses"]]
    expected_csvs_per_scenario = len(output_buses)
    total_csv_files = 0
    total_png_files = 0
    total_reports = 0

    for scenario in manifest["scenarios"]:
        scenario_dir = output_dir / scenario["path"]
        csv_files = sorted((scenario_dir / "csv").glob("*.csv"))
        png_files = sorted((scenario_dir / "plots").glob("*.png"))
        report = scenario_dir / "engineering_review.md"
        total_csv_files += len(csv_files)
        total_png_files += len(png_files)
        total_reports += int(report.exists())
        if len(csv_files) != expected_csvs_per_scenario:
            errors.append(f"{scenario['scenario_id']}: expected {expected_csvs_per_scenario} CSVs, found {len(csv_files)}")
        for csv_file in csv_files:
            if csv_file.name.startswith("Bus") and filename_style != "raw":
                errors.append(f"{scenario['scenario_id']}: legacy filename remains: {csv_file.name}")
            match = re.search(r"Bus(\d+)_Competition", csv_file.name)
            if not match:
                errors.append(f"{scenario['scenario_id']}: cannot parse bus from {csv_file.name}")
                continue
            bus = int(match.group(1))
            if bus not in output_buses:
                errors.append(f"{scenario['scenario_id']}: unexpected bus {bus} in {csv_file.name}")
            try:
                df = pd.read_csv(csv_file)
            except Exception as exc:
                errors.append(f"{scenario['scenario_id']}: failed to read {csv_file.name}: {exc}")
                continue
            if list(df.columns) != bus_columns(bus):
                errors.append(f"{scenario['scenario_id']}: bad columns in {csv_file.name}")
            labels = set(pd.to_numeric(df["Event"], errors="coerce").dropna().astype(int).unique())
            if not labels <= allowed_labels:
                errors.append(f"{scenario['scenario_id']}: invalid labels in {csv_file.name}: {sorted(labels - allowed_labels)}")
            if df["TIMESTAMP"].isna().any():
                errors.append(f"{scenario['scenario_id']}: NaN timestamps in {csv_file.name}")
            data_present = pd.to_numeric(df["DATA_PRESENT"], errors="coerce")
            if data_present.isna().any() or not set(data_present.astype(int).unique()) <= {0, 1}:
                errors.append(f"{scenario['scenario_id']}: DATA_PRESENT must be binary in {csv_file.name}")
                continue
            events = pd.to_numeric(df["Event"], errors="coerce").fillna(-1).astype(int)
            present = data_present.astype(int)
            measurements = df[measurement_columns(bus)]
            measurement_has_nan = measurements.isna().any(axis=1)
            if ((present == 0) & (~events.isin([5, 6]))).any():
                errors.append(f"{scenario['scenario_id']}: DATA_PRESENT=0 outside labels 5/6 in {csv_file.name}")
            if (events.isin([5, 6]) & (present != 0)).any():
                errors.append(f"{scenario['scenario_id']}: labels 5/6 require DATA_PRESENT=0 in {csv_file.name}")
            if ((present == 1) & measurement_has_nan).any():
                errors.append(f"{scenario['scenario_id']}: DATA_PRESENT=1 rows contain NaN measurements in {csv_file.name}")

    summary = {
        "dataset_root": str(output_dir.resolve()),
        "scenario_dirs": len(manifest["scenarios"]),
        "output_buses_per_scenario": expected_csvs_per_scenario,
        "bus_csv_files": total_csv_files,
        "review_png_files": total_png_files,
        "engineering_review_markdown": total_reports,
        "aggregate_label_counts": manifest.get("aggregate_label_counts", {}),
        "event_kind_counts": manifest.get("event_kind_counts", {}),
        "branch_model_source": manifest.get("branch_model_source"),
        "andes": manifest.get("andes"),
        "filename_style": manifest.get("filename_style"),
        "errors": errors,
        "passed": not errors,
    }
    (output_dir / "validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def plot_and_report_scenario(
    scenario: dict[str, Any],
    frames: dict[int, pd.DataFrame],
    csv_files: dict[int, str],
    scenario_dir: Path,
    config: dict[str, Any],
) -> list[str]:
    review = config.get("review", {})
    plot_dir = scenario_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    events = scenario["events"]
    buses_option = review.get("plot_buses_per_scenario", review.get("plot_pmu_buses_per_scenario", "all"))
    if buses_option == "all":
        buses = sorted(frames)
    elif isinstance(buses_option, str) and buses_option.lower() in {"pmu", "pmus"}:
        buses = [int(bus) for bus in config.get("pmu_buses", PMU_BUSES) if int(bus) in frames]
    elif isinstance(buses_option, list):
        buses = [int(bus) for bus in buses_option if int(bus) in frames]
    else:
        buses = sorted(frames)[: int(buses_option)]

    plot_paths: list[Path] = []
    for bus in buses:
        plot_paths.append(plot_voltage_phases(frames[bus], bus, events, plot_dir))
        if bool(review.get("include_current_plots", True)):
            plot_paths.append(plot_current_phases(frames[bus], bus, events, plot_dir))
        if bool(review.get("include_frequency_plots", True)):
            plot_paths.append(plot_frequency_rocof(frames[bus], bus, events, plot_dir))
    branches = [
        (int(branch[0]), int(branch[1]))
        for branch in scenario.get("ieee39_branches", IEEE39_BRANCHES)
        if len(branch) >= 2
    ]
    plot_paths.append(
        plot_ieee39_diagram(
            branches,
            config.get("pmu_buses", PMU_BUSES),
            events,
            plot_dir,
            scenario_id=scenario.get("scenario_id"),
        )
    )
    report_path = write_engineering_report(
        scenario,
        csv_files,
        [path.relative_to(scenario_dir).as_posix() for path in plot_paths],
        scenario_dir,
    )
    plot_paths.append(report_path)
    return [path.relative_to(scenario_dir).as_posix() for path in plot_paths]


def scenario_to_jsonable(
    scenario_id: str,
    scenario_seed: int,
    config: dict[str, Any],
    events: list[EventSpec],
    andes_info: dict[str, Any],
    frames: dict[int, pd.DataFrame],
    pmu_buses: list[int],
) -> dict[str, Any]:
    return {
        "scenario_id": scenario_id,
        "seed": scenario_seed,
        "fps": int(config["fps"]),
        "duration_sec": float(config["duration_sec"]),
        "nominal_frequency_hz": float(config.get("nominal_frequency_hz", 60.0)),
        "pmu_buses": pmu_buses,
        "output_buses": sorted(frames),
        "event_label_map": {str(k): v for k, v in EVENT_LABELS.items()},
        "no_raw_data_sampling": True,
        "andes": andes_info,
        "ieee39_branches": [[left, right] for left, right in IEEE39_BRANCHES],
        "branch_reactance_pu": {
            f"{left}-{right}": BRANCH_REACTANCE_BY_EDGE.get(tuple(sorted((left, right))))
            for left, right in IEEE39_BRANCHES
        },
        "events": [asdict(event) for event in events],
        "label_counts": label_counts(frames),
        "units": {
            "TIMESTAMP": "seconds from scenario start",
            "V*_MAG": "line-neutral RMS volts",
            "V*_ANG": "electrical degrees",
            "I*_MAG": "amperes",
            "I*_ANG": "electrical degrees",
            "Freq": "Hz",
            "ROCOF": "Hz/s",
            "DATA_PRESENT": "1 if telemetry is present, 0 if missing",
            "Event": "integer class label matching SGSMA event taxonomy",
        },
    }


def generate_dataset(
    config: dict[str, Any],
    output_dir: Path,
    make_plots: bool,
) -> dict[str, Any]:
    global BRANCH_REACTANCE_BY_EDGE, IEEE39_BRANCHES

    seed = int(config["seed"])
    n_scenarios = int(config["n_scenarios"])
    pmu_buses = [int(bus) for bus in config.get("pmu_buses", PMU_BUSES)]
    output_buses = resolve_output_buses(config)
    metadata_dir = resolve_project_path(config["metadata_dir"])
    metadata = parse_pmu_metadata(metadata_dir, config)
    IEEE39_BRANCHES, BRANCH_REACTANCE_BY_EDGE, branch_model_source = parse_raw_branch_model(metadata_dir)
    andes_info = inspect_andes(metadata_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "dataset": "SGSMA26_V2_synthetic_training",
        "seed": seed,
        "n_scenarios": n_scenarios,
        "pmu_buses": pmu_buses,
        "output_buses": output_buses,
        "event_label_map": {str(k): v for k, v in EVENT_LABELS.items()},
        "no_raw_data_sampling": True,
        "config": config,
        "filename_style": config.get("filename_style", "pmu_prefix"),
        "andes": andes_info,
        "branch_model_source": branch_model_source,
        "branch_reactance_pu": {
            f"{left}-{right}": BRANCH_REACTANCE_BY_EDGE.get(tuple(sorted((left, right))))
            for left, right in IEEE39_BRANCHES
        },
        "scenarios": [],
    }

    plot_limit_setting = config.get("review", {}).get("plot_first_n_scenarios", 0)
    plot_limit = n_scenarios if str(plot_limit_setting).lower() == "all" else int(plot_limit_setting)
    coverage_first = list(config.get("coverage_first", []))
    for scenario_index in range(1, n_scenarios + 1):
        scenario_id = f"SIM_{scenario_index:04d}"
        scenario_seed = seed + scenario_index * 7919
        rng = np.random.default_rng(scenario_seed)
        scenario_dir = prepare_scenario_dir(output_dir, scenario_id)

        t = make_time_grid(config)
        frames = create_base_frames(metadata, output_buses, t, config, rng)
        forced_spec = coverage_first[scenario_index - 1] if scenario_index <= len(coverage_first) else None
        events = sample_scenario_events(
            scenario_id,
            config,
            rng,
            pmu_buses,
            output_buses,
            forced_spec=forced_spec,
        )
        apply_events(frames, events, t, config, rng)
        clip_physical_ranges(frames)

        csv_files = write_bus_csvs(frames, scenario_dir, pmu_buses, config)
        scenario = scenario_to_jsonable(scenario_id, scenario_seed, config, events, andes_info, frames, pmu_buses)
        scenario["csv_files"] = {str(bus): path for bus, path in csv_files.items()}

        if make_plots and scenario_index <= plot_limit:
            scenario["review_artifacts"] = plot_and_report_scenario(
                scenario,
                frames,
                csv_files,
                scenario_dir,
                config,
            )
        else:
            scenario["review_artifacts"] = []

        scenario_json = scenario_dir / f"{scenario_id}.json"
        scenario_json.write_text(json.dumps(scenario, indent=2), encoding="utf-8")
        manifest["scenarios"].append(
            {
                "scenario_id": scenario_id,
                "seed": scenario_seed,
                "path": scenario_dir.relative_to(output_dir).as_posix(),
                "json": scenario_json.relative_to(output_dir).as_posix(),
                "csv_files": scenario["csv_files"],
                "review_artifacts": scenario["review_artifacts"],
                "events": [
                    {
                        "kind": event.kind,
                        "label": event.label,
                        "start_sec": event.start_sec,
                        "end_sec": event.end_sec,
                        "nodes": event.nodes,
                        "line": event.line,
                        "pmu_bus": event.pmu_bus,
                    }
                    for event in events
                ],
                "label_counts": scenario["label_counts"],
            }
        )

        print(
            f"[{scenario_id}] wrote {len(csv_files)} bus CSVs, "
            f"{len(events)} event records, labels={scenario['label_counts']}"
        )

    aggregate_counts = {str(label): 0 for label in EVENT_LABELS}
    event_kind_counts: dict[str, int] = {}
    for scenario in manifest["scenarios"]:
        for label, count in scenario["label_counts"].items():
            aggregate_counts[str(label)] = aggregate_counts.get(str(label), 0) + int(count)
        for event in scenario["events"]:
            kind = str(event["kind"])
            event_kind_counts[kind] = event_kind_counts.get(kind, 0) + 1
    manifest["aggregate_label_counts"] = aggregate_counts
    manifest["event_kind_counts"] = dict(sorted(event_kind_counts.items()))
    manifest["total_bus_rows"] = int(sum(aggregate_counts.values()))
    manifest["artifact_counts"] = {
        "scenario_dirs": n_scenarios,
        "bus_csv_files": n_scenarios * len(output_buses),
        "review_png_files": sum(
            1
            for scenario in manifest["scenarios"]
            for artifact in scenario.get("review_artifacts", [])
            if str(artifact).endswith(".png")
        ),
        "engineering_review_markdown": sum(
            1
            for scenario in manifest["scenarios"]
            for artifact in scenario.get("review_artifacts", [])
            if str(artifact).endswith(".md")
        ),
    }

    manifest_path = output_dir / "synthetic_dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    validation = validate_generated_dataset(output_dir, manifest)
    if not validation["passed"]:
        raise RuntimeError(
            "Synthetic dataset validation failed. See "
            f"{output_dir / 'validation_summary.json'} for details."
        )
    return manifest


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = json.loads(json.dumps(config))
    if args.n_scenarios is not None:
        config["n_scenarios"] = int(args.n_scenarios)
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    if args.seed is not None:
        config["seed"] = int(args.seed)
    if args.duration_sec is not None:
        config["duration_sec"] = float(args.duration_sec)
    review = config.setdefault("review", {})
    if getattr(args, "plot_all", False):
        review["plot_first_n_scenarios"] = "all"
    elif getattr(args, "plot_first_n", None) is not None:
        review["plot_first_n_scenarios"] = int(args.plot_first_n)
    if getattr(args, "plot_buses", None) is not None:
        review["plot_buses_per_scenario"] = parse_plot_bus_option(
            args.plot_buses,
            [int(bus) for bus in config.get("pmu_buses", PMU_BUSES)],
        )
    return config


def main() -> None:
    args = parse_args()
    config = apply_cli_overrides(load_config(args.config), args)
    output_dir = resolve_project_path(config["output_dir"])
    manifest = generate_dataset(config, output_dir, make_plots=not args.no_plots)
    manifest_path = output_dir / "synthetic_dataset_manifest.json"
    print(f"Finished synthetic dataset: {manifest['n_scenarios']} scenarios at {manifest_path}")


if __name__ == "__main__":
    main()
