from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import pandas as pd

from src.detectors.shared.preprocessing.pmu_alignment import align_scenario_pmu_dir


_BUS_FILE_PATTERN = re.compile(r"^Bus\d+_Competition_Data_(?:nanmask|sim|.*)\.csv$", re.IGNORECASE)


@dataclass(slots=True)
class RawScenarioRef:
    scenario_id: str
    scenario_dir: Path
    pmu_dir: Path


def event_to_binary(event: int | float | str | None) -> int:
    if event is None:
        return 0
    try:
        value = int(float(event))
    except (TypeError, ValueError):
        return 0
    return int(value != 0)


def event_family_from_event(event: int | float | str | None) -> str:
    try:
        value = int(float(event))
    except (TypeError, ValueError):
        return "normal"
    if value == 0:
        return "normal"
    if value in {1, 2, 3, 4}:
        return "physical_heavy"
    if value in {5, 7}:
        return "cyber_heavy"
    if value in {6, 8}:
        return "concurrent_heavy"
    return "unknown"


def is_raw_pmu_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    if any(_BUS_FILE_PATTERN.match(child.name) for child in path.iterdir() if child.is_file()):
        return True
    pmu_dir = path / "pmu"
    return pmu_dir.exists() and pmu_dir.is_dir() and any(_BUS_FILE_PATTERN.match(child.name) for child in pmu_dir.iterdir() if child.is_file())


def discover_raw_scenario_dirs(raw_input_root: Path) -> list[RawScenarioRef]:
    root = raw_input_root.resolve()
    if is_raw_pmu_dir(root):
        return [RawScenarioRef(scenario_id=root.name, scenario_dir=root, pmu_dir=root / "pmu" if (root / "pmu").exists() else root)]

    refs: list[RawScenarioRef] = []
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        if is_raw_pmu_dir(child):
            refs.append(
                RawScenarioRef(
                    scenario_id=child.name,
                    scenario_dir=child,
                    pmu_dir=child / "pmu" if (child / "pmu").exists() else child,
                )
            )
    return refs


def load_raw_scenario_frame(scenario_dir: Path) -> pd.DataFrame:
    pmu_dir = scenario_dir / "pmu" if (scenario_dir / "pmu").exists() else scenario_dir
    frame = align_scenario_pmu_dir(pmu_dir)
    frame = frame.copy()
    frame["TIMESTAMP"] = pd.to_numeric(frame["TIMESTAMP"], errors="coerce")
    frame = frame.dropna(subset=["TIMESTAMP"]).sort_values("TIMESTAMP").drop_duplicates(subset=["TIMESTAMP"]).reset_index(drop=True)
    frame["EVENT"] = pd.to_numeric(frame["EVENT"], errors="coerce").fillna(0).astype(int)
    frame["DATA_PRESENT"] = pd.to_numeric(frame["DATA_PRESENT"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    return frame


def load_raw_holdout_frames(raw_input_root: Path) -> list[RawScenarioRef]:
    refs = discover_raw_scenario_dirs(raw_input_root)
    if not refs:
        raise FileNotFoundError(f"No RAW scenario folders found under {raw_input_root}")
    return refs
