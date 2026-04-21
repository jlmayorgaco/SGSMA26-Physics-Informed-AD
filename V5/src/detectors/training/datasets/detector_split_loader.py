from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(slots=True)
class SplitRecord:
    scenario_id: str
    scenario_dir: Path
    split: str
    template_name: str = ""
    event_coarse: int | None = None
    difficulty_level: str = ""
    scenario_family: str = ""
    seed_family: str = ""


def _to_record(row: pd.Series, split: str, workspace_root: Path | None) -> SplitRecord:
    scenario_dir = Path(str(row.get("scenario_dir", "")).strip())
    if workspace_root is not None and not scenario_dir.is_absolute():
        scenario_dir = (workspace_root / scenario_dir).resolve()
    event_value = row.get("event_coarse", None)
    event_coarse = int(event_value) if str(event_value).strip() not in {"", "nan", "None"} else None
    return SplitRecord(
        scenario_id=str(row.get("scenario_id", scenario_dir.name)),
        scenario_dir=scenario_dir,
        split=split,
        template_name=str(row.get("template_name", "")),
        event_coarse=event_coarse,
        difficulty_level=str(row.get("difficulty_level", "")),
        scenario_family=str(row.get("scenario_family", "")),
        seed_family=str(row.get("seed_family", "")),
    )


def load_split_csv(split_csv: Path, *, split_name: str | None = None, workspace_root: Path | None = None) -> list[SplitRecord]:
    frame = pd.read_csv(split_csv)
    split = split_name or str(frame.get("split", pd.Series([""])).iloc[0] or split_csv.stem).lower()
    return [_to_record(row, split, workspace_root) for _, row in frame.iterrows()]


def validate_no_split_leakage(records_by_split: dict[str, list[SplitRecord]]) -> dict[str, bool]:
    ids_by_split = {k: {r.scenario_id for r in v} for k, v in records_by_split.items()}
    families_by_split = {k: {r.scenario_family for r in v if r.scenario_family} for k, v in records_by_split.items()}
    splits = list(records_by_split.keys())
    scenario_leak = False
    family_leak = False
    for i, s1 in enumerate(splits):
        for s2 in splits[i + 1 :]:
            if ids_by_split[s1] & ids_by_split[s2]:
                scenario_leak = True
            if families_by_split[s1] and families_by_split[s2] and (families_by_split[s1] & families_by_split[s2]):
                family_leak = True
    return {
        "no_scenario_leakage": not scenario_leak,
        "no_family_leakage": not family_leak,
    }

