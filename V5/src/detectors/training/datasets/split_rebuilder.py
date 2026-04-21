from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


CYBER_EVENTS = {5, 7}
PHYSICAL_EVENTS = {1, 2, 3, 4}
CONCURRENT_EVENTS = {6, 8}


@dataclass(slots=True)
class RebuildResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    manifest: dict[str, object]


def _family(event: int) -> str:
    if event == 0:
        return "normal"
    if event in PHYSICAL_EVENTS:
        return "physical_heavy"
    if event in CYBER_EVENTS:
        return "cyber_heavy"
    if event in CONCURRENT_EVENTS:
        return "concurrent_heavy"
    return "unknown"


def _coverage(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, group in frame.groupby("split"):
        for fam, fam_group in group.groupby("family"):
            rows.append(
                {
                    "split": split,
                    "family": fam,
                    "scenarios": int(fam_group["scenario_id"].nunique()),
                    "event_values": ",".join(sorted({str(int(v)) for v in fam_group["event_coarse"].astype(int).tolist()})),
                }
            )
    return pd.DataFrame(rows).sort_values(["split", "family"]).reset_index(drop=True)


def rebuild_split_v3(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    min_val_per_family: int = 1,
    min_test_per_family: int = 1,
    additional_val_per_family: int = 0,
    additional_test_per_family: int = 0,
    extra_pool_df: pd.DataFrame | None = None,
) -> RebuildResult:
    frames = [train_df, val_df, test_df]
    if extra_pool_df is not None and not extra_pool_df.empty:
        frames.append(extra_pool_df)
    union = pd.concat(frames, ignore_index=True)
    union = union.drop_duplicates(subset=["scenario_id"]).copy()
    union["event_coarse"] = pd.to_numeric(union["event_coarse"], errors="coerce").fillna(0).astype(int)
    union["family"] = union["event_coarse"].apply(_family)

    selected_val: list[str] = []
    selected_test: list[str] = []
    occupied: set[str] = set()
    for fam in ["normal", "physical_heavy", "cyber_heavy", "concurrent_heavy"]:
        pool = union.loc[union["family"] == fam].sort_values(["difficulty_level", "scenario_id"])
        if pool.empty:
            continue
        val_pick = pool.loc[~pool["scenario_id"].isin(occupied)].head(min_val_per_family)["scenario_id"].tolist()
        occupied.update(val_pick)
        selected_val.extend(val_pick)
        test_pick = pool.loc[~pool["scenario_id"].isin(occupied)].head(min_test_per_family)["scenario_id"].tolist()
        occupied.update(test_pick)
        selected_test.extend(test_pick)
        if additional_val_per_family > 0:
            more_val = pool.loc[~pool["scenario_id"].isin(occupied)].head(additional_val_per_family)["scenario_id"].tolist()
            occupied.update(more_val)
            selected_val.extend(more_val)
        if additional_test_per_family > 0:
            more_test = pool.loc[~pool["scenario_id"].isin(occupied)].head(additional_test_per_family)["scenario_id"].tolist()
            occupied.update(more_test)
            selected_test.extend(more_test)

    # Keep existing split membership where possible for remaining rows.
    existing = {
        **{sid: "train" for sid in train_df["scenario_id"].astype(str).tolist()},
        **{sid: "val" for sid in val_df["scenario_id"].astype(str).tolist()},
        **{sid: "test" for sid in test_df["scenario_id"].astype(str).tolist()},
    }

    def assign(sid: str) -> str:
        if sid in selected_val:
            return "val"
        if sid in selected_test:
            return "test"
        return existing.get(sid, "train")

    union["split"] = union["scenario_id"].astype(str).map(assign)

    # Guarantee at least one train scenario for fitting.
    if int((union["split"] == "train").sum()) == 0 and len(union) > 0:
        fallback_idx = union.index[0]
        union.loc[fallback_idx, "split"] = "train"

    train_out = union.loc[union["split"] == "train"].drop(columns=["family"]).reset_index(drop=True)
    val_out = union.loc[union["split"] == "val"].drop(columns=["family"]).reset_index(drop=True)
    test_out = union.loc[union["split"] == "test"].drop(columns=["family"]).reset_index(drop=True)

    coverage = _coverage(union)
    manifest = {
        "total_scenarios": int(union["scenario_id"].nunique()),
        "counts": {
            "train": int(len(train_out)),
            "val": int(len(val_out)),
            "test": int(len(test_out)),
        },
        "coverage": coverage.to_dict(orient="records"),
        "constraints": {
            "min_val_per_family": int(min_val_per_family),
            "min_test_per_family": int(min_test_per_family),
            "additional_val_per_family": int(additional_val_per_family),
            "additional_test_per_family": int(additional_test_per_family),
        },
        "selected": {
            "val": selected_val,
            "test": selected_test,
        },
    }
    return RebuildResult(train=train_out, val=val_out, test=test_out, manifest=manifest)


def export_rebuilt_splits(result: RebuildResult, output_metadata_dir: Path) -> tuple[Path, Path, Path, Path, Path]:
    output_metadata_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_metadata_dir / "train_scenarios_v3.csv"
    val_path = output_metadata_dir / "val_scenarios_v3.csv"
    test_path = output_metadata_dir / "test_scenarios_v3.csv"
    manifest_path = output_metadata_dir / "split_manifest_v3.json"
    coverage_path = output_metadata_dir / "family_coverage_report.csv"

    result.train.to_csv(train_path, index=False)
    result.val.to_csv(val_path, index=False)
    result.test.to_csv(test_path, index=False)
    import json

    manifest_path.write_text(json.dumps(result.manifest, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(result.manifest.get("coverage", [])).to_csv(coverage_path, index=False)
    return train_path, val_path, test_path, manifest_path, coverage_path


def load_split_frames(train_csv: Path, val_csv: Path, test_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(train_csv), pd.read_csv(val_csv), pd.read_csv(test_csv)


def discover_scenario_pool(workspace_root: Path, *, existing_ids: Iterable[str] | None = None) -> pd.DataFrame:
    root = workspace_root / "data" / "scenarios"
    if not root.exists():
        return pd.DataFrame()
    existing = {str(s) for s in (existing_ids or [])}
    rows: list[dict[str, object]] = []
    for scenario_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        sid = scenario_dir.name
        if sid in existing:
            continue
        pmu_dir = scenario_dir / "pmu"
        labels_dir = scenario_dir / "labels"
        intervals = labels_dir / "event_intervals.csv"
        if not pmu_dir.exists() or not intervals.exists():
            continue
        event_coarse = 0
        template_name = "AUTO_EVENT0_NORMAL"
        scenario_family = f"{template_name}::::{sid}"
        try:
            frame = pd.read_csv(intervals)
            events = pd.to_numeric(frame.get("EVENT", pd.Series([], dtype=float)), errors="coerce").dropna().astype(int).tolist()
            families = frame.get("EVENT_FAMILY", pd.Series([], dtype=str)).astype(str).str.lower().tolist()
            has_physical = any(f == "physical" for f in families) or any(int(v) in PHYSICAL_EVENTS for v in events)
            has_cyber = any(f == "cyber" for f in families) or any(int(v) in CYBER_EVENTS | CONCURRENT_EVENTS for v in events)
            if has_physical and has_cyber:
                event_coarse = 8
            elif has_cyber:
                event_coarse = 5
            elif has_physical:
                event_coarse = 1
            elif events:
                event_coarse = int(max(events))
            template_name = f"AUTO_EVENT{event_coarse}"
            scenario_family = f"{template_name}::{sid}"
        except Exception:
            pass
        rows.append(
            {
                "scenario_id": sid,
                "scenario_dir": str((Path("data") / "scenarios" / sid).as_posix()),
                "template_name": template_name,
                "event_coarse": int(event_coarse),
                "difficulty_level": "auto",
                "scenario_family": scenario_family,
                "seed_family": f"{template_name}::auto",
                "split": "",
            }
        )
    return pd.DataFrame(rows)
