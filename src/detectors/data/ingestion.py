from __future__ import annotations

from pathlib import Path
import json
import re

import numpy as np
import pandas as pd

from src.detectors.models import ScenarioRecord


def _scenario_from_row(row: pd.Series) -> ScenarioRecord:
    scenario_dir = Path(str(row.get("scenario_dir", "")).strip())
    return ScenarioRecord(
        scenario_id=str(row.get("scenario_id", scenario_dir.name)),
        scenario_dir=scenario_dir,
        template_name=str(row.get("template_name", "")),
        event_coarse=int(row["event_coarse"]) if str(row.get("event_coarse", "")).strip() else None,
        difficulty_level=str(row.get("difficulty_level", "")),
        scenario_family=str(row.get("scenario_family", "")),
        seed_family=str(row.get("seed_family", "")),
        split=str(row.get("split", "")),
    )


def load_split_records(split_csv: Path, workspace_root: Path | None = None) -> list[ScenarioRecord]:
    frame = pd.read_csv(split_csv)
    rows: list[ScenarioRecord] = []
    for _, row in frame.iterrows():
        rec = _scenario_from_row(row)
        if not rec.scenario_dir.is_absolute() and workspace_root is not None:
            rec.scenario_dir = (workspace_root / rec.scenario_dir).resolve()
        rows.append(rec)
    return rows


def _bus_token_from_name(name: str) -> str:
    m = re.search(r"Bus(\d+)", name, flags=re.IGNORECASE)
    if not m:
        return "UNK"
    return f"BUS{int(m.group(1))}"


def _event_frame_from_labels(labels_path: Path) -> pd.DataFrame | None:
    if not labels_path.exists():
        return None
    frame = pd.read_csv(labels_path)
    if "TIMESTAMP" not in frame.columns or "EVENT" not in frame.columns:
        return None
    out = frame[["TIMESTAMP", "EVENT"]].copy()
    out["TIMESTAMP"] = pd.to_numeric(out["TIMESTAMP"], errors="coerce")
    out["EVENT"] = pd.to_numeric(out["EVENT"], errors="coerce").fillna(0).astype(int)
    return out.dropna(subset=["TIMESTAMP"]).sort_values("TIMESTAMP")


def _list_pmu_csvs(pmu_dir: Path) -> list[Path]:
    return sorted(pmu_dir.glob("Bus*_Competition_Data_*.csv"))


def load_multibus_pmu_frame(pmu_dir: Path, labels_dir: Path | None = None) -> pd.DataFrame:
    files = _list_pmu_csvs(pmu_dir)
    if not files:
        raise FileNotFoundError(f"No PMU CSV files found in {pmu_dir}")
    merged: pd.DataFrame | None = None
    data_present_cols: list[str] = []
    event_cols: list[str] = []
    for path in files:
        bus = _bus_token_from_name(path.name)
        df = pd.read_csv(path)
        if "TIMESTAMP" not in df.columns:
            continue
        df["TIMESTAMP"] = pd.to_numeric(df["TIMESTAMP"], errors="coerce")
        keep_cols = [c for c in df.columns if c != "TIMESTAMP"]
        rename: dict[str, str] = {}
        if "DATA_PRESENT" in keep_cols:
            rename["DATA_PRESENT"] = f"DATA_PRESENT_{bus}"
            data_present_cols.append(f"DATA_PRESENT_{bus}")
        if "Event" in keep_cols:
            rename["Event"] = f"EVENT_{bus}"
            event_cols.append(f"EVENT_{bus}")
        bus_df = df[["TIMESTAMP"] + keep_cols].rename(columns=rename)
        merged = bus_df if merged is None else merged.merge(bus_df, on="TIMESTAMP", how="outer")
    if merged is None:
        raise ValueError(f"No valid PMU data could be read from {pmu_dir}")
    merged = merged.sort_values("TIMESTAMP").drop_duplicates(subset=["TIMESTAMP"]).reset_index(drop=True)
    for col in data_present_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0).astype(float)
    for col in event_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0).astype(int)
    merged["DATA_PRESENT"] = merged[data_present_cols].min(axis=1) if data_present_cols else 1.0
    merged["EVENT"] = merged[event_cols].max(axis=1).astype(int) if event_cols else 0
    if labels_dir is not None:
        labels_frame = _event_frame_from_labels(labels_dir / "event_frame_labels.csv")
        if labels_frame is not None and not labels_frame.empty:
            merged = merged.merge(labels_frame, on="TIMESTAMP", how="left", suffixes=("", "_LABELS"))
            merged["EVENT"] = merged["EVENT_LABELS"].fillna(merged["EVENT"]).astype(int)
            merged = merged.drop(columns=["EVENT_LABELS"])
    return merged


def load_scenario_frame(record: ScenarioRecord) -> pd.DataFrame:
    pmu_dir = record.scenario_dir / "pmu"
    labels_dir = record.scenario_dir / "labels"
    frame = load_multibus_pmu_frame(pmu_dir=pmu_dir, labels_dir=labels_dir)
    frame["SCENARIO_ID"] = record.scenario_id
    frame["TEMPLATE_NAME"] = record.template_name
    frame["EVENT_COARSE"] = record.event_coarse if record.event_coarse is not None else np.nan
    frame["DIFFICULTY_LEVEL"] = record.difficulty_level
    frame["SCENARIO_FAMILY"] = record.scenario_family
    frame["SEED_FAMILY"] = record.seed_family
    frame["SPLIT"] = record.split
    scoring_path = record.scenario_dir / "metadata" / "scenario_scoring.json"
    if scoring_path.exists():
        try:
            scoring = json.loads(scoring_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            scoring = {}
        for key in ("estimator_difficulty_score", "detector_difficulty_score", "classifier_difficulty_score", "localizer_difficulty_score", "overall_training_value_score"):
            frame[key.upper()] = float(scoring.get(key, 0.0) or 0.0)
    return frame


def load_raw_frame(raw_dir: Path) -> pd.DataFrame:
    frame = load_multibus_pmu_frame(raw_dir, labels_dir=None)
    frame["SCENARIO_ID"] = raw_dir.name.upper()
    frame["TEMPLATE_NAME"] = "RAW_REFERENCE"
    frame["EVENT_COARSE"] = np.nan
    frame["DIFFICULTY_LEVEL"] = "unknown"
    frame["SCENARIO_FAMILY"] = f"RAW::{raw_dir.name}"
    frame["SEED_FAMILY"] = ""
    frame["SPLIT"] = "raw"
    return frame

