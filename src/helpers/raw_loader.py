from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd


_BUS_FILE_PATTERN = re.compile(r"^Bus\d+_Competition_Data_nanmask\.csv$", re.IGNORECASE)
_META_COLUMNS = {"TIMESTAMP", "DATA_PRESENT", "Event", "EVENT"}


@dataclass(slots=True)
class RawAlignedData:
    timeline: pd.Index
    aligned_frames: dict[str, pd.DataFrame]


def discover_raw_bus_files(raw_dir: Path) -> dict[str, Path]:
    if not raw_dir.exists() or not raw_dir.is_dir():
        raise FileNotFoundError(f"RAW directory not found: {raw_dir}")
    files: dict[str, Path] = {}
    for path in sorted(raw_dir.iterdir()):
        if not path.is_file():
            continue
        if not _BUS_FILE_PATTERN.match(path.name):
            continue
        bus = path.name.split("_", maxsplit=1)[0]
        files[bus] = path
    if not files:
        raise FileNotFoundError(f"No RAW bus files matching '*_Competition_Data_nanmask.csv' under {raw_dir}")
    return files


def load_raw_bus_frame(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    required = {"TIMESTAMP", "DATA_PRESENT", "Event"}
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")
    frame = frame.copy()
    frame["TIMESTAMP"] = pd.to_numeric(frame["TIMESTAMP"], errors="coerce")
    frame = frame.dropna(subset=["TIMESTAMP"]).sort_values("TIMESTAMP")
    frame = frame.drop_duplicates(subset=["TIMESTAMP"], keep="first")
    frame["DATA_PRESENT"] = pd.to_numeric(frame["DATA_PRESENT"], errors="coerce")
    frame["Event"] = pd.to_numeric(frame["Event"], errors="coerce")
    return frame


def measurement_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in _META_COLUMNS]


def align_raw_frames(frames_by_bus: dict[str, pd.DataFrame]) -> RawAlignedData:
    if not frames_by_bus:
        raise ValueError("frames_by_bus must contain at least one bus.")
    timeline = (
        pd.concat([frame["TIMESTAMP"] for frame in frames_by_bus.values()], axis=0, ignore_index=True)
        .dropna()
        .drop_duplicates()
        .sort_values(ignore_index=True)
    )
    timeline_index = pd.Index(timeline.to_numpy(dtype=float), name="TIMESTAMP")

    aligned: dict[str, pd.DataFrame] = {}
    for bus, frame in sorted(frames_by_bus.items()):
        local = frame.set_index("TIMESTAMP").sort_index()
        local = local.reindex(timeline_index)
        local.index.name = "TIMESTAMP"
        aligned[bus] = local
    return RawAlignedData(timeline=timeline_index, aligned_frames=aligned)


def load_and_align_raw_directory(raw_dir: Path) -> RawAlignedData:
    bus_files = discover_raw_bus_files(raw_dir)
    frames = {bus: load_raw_bus_frame(path) for bus, path in bus_files.items()}
    return align_raw_frames(frames)

