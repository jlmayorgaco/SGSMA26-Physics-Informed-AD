"""Use case for generating raw PMU noise profiles from chunked CSV artifacts."""

from __future__ import annotations

from collections import defaultdict
import glob
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.calibration.noise_profiler import analyze_raw_signal
from src.calibration.profile_aggregation import aggregate_entries
from src.calibration.profile_schema import EXCLUDED_SIGNAL_COLUMNS


def run_profile_noise_use_case(
    chunks_dir: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Generate aggregated raw-noise profiles from chunked raw CSV inputs."""
    chunks_dir = Path(chunks_dir)
    output_path = Path(output_path)
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    chunk_dirs = sorted(glob.glob(str(chunks_dir / "chunk*_event_*")))
    for chunk_dir in chunk_dirs:
        name = Path(chunk_dir).name
        try:
            event_type = str(int(name.split("_")[2]))
        except (IndexError, ValueError):
            continue

        for csv_path in sorted(glob.glob(str(Path(chunk_dir) / "Bus*.csv"))):
            bus_id = Path(csv_path).stem.replace("Bus", "")
            df = pd.read_csv(csv_path)
            timestamps = df["TIMESTAMP"].to_numpy() if "TIMESTAMP" in df.columns else None
            for col in df.columns:
                if col in EXCLUDED_SIGNAL_COLUMNS:
                    continue
                values = pd.to_numeric(df[col], errors="coerce").to_numpy()
                analysis = analyze_raw_signal(values, col, timestamps)
                if analysis is not None:
                    acc[event_type][bus_id][col].append(analysis)

    profiles: dict[str, Any] = {}
    for event_type in sorted(acc.keys(), key=lambda x: int(x)):
        profiles[event_type] = {}
        for bus_id in sorted(acc[event_type].keys(), key=lambda x: int(x)):
            profiles[event_type][bus_id] = {}
            for col in sorted(acc[event_type][bus_id].keys()):
                profiles[event_type][bus_id][col] = aggregate_entries(acc[event_type][bus_id][col])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(profiles, indent=2), encoding="utf-8")
    return profiles
