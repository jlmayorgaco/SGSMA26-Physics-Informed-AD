"""Raw event-0 chunk loading helpers for m3 calibration."""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.calibration.calibration_metrics import wrap_deg


def normalize_bus_id(bus_id: Any) -> str:
    """Normalize bus ids to non-padded string format (e.g., 2 -> '2', '02' -> '2')."""
    raw = str(bus_id).strip()
    if raw.isdigit():
        return str(int(raw))
    return raw


def discover_event0_chunks(raw_chunks_dir: str | Path) -> list[Path]:
    """Discover event-0 chunk directories only."""
    base = Path(raw_chunks_dir)
    return sorted([Path(p) for p in glob.glob(str(base / "*_event_0_*"))])


def load_event0_chunks_for_bus(bus_id, raw_chunks_dir: str | Path) -> list[dict]:
    """Load event-0 chunk records for one bus."""
    bus = normalize_bus_id(bus_id)
    chunks = []
    for chunk_dir in discover_event0_chunks(raw_chunks_dir):
        path = chunk_dir / f"Bus{bus}.csv"
        if path.exists():
            chunks.append({"chunk_id": chunk_dir.name, "path": str(path), "df": pd.read_csv(path)})
    return chunks


def raw_col(bus_id, suffix) -> str:
    return f"BUS{normalize_bus_id(bus_id)}_{suffix}"


def transform_real_for_spec(values, spec):
    """Apply raw representation transform for experimental wrapped-delta specs."""
    if spec["support_status"] in {"supported_wrapped_delta", "experimental_relative_only"}:
        return wrap_deg(values - np.median(values))
    return values


def raw_values_from_chunks(chunks, bus_id, suffix, representation="raw") -> np.ndarray:
    """Concatenate raw values for one bus/suffix across event-0 chunks."""
    values = []
    col = raw_col(bus_id, suffix)
    for chunk in chunks:
        if col not in chunk["df"].columns:
            continue
        y = pd.to_numeric(chunk["df"][col], errors="coerce").dropna().to_numpy(float)
        if representation == "wrapped_delta" and len(y):
            y = wrap_deg(y - np.median(y))
        if len(y):
            values.append(y)
    return np.concatenate(values) if values else np.array([], dtype=float)
