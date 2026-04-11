"""Write prediction CSVs in competition submission format.

Submission format (per CLAUDE.md §11 / spec §11)
-------------------------------------------------
- Append two columns to each bus CSV:
    Predicted_Event    — integer event label (0–8)
    Predicted_Location — integer competition bus number (-1 = unknown)
- TIMESTAMP preserved exactly (3-decimal rounding per spec §11)
- Row order preserved exactly (positional alignment)

Also write a single combined submission.csv:
    TIMESTAMP, Bus, Predicted_Event, Predicted_Location
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from src.io.load_csv import PMU_BUSES

log = logging.getLogger(__name__)

_BUS_CSV_TEMPLATE = "Bus{bus}_Competition_Data_nanmask.csv"


def write_submission(
    df: pd.DataFrame,
    predicted_event: np.ndarray | Mapping[int, np.ndarray],
    predicted_location: np.ndarray | Mapping[int, np.ndarray],
    out_dir: Path | str,
    data_dir: Path | str,
) -> list[Path]:
    """Write per-bus prediction CSVs and combined submission.csv.

    Args:
        df:                 Merged DataFrame (used for TIMESTAMP reference).
        predicted_event:    (N,) array of global predicted labels, or
                            {bus: (N,) array} for bus-specific labels.
        predicted_location: (N,) array of global predicted locations, or
                            {bus: (N,) array} for bus-specific locations.
        out_dir:            Directory to write outputs into (created if absent).
        data_dir:           Directory containing original bus CSVs (to read TIMESTAMP
                            values from each bus file separately — preserving each
                            bus's original floating-point TIMESTAMP).

    Returns:
        List of output file paths written.
    """
    out_dir  = Path(out_dir)
    data_dir = Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    event_by_bus = isinstance(predicted_event, Mapping)
    loc_by_bus = isinstance(predicted_location, Mapping)
    first_event = next(iter(predicted_event.values())) if event_by_bus else predicted_event
    predicted_event_arr = np.asarray(first_event, dtype=int)
    n = len(predicted_event_arr)

    if not loc_by_bus and len(np.asarray(predicted_location, dtype=int)) != n:
        raise ValueError("predicted_event and predicted_location must have the same length")

    written: list[Path] = []
    combined_parts: list[pd.DataFrame] = []

    for bus in PMU_BUSES:
        # Read original bus CSV to get its exact TIMESTAMP column
        src_csv = data_dir / _BUS_CSV_TEMPLATE.format(bus=bus)
        if src_csv.exists():
            bus_df = pd.read_csv(src_csv, usecols=["TIMESTAMP"])
            bus_ts = bus_df["TIMESTAMP"].to_numpy(float)
            # Truncate or pad to match prediction length
            actual_n = min(len(bus_ts), n)
            ts_col = np.round(bus_ts[:actual_n], 3)
        else:
            # Fall back to merged TIMESTAMP
            ts_col = np.round(df["TIMESTAMP"].to_numpy(float)[:n], 3)
            actual_n = n

        if event_by_bus:
            if bus not in predicted_event:
                raise KeyError(f"Missing predicted_event array for Bus{bus}")
            bus_event = np.asarray(predicted_event[bus], dtype=int)
        else:
            bus_event = np.asarray(predicted_event, dtype=int)
        if loc_by_bus:
            if bus not in predicted_location:
                raise KeyError(f"Missing predicted_location array for Bus{bus}")
            bus_location = np.asarray(predicted_location[bus], dtype=int)
        else:
            bus_location = np.asarray(predicted_location, dtype=int)
        if len(bus_event) != n or len(bus_location) != n:
            raise ValueError(f"Bus{bus}: prediction arrays must have length {n}")

        out_df = pd.DataFrame({
            "TIMESTAMP":          ts_col,
            "Predicted_Event":    bus_event[:actual_n],
            "Predicted_Location": bus_location[:actual_n],
        })

        out_path = out_dir / f"Bus{bus}_Predictions.csv"
        out_df.to_csv(out_path, index=False)
        written.append(out_path)
        log.info("Wrote %s (%d rows)", out_path, len(out_df))

        # Accumulate for combined file
        part = out_df.copy()
        part.insert(1, "Bus", bus)
        combined_parts.append(part)

    # Write combined submission.csv
    combined = pd.concat(combined_parts, ignore_index=True)
    combined_path = out_dir / "submission.csv"
    combined.to_csv(combined_path, index=False)
    log.info("Wrote combined %s (%d rows)", combined_path, len(combined))
    written.append(combined_path)

    return written


def verify_timestamps(
    out_dir: Path | str,
    data_dir: Path | str,
    tol: float = 1e-3,
) -> dict[int, bool]:
    """Verify that output TIMESTAMP columns match source CSVs to `tol`.

    Returns {bus: ok} dict.
    """
    out_dir  = Path(out_dir)
    data_dir = Path(data_dir)
    result: dict[int, bool] = {}

    for bus in PMU_BUSES:
        src_csv = data_dir / _BUS_CSV_TEMPLATE.format(bus=bus)
        out_csv = out_dir / f"Bus{bus}_Predictions.csv"
        if not src_csv.exists() or not out_csv.exists():
            result[bus] = False
            continue

        src_ts = pd.read_csv(src_csv, usecols=["TIMESTAMP"])["TIMESTAMP"].to_numpy(float)
        out_ts = pd.read_csv(out_csv, usecols=["TIMESTAMP"])["TIMESTAMP"].to_numpy(float)

        n = min(len(src_ts), len(out_ts))
        ok = bool(np.all(np.abs(src_ts[:n] - out_ts[:n]) <= tol))
        result[bus] = ok
        if not ok:
            bad = np.where(np.abs(src_ts[:n] - out_ts[:n]) > tol)[0]
            log.warning("Bus%d: %d timestamp mismatches (first at row %d)", bus, len(bad), bad[0])

    return result
