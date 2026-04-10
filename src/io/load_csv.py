"""Load and merge the 8 PMU CSVs.

Header inspection is mandatory — never assume spec column order.
Per CLAUDE.md §2.1: actual order is ANG before MAG, which is opposite to the spec.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

# PMU buses present in the competition data
PMU_BUSES = [2, 5, 6, 10, 19, 22, 29, 39]

# Expected column suffixes per bus (14 measurement channels + DATA_PRESENT + Event)
_EXPECTED_SUFFIXES = [
    "VA_ANG", "VA_MAG", "VB_ANG", "VB_MAG", "VC_ANG", "VC_MAG",
    "IA_ANG", "IA_MAG", "IB_ANG", "IB_MAG", "IC_ANG", "IC_MAG",
    "Freq", "ROCOF",
]


def _csv_path(data_dir: Path, bus: int) -> Path:
    return data_dir / f"Bus{bus}_Competition_Data_nanmask.csv"


def inspect_header(data_dir: Path | str, bus: int = 2) -> list[str]:
    """Return the column list of a single bus CSV and assert expected layout."""
    data_dir = Path(data_dir)
    path = _csv_path(data_dir, bus)
    cols = pd.read_csv(path, nrows=0).columns.tolist()

    prefix = f"BUS{bus}_"
    expected_first = f"{prefix}VA_ANG"
    if cols[1] != expected_first:
        log.warning(
            "Column order differs from CLAUDE.md §2.1. Got %s, expected %s. "
            "Proceeding with actual order.",
            cols[1],
            expected_first,
        )
    else:
        log.info("Header check passed: ANG before MAG as documented.")

    assert cols[0] == "TIMESTAMP", f"First column must be TIMESTAMP, got {cols[0]}"
    assert cols[-1] == "Event", f"Last column must be Event, got {cols[-1]}"
    assert "DATA_PRESENT" in cols, "DATA_PRESENT column missing"
    return cols


def load_single_bus(data_dir: Path | str, bus: int) -> pd.DataFrame:
    """Load one bus CSV with proper dtypes."""
    path = _csv_path(Path(data_dir), bus)
    df = pd.read_csv(path, dtype={"Event": "Int8", "DATA_PRESENT": "Int8"})
    df["TIMESTAMP"] = df["TIMESTAMP"].astype(float)
    return df


def load_all(data_dir: Path | str) -> pd.DataFrame:
    """Merge all 8 PMU CSVs into a single DataFrame aligned by row position.

    Per CLAUDE.md §2.5: timestamps differ by ~1e-11 s across CSVs (floating-point
    clock drift). All CSVs have identical row counts in identical temporal order.
    We align positionally rather than by TIMESTAMP value to avoid outer-join bloat.

    REAL-DATA CORRECTION to CLAUDE.md §2.3:
    Event labels are NOT globally identical across CSVs. Cyber events (5, 6) appear
    only in Bus29's CSV because they represent Bus29-specific data drops. Bus2 shows
    Event=7 (bad data) at startup rows ~91-111 that other buses don't mark.
    We compute a global event by taking the max non-zero event across all buses.
    Ties between non-zero events are broken by using the per-bus BUSk_Event columns
    (retained for debugging).

    Returns a single DataFrame with:
      - TIMESTAMP (from Bus2, representative)
      - 8 × 14 = 112 measurement columns (prefixed BUSk_)
      - 8 BUSk_DATA_PRESENT columns
      - 8 BUSk_Event columns (per-bus raw labels, for debugging)
      - 1 Event column (global label = max non-zero across buses)
    """
    data_dir = Path(data_dir)
    inspect_header(data_dir)  # sanity check on Bus2

    frames: list[pd.DataFrame] = []
    for bus in PMU_BUSES:
        df = load_single_bus(data_dir, bus)
        df = df.rename(columns={
            "DATA_PRESENT": f"BUS{bus}_DATA_PRESENT",
            "Event": f"BUS{bus}_Event",
        })
        frames.append(df)

    # Verify all CSVs have the same row count
    row_counts = [len(f) for f in frames]
    if len(set(row_counts)) > 1:
        log.warning("CSVs have different row counts: %s — truncating to min", row_counts)
        min_rows = min(row_counts)
        frames = [f.iloc[:min_rows].reset_index(drop=True) for f in frames]

    # Use Bus2 TIMESTAMP as the reference
    ref_ts = frames[0][["TIMESTAMP"]].reset_index(drop=True)

    # Drop TIMESTAMP from all frames before concat
    meas_frames: list[pd.DataFrame] = [
        df.drop(columns=["TIMESTAMP"]).reset_index(drop=True) for df in frames
    ]

    merged = pd.concat([ref_ts] + meas_frames, axis=1)

    # Compute global Event = max non-zero event across all buses
    event_cols = [f"BUS{bus}_Event" for bus in PMU_BUSES]
    event_matrix = merged[event_cols].fillna(0).astype(int)
    merged["Event"] = event_matrix.max(axis=1).astype("Int8")

    log.info(
        "Merged DataFrame: %d rows × %d cols (expected 161 379 rows)",
        len(merged),
        len(merged.columns),
    )
    log.info("Global event distribution: %s", merged["Event"].value_counts().to_dict())
    return merged


def get_measurement_cols(bus: int) -> list[str]:
    """Return the 14 measurement column names for a given bus."""
    prefix = f"BUS{bus}_"
    return [f"{prefix}{s}" for s in _EXPECTED_SUFFIXES]


def get_all_measurement_cols() -> list[str]:
    """Return all 112 measurement column names."""
    return [col for bus in PMU_BUSES for col in get_measurement_cols(bus)]
