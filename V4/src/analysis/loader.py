from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config.constants import MEASUREMENT_COLUMNS, OPTIONAL_COLUMNS, TIMESTAMP_COLUMN
from src.config.models import BusData
from src.utils.naming import infer_bus_id, normalize_raw_column_name


def canonicalize_pmu_columns(df: pd.DataFrame, path: Optional[Path] = None) -> pd.DataFrame:
    """
    Convert headers like:
      BUS2_VA_ANG, BUS2_VA_MAG, ..., BUS2_Freq, BUS2_ROCOF
    into canonical names:
      VA_ang, VA_mag, ..., Frequency, ROCOF

    Keeps TIMESTAMP, DATA_PRESENT, Event as-is.
    """
    original_cols = list(df.columns)
    cols = [normalize_raw_column_name(c) for c in original_cols]

    detected_bus_prefixes = set()
    renamed: dict[str, str] = {}

    canonical_map = {
        "VA_MAG": "VA_mag",
        "VA_ANG": "VA_ang",
        "VB_MAG": "VB_mag",
        "VB_ANG": "VB_ang",
        "VC_MAG": "VC_mag",
        "VC_ANG": "VC_ang",
        "IA_MAG": "IA_mag",
        "IA_ANG": "IA_ang",
        "IB_MAG": "IB_mag",
        "IB_ANG": "IB_ang",
        "IC_MAG": "IC_mag",
        "IC_ANG": "IC_ang",
        "FREQ": "Frequency",
        "FREQUENCY": "Frequency",
        "ROCOF": "ROCOF",
    }

    for raw_col, col in zip(original_cols, cols):
        upper = col.upper()

        if upper == "TIMESTAMP":
            renamed[raw_col] = "TIMESTAMP"
            continue
        if upper == "DATA_PRESENT":
            renamed[raw_col] = "DATA_PRESENT"
            continue
        if upper == "EVENT":
            renamed[raw_col] = "Event"
            continue

        m = re.match(r"^(BUS\d+)_(VA|VB|VC|IA|IB|IC)_(ANG|MAG)$", upper)
        if m:
            bus_prefix, ph, suffix = m.groups()
            detected_bus_prefixes.add(bus_prefix)
            renamed[raw_col] = f"{ph}_{suffix.lower()}"
            continue

        m = re.match(r"^(BUS\d+)_(FREQ|FREQUENCY)$", upper)
        if m:
            bus_prefix, _ = m.groups()
            detected_bus_prefixes.add(bus_prefix)
            renamed[raw_col] = "Frequency"
            continue

        m = re.match(r"^(BUS\d+)_ROCOF$", upper)
        if m:
            bus_prefix = m.group(1)
            detected_bus_prefixes.add(bus_prefix)
            renamed[raw_col] = "ROCOF"
            continue

        if upper in canonical_map:
            renamed[raw_col] = canonical_map[upper]
            continue

        renamed[raw_col] = col

    if len(detected_bus_prefixes) > 1:
        raise ValueError(
            f"CSV appears to contain multiple BUS prefixes {sorted(detected_bus_prefixes)}"
            + (f" in file {path}" if path is not None else "")
        )

    return df.rename(columns=renamed)


def required_columns_present(df: pd.DataFrame, path: Optional[Path] = None) -> None:
    missing = [c for c in MEASUREMENT_COLUMNS if c not in df.columns]
    if missing:
        where = f" in file {path}" if path is not None else ""
        raise ValueError(
            f"Missing required columns{where}: {missing}\n"
            f"Detected columns: {list(df.columns)}"
        )

    if TIMESTAMP_COLUMN not in df.columns:
        raise ValueError(
            f"Missing {TIMESTAMP_COLUMN} column" + (f" in file {path}" if path is not None else "")
        )


def load_bus_csv(path: Path) -> BusData:
    df = pd.read_csv(path, sep=None, engine="python")
    df = canonicalize_pmu_columns(df, path=path)
    required_columns_present(df, path=path)

    df = df.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)

    numeric_cols = [TIMESTAMP_COLUMN] + MEASUREMENT_COLUMNS + [c for c in OPTIONAL_COLUMNS if c in df.columns]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "DATA_PRESENT" not in df.columns:
        df["DATA_PRESENT"] = 1

    if "Event" not in df.columns:
        df["Event"] = 0

    t = df[TIMESTAMP_COLUMN].to_numpy(dtype=float)
    dt = t[1:] - t[:-1]
    valid_dt = dt[(dt > 0)]
    sampling_rate_hz = float(1.0 / valid_dt.mean()) if len(valid_dt) else float("nan")

    return BusData(
        bus_id=infer_bus_id(path),
        path=path,
        df=df,
        sampling_rate_hz=sampling_rate_hz,
    )


def load_all_buses(input_dir: Path, pattern: str) -> list[BusData]:
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found in {input_dir} with pattern {pattern!r}")

    buses = [load_bus_csv(p) for p in files]
    buses = sorted(buses, key=lambda b: int(re.sub(r"\D", "", b.bus_id) or "0"))
    return buses