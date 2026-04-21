"""Raw PMU CSV loading and synchronization helpers for m1 normalization."""

from __future__ import annotations

from pathlib import Path
import glob
import os

import pandas as pd


def load_and_synchronize_data(input_dir: str | Path) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Load all `*_nanmask.csv` files and synchronize Event labels by timestamp."""
    input_dir_str = str(input_dir)
    all_files = glob.glob(os.path.join(input_dir_str, "*_nanmask.csv"))
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir_str}")

    bus_data: dict[str, pd.DataFrame] = {}
    for file_path in all_files:
        bus_name = Path(file_path).name.split("_")[0]
        df = pd.read_csv(file_path)
        df["TIMESTAMP"] = pd.to_numeric(df["TIMESTAMP"], errors="coerce").round(3)
        df = df.dropna(subset=["TIMESTAMP"])
        df = df.drop_duplicates(subset=["TIMESTAMP"], keep="first")
        df = df.sort_values("TIMESTAMP")
        df = df.set_index("TIMESTAMP")
        bus_data[bus_name] = df

    event_df = pd.DataFrame()
    for bus_name, df in bus_data.items():
        event_df[bus_name] = pd.to_numeric(df["Event"], errors="coerce")
    event_df = event_df.sort_index().ffill().fillna(0).astype(int)
    return bus_data, event_df
