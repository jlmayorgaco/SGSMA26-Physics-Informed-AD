#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


EVENT_LABELS = {
    0: "Normal",
    1: "Fault",
    2: "Line outage",
    3: "Generation change/outage",
    4: "Load change/drop",
    5: "Missing data",
    6: "Missing data + physical event",
    7: "Bad data",
    8: "Unknown event",
}


def plot_event_histogram_all_buses(
    input_dir: Path,
    pattern: str = "Bus*_Competition_Data*.csv",
    output_dir: Path | None = None,
) -> None:
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {input_dir} with pattern {pattern!r}")

    per_bus_counts: list[pd.DataFrame] = []
    all_events: list[pd.Series] = []

    for path in files:
        df = pd.read_csv(path)

        if "Event" not in df.columns:
            raise ValueError(f"'Event' column not found in {path.name}")

        bus_name = path.stem.replace("_Competition_Data_nanmask", "").replace("_Competition_Data", "")
        event_series = pd.to_numeric(df["Event"], errors="coerce").dropna().astype(int)

        counts = event_series.value_counts().sort_index()
        counts_df = counts.rename("count").reset_index().rename(columns={"index": "event_id"})
        counts_df["bus"] = bus_name
        per_bus_counts.append(counts_df)

        all_events.append(event_series)

    counts_by_bus = pd.concat(per_bus_counts, ignore_index=True)

    all_events_series = pd.concat(all_events, ignore_index=True)
    total_counts = all_events_series.value_counts().sort_index()

    full_index = pd.Index(range(0, 9), name="event_id")
    total_counts = total_counts.reindex(full_index, fill_value=0)

    plot_df = total_counts.reset_index()
    plot_df.columns = ["event_id", "count"]
    plot_df["label"] = plot_df["event_id"].map(EVENT_LABELS).fillna("Unknown")

    plt.figure(figsize=(12, 6))
    plt.bar(plot_df["event_id"].astype(str), plot_df["count"])
    plt.title("Histogram of Event labels across all raw PMU buses")
    plt.xlabel("Event ID")
    plt.ylabel("Row count")
    plt.grid(True, axis="y", alpha=0.3)

    for i, row in plot_df.iterrows():
        plt.text(i, row["count"], str(int(row["count"])), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()

    if output_dir is None:
        output_dir = input_dir / "event_histogram_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig_path = output_dir / "event_histogram_all_buses.png"
    csv_total_path = output_dir / "event_histogram_all_buses.csv"
    csv_by_bus_path = output_dir / "event_histogram_per_bus.csv"

    plt.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close()

    plot_df.to_csv(csv_total_path, index=False)
    counts_by_bus.to_csv(csv_by_bus_path, index=False)

    print("Done.")
    print(f"Figure: {fig_path}")
    print(f"Totals CSV: {csv_total_path}")
    print(f"Per-bus CSV: {csv_by_bus_path}")
    print()
    print("Event totals across all buses:")
    print(plot_df.to_string(index=False))


if __name__ == "__main__":
    plot_event_histogram_all_buses(
        input_dir=Path("data/RAW0001"),   # <-- cámbialo a tu carpeta real
        pattern="Bus*_Competition_Data*.csv",
        output_dir=Path("outputs/event_histogram"),
    )