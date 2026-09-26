"""Chunking pipeline migrated from legacy m0 with parity-first behavior."""

from __future__ import annotations

from dataclasses import dataclass
import glob
import json
import os
from typing import Any

import numpy as np
import pandas as pd


EVENT_MAP = {
    0: "Normal operation",
    1: "Fault",
    2: "Line outage",
    3: "Generation change/outage",
    4: "Load change/drop",
    5: "Missing data",
    6: "Missing data + physical event",
    7: "Bad data",
    8: "Unknown event",
}

CATEGORY_BACKGROUNDS = {
    "Cyber": {"events": [5, 7], "color": "#E6D8FF", "alpha": 0.5},
    "Physical": {"events": [1, 2, 3, 4], "color": "#DDEBFF", "alpha": 0.5},
    "Cyber Physical": {"events": [6], "color": "#FFF5CC", "alpha": 0.5},
    "Unknown": {"events": [8], "color": "#EEEEEE", "alpha": 0.5},
}


@dataclass(frozen=True)
class ChunkPipelinePaths:
    """Path bundle for m0-compatible output layout."""

    input_dir: str
    output_dir: str

    @property
    def chunks_dir(self) -> str:
        return os.path.join(self.output_dir, "chunks")

    @property
    def raw_sanity_summary_csv(self) -> str:
        return os.path.join(self.output_dir, "raw_signal_sanity_summary.csv")

    @property
    def chunk0_index_json(self) -> str:
        return os.path.join(self.output_dir, "chunk0_index.json")

    @property
    def chunks_metadata_json(self) -> str:
        return os.path.join(self.output_dir, "chunks_metadata.json")


def load_and_synchronize_data(input_dir: str) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Load all bus CSVs, round timestamps to 30 Hz grid, and align events."""
    all_files = glob.glob(os.path.join(input_dir, "*_nanmask.csv"))
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    bus_data: dict[str, pd.DataFrame] = {}
    for file_path in all_files:
        bus_name = os.path.basename(file_path).split("_")[0]
        df = pd.read_csv(file_path)
        df["TIMESTAMP"] = pd.to_numeric(df["TIMESTAMP"], errors="coerce").round(3)
        df = df.dropna(subset=["TIMESTAMP"])
        df = df.drop_duplicates(subset=["TIMESTAMP"], keep="first")
        df = df.sort_values("TIMESTAMP")
        df.set_index("TIMESTAMP", inplace=True)
        bus_data[bus_name] = df

    event_df = pd.DataFrame()
    for bus_name, df in bus_data.items():
        event_df[bus_name] = pd.to_numeric(df["Event"], errors="coerce")
    event_df = event_df.sort_index().ffill().fillna(0).astype(int)
    return bus_data, event_df


def find_chunk_boundaries(event_df: pd.DataFrame) -> tuple[list[float], pd.Series]:
    """Find contiguous chunks where global dominant event label is constant."""
    global_state = event_df.max(axis=1)
    state_changes = global_state != global_state.shift(1)

    boundaries = global_state.index[state_changes].tolist()
    if event_df.index[0] not in boundaries:
        boundaries.insert(0, event_df.index[0])
    boundaries.append(event_df.index[-1])
    return boundaries, global_state


def get_chunk_metadata(
    event_df_slice: pd.DataFrame,
    start_t: float,
    end_t: float,
    order_idx: int,
    dominant_label: int,
) -> dict[str, Any]:
    """Generate metadata for one chunk with legacy-compatible keys."""
    labels_present = set(event_df_slice.values.flatten())
    per_bus_labels = {bus: int(event_df_slice[bus].max()) for bus in event_df_slice.columns}
    affected = [bus for bus, evt in per_bus_labels.items() if evt != 0]
    duration_s = round(float(end_t - start_t), 10)

    chunk_meta: dict[str, Any] = {
        "chunk_order": order_idx,
        "label": int(dominant_label),
        "label_name": EVENT_MAP.get(dominant_label, "Unknown"),
        "description": EVENT_MAP.get(dominant_label, "Unknown"),
        "chunk_type": "event" if dominant_label != 0 else "normal",
        "start_time_s": float(start_t),
        "end_time_s": float(end_t),
        "duration_s": duration_s,
        "start_time_min": float(start_t / 60.0),
        "end_time_min": float(end_t / 60.0),
        "duration_min": float(duration_s / 60.0),
        "source_event_instance_id": None,
        "category": None,
        "impact_scope": None,
        "affected_buses": affected,
        "labels_present": sorted([int(label) for label in labels_present]),
        "labels_names": [EVENT_MAP.get(label) for label in sorted(labels_present)],
        "per_bus_labels": per_bus_labels,
    }

    for cat, props in CATEGORY_BACKGROUNDS.items():
        if dominant_label in props["events"]:
            chunk_meta["category"] = cat
            break
    return chunk_meta


def chunk_mask(index: pd.Index, start_t: float, end_t: float, include_end: bool) -> np.ndarray:
    """Legacy-compatible chunk mask helper."""
    if include_end:
        return (index >= start_t) & (index <= end_t)
    return (index >= start_t) & (index < end_t)


def append_sanity_rows(
    sanity_rows: list[dict[str, Any]],
    bus_slice: pd.DataFrame,
    bus_name: str,
    chunk_order: int,
    chunk_dir_name: str,
    dominant_label: int,
) -> None:
    """Append per-signal sanity rows with legacy-compatible fields."""
    if bus_slice.empty:
        return

    timestamps = bus_slice.index.to_numpy(dtype=float)
    dt = np.diff(timestamps) if len(timestamps) > 1 else np.array([])
    dt_pos = dt[dt > 0]
    median_dt = float(np.median(dt_pos)) if len(dt_pos) else np.nan
    sample_rate = float(1.0 / median_dt) if np.isfinite(median_dt) and median_dt > 0 else np.nan

    event_col = pd.to_numeric(bus_slice.get("Event"), errors="coerce")
    data_present_col = pd.to_numeric(bus_slice.get("DATA_PRESENT"), errors="coerce")

    common = {
        "chunk_order": chunk_order,
        "chunk_dir": chunk_dir_name,
        "bus_id": bus_name,
        "dominant_event_label": int(dominant_label),
        "dominant_event_name": EVENT_MAP.get(int(dominant_label), "Unknown"),
        "timestamp_start_s": float(timestamps[0]),
        "timestamp_end_s": float(timestamps[-1]),
        "n_rows": int(len(bus_slice)),
        "timestamp_monotonic": bool(np.all(np.diff(timestamps) >= 0)),
        "duplicate_timestamp_count": int(np.sum(np.diff(timestamps) == 0)) if len(timestamps) > 1 else 0,
        "sample_period_median_s": median_dt,
        "sample_rate_hz_est": sample_rate,
        "event_min": float(np.nanmin(event_col)) if event_col is not None else np.nan,
        "event_max": float(np.nanmax(event_col)) if event_col is not None else np.nan,
        "event_unique_count": int(event_col.nunique(dropna=True)) if event_col is not None else 0,
        "data_present_mean": float(np.nanmean(data_present_col)) if data_present_col is not None else np.nan,
    }

    excluded = {"DATA_PRESENT", "Event"}
    for signal in bus_slice.columns:
        if signal in excluded:
            continue
        series = pd.to_numeric(bus_slice[signal], errors="coerce")
        valid = series.dropna()
        n_total = int(len(series))
        n_valid = int(len(valid))
        n_nan = n_total - n_valid

        row = dict(common)
        row.update(
            {
                "signal_name": signal,
                "n_valid": n_valid,
                "n_nan": n_nan,
                "nan_pct": float(100.0 * n_nan / n_total) if n_total else np.nan,
                "mean": float(valid.mean()) if n_valid else np.nan,
                "std": float(valid.std(ddof=0)) if n_valid else np.nan,
                "min": float(valid.min()) if n_valid else np.nan,
                "max": float(valid.max()) if n_valid else np.nan,
                "p01": float(valid.quantile(0.01)) if n_valid else np.nan,
                "p50": float(valid.quantile(0.50)) if n_valid else np.nan,
                "p99": float(valid.quantile(0.99)) if n_valid else np.nan,
            }
        )
        sanity_rows.append(row)


def plot_event_timeline(chunk_meta_list: list[dict[str, Any]], max_time_s: float, output_path: str, bus_name: str | None = None) -> None:
    """Generate event timeline plot."""
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(18, 3), dpi=200)
    ax1.set_xlim(0, max_time_s)
    ax1.set_ylim(-0.5, len(EVENT_MAP) - 0.5)

    ax1.set_yticks(list(EVENT_MAP.keys()))
    ax1.set_yticklabels(list(EVENT_MAP.values()), fontsize=6)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Event label")

    ax2 = ax1.twiny()
    ax2.set_xlim(0, max_time_s / 60.0)
    ax2.set_xlabel("Time (min)")
    ax1.axhline(y=0, color="black", linewidth=0.5)

    legend_patches = []
    for cat, props in CATEGORY_BACKGROUNDS.items():
        patch = mpatches.Patch(color=props["color"], label=cat, alpha=props["alpha"])
        legend_patches.append(patch)

    for chunk in chunk_meta_list:
        start = chunk["start_time_s"]
        duration = chunk["duration_s"]
        category = chunk.get("category")

        if category:
            props = CATEGORY_BACKGROUNDS[category]
            ax1.axvspan(
                start,
                start + duration,
                facecolor=props["color"],
                alpha=props["alpha"],
                edgecolor="black",
                linestyle="--",
                linewidth=0.5,
            )

        if bus_name:
            events_to_plot = [chunk["per_bus_labels"].get(bus_name, 0)]
        else:
            events_to_plot = [evt for evt in chunk["labels_present"] if evt != 0]

        for evt in events_to_plot:
            if evt == 0:
                continue
            color = (
                "#8da0cb"
                if evt in [5, 7]
                else "#66c2a5"
                if evt == 3
                else "#fc8d62"
                if evt in [1, 2]
                else "gray"
            )
            ax1.broken_barh(
                [(start, duration)],
                (evt - 0.4, 0.8),
                facecolors=color,
                edgecolors="black",
                linewidth=0.5,
            )

    title = f"{bus_name} - Event timeline" if bus_name else "All PMU buses - Event timeline"
    plt.title(title, pad=20)
    ax1.legend(handles=legend_patches, loc="upper right", fontsize=6)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def run_chunk_pipeline(
    paths: ChunkPipelinePaths,
    enable_sanity_check: bool = True,
    generate_plots: bool = False,
) -> None:
    """Run chunking pipeline with m0-compatible outputs."""
    os.makedirs(paths.chunks_dir, exist_ok=True)

    bus_data, event_df = load_and_synchronize_data(paths.input_dir)
    boundaries, global_state = find_chunk_boundaries(event_df)
    max_time_s = boundaries[-1]
    sanity_rows: list[dict[str, Any]] = []
    chunks_meta_list: list[dict[str, Any]] = []

    for i in range(len(boundaries) - 1):
        start_t = boundaries[i]
        end_t = boundaries[i + 1]
        include_end = i == len(boundaries) - 2
        dominant_label = int(global_state.loc[start_t])

        event_mask = chunk_mask(event_df.index, start_t, end_t, include_end)
        event_slice = event_df.loc[event_mask]
        if event_slice.empty:
            continue

        meta = get_chunk_metadata(event_slice, start_t, end_t, i + 1, dominant_label)
        label_name_safe = EVENT_MAP[dominant_label].replace(" ", "_").replace("/", "_").lower()
        chunk_dir_name = f"chunk{i + 1:02d}_event_{dominant_label}_{label_name_safe}"
        meta["chunk_dir"] = chunk_dir_name
        chunks_meta_list.append(meta)
        chunk_path = os.path.join(paths.chunks_dir, chunk_dir_name)
        os.makedirs(chunk_path, exist_ok=True)

        for bus, df in bus_data.items():
            bus_mask = chunk_mask(df.index, start_t, end_t, include_end)
            bus_slice = df.loc[bus_mask].copy()
            csv_out = os.path.join(chunk_path, f"{bus}.csv")
            bus_slice.to_csv(csv_out)

            if enable_sanity_check:
                append_sanity_rows(
                    sanity_rows=sanity_rows,
                    bus_slice=bus_slice,
                    bus_name=bus,
                    chunk_order=i + 1,
                    chunk_dir_name=chunk_dir_name,
                    dominant_label=dominant_label,
                )

            if not generate_plots or bus_slice.empty:
                continue

            import matplotlib.pyplot as plt

            signals_to_plot = [col for col in bus_slice.columns if col not in ["DATA_PRESENT", "Event"]]
            if not signals_to_plot:
                continue
            fig, axes = plt.subplots(
                len(signals_to_plot),
                1,
                figsize=(10, 2 * len(signals_to_plot)),
                sharex=True,
            )
            if len(signals_to_plot) == 1:
                axes = [axes]
            for ax, signal in zip(axes, signals_to_plot):
                ax.plot(bus_slice.index, bus_slice[signal])
                ax.set_ylabel(signal)
                ax.grid(True, alpha=0.3)
            plt.xlabel("Time (s)")
            plt.tight_layout()
            plt.savefig(os.path.join(chunk_path, f"{bus}_signals_plot.png"))
            plt.close()

    with open(paths.chunks_metadata_json, "w", encoding="utf-8") as fh:
        json.dump({"chunks": chunks_meta_list}, fh, indent=2)

    chunk0_index = [
        {
            "chunk_order": int(chunk["chunk_order"]),
            "chunk_dir": chunk.get("chunk_dir"),
            "start_time_s": float(chunk["start_time_s"]),
            "end_time_s": float(chunk["end_time_s"]),
            "duration_s": float(chunk["duration_s"]),
            "affected_buses": chunk.get("affected_buses", []),
        }
        for chunk in chunks_meta_list
        if int(chunk.get("label", -1)) == 0
    ]
    with open(paths.chunk0_index_json, "w", encoding="utf-8") as fh:
        json.dump({"event_label": 0, "chunks": chunk0_index}, fh, indent=2)

    if enable_sanity_check:
        sanity_df = pd.DataFrame(sanity_rows)
        if not sanity_df.empty:
            sanity_df.sort_values(
                by=["chunk_order", "bus_id", "signal_name"],
                inplace=True,
                kind="stable",
            )
        sanity_df.to_csv(paths.raw_sanity_summary_csv, index=False)

    if generate_plots:
        plot_event_timeline(
            chunks_meta_list,
            max_time_s,
            os.path.join(paths.output_dir, "event_plot_all_buses_timeline.png"),
        )
        for bus in bus_data.keys():
            plot_event_timeline(
                chunks_meta_list,
                max_time_s,
                os.path.join(paths.output_dir, f"event_plot_{bus}.png"),
                bus_name=bus,
            )
