"""Export helpers for normalized m1 artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.data_engineering.chunking import CATEGORY_BACKGROUNDS, EVENT_MAP, chunk_mask, find_chunk_boundaries
from src.data_engineering.normalized_index import build_normalized_chunk_index


def export_baselines_csv(baseline_df: pd.DataFrame, output_dir: str | Path) -> Path:
    """Export normalization baseline report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "normalization_baselines.csv"
    baseline_df.to_csv(out_path, index=False)
    return out_path


def _chunk_dir_name(chunk_order: int, dominant_label: int) -> str:
    label_name_safe = EVENT_MAP[dominant_label].replace(" ", "_").replace("/", "_").lower()
    return f"chunk{chunk_order:02d}_event_{dominant_label}_{label_name_safe}"


def _chunk_category(dominant_label: int) -> str | None:
    for cat, props in CATEGORY_BACKGROUNDS.items():
        if dominant_label in props["events"]:
            return cat
    return None


def _write_chunk_signal_plot(
    bus_slice: pd.DataFrame,
    bus: str,
    chunk_order: int,
    label_name: str,
    output_path: Path,
) -> None:
    signals_to_plot = [col for col in bus_slice.columns if col not in ["DATA_PRESENT", "Event"]]
    if bus_slice.empty or len(signals_to_plot) == 0:
        return

    fig, axes = plt.subplots(
        len(signals_to_plot),
        1,
        figsize=(12, 2 * len(signals_to_plot)),
        sharex=True,
    )
    if len(signals_to_plot) == 1:
        axes = [axes]

    for ax, signal in zip(axes, signals_to_plot):
        ax.plot(bus_slice.index, bus_slice[signal], linewidth=1.0, color="#1f77b4")

        signal_upper = signal.upper()
        if "MAG" in signal_upper or signal_upper.endswith("_FREQ"):
            ax.axhline(1.0, color="red", linestyle="--", alpha=0.6, linewidth=1.0)
        elif "ROCOF" in signal_upper or "SPEED" in signal_upper or "DELTA" in signal_upper:
            ax.axhline(0.0, color="gray", linestyle="--", alpha=0.6, linewidth=1.0)

        ax.set_ylabel(signal, fontsize=8)
        ax.tick_params(axis="both", which="major", labelsize=8)
        ax.grid(True, alpha=0.3)

    plt.xlabel("Time (s)")
    plt.suptitle(
        f"{bus} Normalized Signals - Chunk {chunk_order} ({label_name})",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def export_normalized_chunks(
    normalized_data: dict[str, pd.DataFrame],
    event_df: pd.DataFrame,
    output_dir: str | Path,
    generate_plots: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    """Export normalized chunk CSVs and return metadata + index entries."""
    output_dir = Path(output_dir)
    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    boundaries, global_state = find_chunk_boundaries(event_df)
    max_time_s = float(boundaries[-1])

    chunks_meta_list: list[dict[str, Any]] = []
    chunk_index_entries: list[dict[str, Any]] = []

    for i in range(len(boundaries) - 1):
        start_t = boundaries[i]
        end_t = boundaries[i + 1]
        include_end = i == len(boundaries) - 2
        dominant_label = int(global_state.loc[start_t])

        mask = chunk_mask(event_df.index, start_t, end_t, include_end)
        event_slice = event_df.loc[mask]
        if event_slice.empty:
            continue

        labels_present = set(event_slice.values.flatten())
        per_bus_labels = {bus: int(event_slice[bus].max()) for bus in event_slice.columns}
        affected = [bus for bus, evt in per_bus_labels.items() if evt != 0]
        chunk_order = i + 1
        chunk_meta = {
            "chunk_order": chunk_order,
            "label": int(dominant_label),
            "label_name": EVENT_MAP.get(dominant_label, "Unknown"),
            "category": _chunk_category(dominant_label),
            "start_time_s": float(start_t),
            "end_time_s": float(end_t),
            "duration_s": float(end_t - start_t),
            "affected_buses": affected,
            "labels_present": sorted([int(label) for label in labels_present]),
            "per_bus_labels": per_bus_labels,
        }
        chunks_meta_list.append(chunk_meta)

        chunk_dir_name = _chunk_dir_name(chunk_order, dominant_label)
        chunk_path = chunks_dir / chunk_dir_name
        chunk_path.mkdir(parents=True, exist_ok=True)
        chunk_index_entry: dict[str, Any] = {
            "chunk_id": chunk_dir_name,
            "chunk_path": str(chunk_path),
            "label": int(dominant_label),
            "label_name": EVENT_MAP.get(dominant_label, "Unknown"),
            "duration_s": float(end_t - start_t),
            "start_time_s": float(start_t),
            "end_time_s": float(end_t),
            "buses_available": [],
            "bus_columns": {},
        }

        for bus, df in normalized_data.items():
            bus_mask = chunk_mask(df.index, start_t, end_t, include_end)
            bus_slice = df.loc[bus_mask].copy()
            csv_out = chunk_path / f"{bus}_normalized.csv"
            bus_slice.to_csv(csv_out)
            if len(bus_slice) > 0:
                chunk_index_entry["buses_available"].append(bus)
                chunk_index_entry["bus_columns"][bus] = [c for c in bus_slice.columns]

            if generate_plots:
                _write_chunk_signal_plot(
                    bus_slice=bus_slice,
                    bus=bus,
                    chunk_order=chunk_order,
                    label_name=chunk_meta["label_name"],
                    output_path=chunk_path / f"{bus}_normalized_signals_plot.png",
                )

        chunk_index_entries.append(chunk_index_entry)

    return chunks_meta_list, chunk_index_entries, max_time_s


def export_normalized_chunk_metadata(chunk_meta_list: list[dict], output_dir: str | Path) -> Path:
    """Export `chunks_metadata.json` for normalized pipeline."""
    output_dir = Path(output_dir)
    out_path = output_dir / "chunks_metadata.json"
    out_path.write_text(json.dumps({"chunks": chunk_meta_list}, indent=2), encoding="utf-8")
    return out_path


def export_normalized_chunk_index(chunk_index_entries: list[dict], output_dir: str | Path) -> Path:
    """Export `normalized_chunk_index.json` payload."""
    output_dir = Path(output_dir)
    out_path = output_dir / "normalized_chunk_index.json"
    payload = build_normalized_chunk_index(chunk_index_entries)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path
