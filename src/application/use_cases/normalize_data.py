"""Use case for m1 normalization pipeline (new architecture, parity-first)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.data_engineering.normalization import normalize_bus_data
from src.data_engineering.normalized_chunk_export import (
    export_baselines_csv,
    export_normalized_chunk_index,
    export_normalized_chunk_metadata,
    export_normalized_chunks,
)
from src.data_engineering.raw_loader import load_and_synchronize_data
from src.data_engineering.timeline_plots import plot_event_timeline


def run_normalize_data_use_case(
    input_dir: str | Path,
    output_dir: str | Path,
    generate_plots: bool = True,
) -> dict[str, Any]:
    """Run end-to-end normalized chunk generation with legacy-compatible outputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_bus_data, event_df = load_and_synchronize_data(input_dir)
    normalized_data, baseline_df = normalize_bus_data(raw_bus_data, event_df)

    baselines_csv = export_baselines_csv(baseline_df, output_dir)
    chunk_meta_list, chunk_index_entries, max_time_s = export_normalized_chunks(
        normalized_data=normalized_data,
        event_df=event_df,
        output_dir=output_dir,
        generate_plots=generate_plots,
    )
    chunks_metadata_json = export_normalized_chunk_metadata(chunk_meta_list, output_dir)
    normalized_chunk_index_json = export_normalized_chunk_index(chunk_index_entries, output_dir)

    if generate_plots:
        plot_event_timeline(
            chunk_meta_list=chunk_meta_list,
            max_time_s=max_time_s,
            output_path=str(output_dir / "event_plot_all_buses_timeline.png"),
        )
        for bus in raw_bus_data.keys():
            plot_event_timeline(
                chunk_meta_list=chunk_meta_list,
                max_time_s=max_time_s,
                output_path=str(output_dir / f"event_plot_{bus}.png"),
                bus_name=bus,
            )

    return {
        "input_dir": str(Path(input_dir)),
        "output_dir": str(output_dir),
        "baselines_csv": str(baselines_csv),
        "chunks_metadata_json": str(chunks_metadata_json),
        "normalized_chunk_index_json": str(normalized_chunk_index_json),
        "chunk_count": len(chunk_meta_list),
        "event0_chunk_count": len([c for c in chunk_meta_list if int(c["label"]) == 0]),
        "bus_count": len(normalized_data),
        "plots_generated": bool(generate_plots),
    }
