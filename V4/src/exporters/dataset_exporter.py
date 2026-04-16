from pathlib import Path
import textwrap
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.cross_bus import cross_bus_correlations, cross_bus_rankings
from src.analysis.dataset_integrity import build_dataset_integrity
from src.analysis.events import event_spans, summarize_event_spans, build_global_event_spans
from src.config.config import DEFAULT_EVENT_DESCRIPTIONS, DEFAULT_EVENT_LABELS, AnalysisConfig
from src.config.constants import MEASUREMENT_COLUMNS
from src.config.models import BusData
from src.plotter.dataset_plots import plot_dataset_correlation_heatmap, plot_missing_raster
from src.utils.serialization import save_dataframe_csv, save_json



def export_dataset_level(
    buses: list[BusData],
    config: AnalysisConfig,
    roots: dict[str, Path],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    label_map = {int(k): v for k, v in DEFAULT_EVENT_LABELS.items()}

    dataset_spans = build_global_event_spans(buses, label_map)
    rankings = cross_bus_rankings(buses, dataset_spans)
    correlations = cross_bus_correlations(buses)
    dataset_integrity = build_dataset_integrity(buses, config)

    save_json(roots["dataset"] / "json" / "dataset_integrity.json", dataset_integrity)
    save_json(roots["dataset"] / "json" / "cross_bus_profile.json", {"rankings": rankings, "correlations": correlations})
    save_json(roots["dataset"] / "json" / "event_library.json", summarize_event_spans(dataset_spans))
    save_json(roots["dataset"] / "json" / "event_instances_reference_bus.json", dataset_spans)

    alignment_rows: list[dict[str, Any]] = []
    for bus_id, payload in dataset_integrity["alignment"]["per_bus_alignment"].items():
        row = {"bus_id": bus_id}
        row.update(payload)
        alignment_rows.append(row)
    save_dataframe_csv(roots["dataset"] / "csv" / "alignment_summary.csv", pd.DataFrame(alignment_rows))

    dataset_md = [
        "# Dataset summary",
        "",
        f"Input directory: `{config.input_dir}`",
        "",
        f"Bus count: **{len(buses)}**",
        "",
        f"Buses: {', '.join(bus.bus_id for bus in buses)}",
        "",
        "## Event dictionary",
        "",
    ]
    for event_id, label in DEFAULT_EVENT_LABELS.items():
        dataset_md.append(f"- `{int(event_id)}` → **{label}**: {DEFAULT_EVENT_DESCRIPTIONS.get(int(event_id), '')}")
    (roots["dataset"] / "reports" / "dataset_summary.md").write_text("\n".join(dataset_md) + "\n", encoding="utf-8")

    for signal, corr_dict in correlations.items():
        frame = pd.DataFrame(corr_dict)
        if not frame.empty:
            plot_dataset_correlation_heatmap(
                corr_df=frame,
                title=f"Cross-bus correlation | {signal}",
                out_path=roots["dataset"] / "plots" / f"correlation_{signal}.png",
                dpi=config.plots_dpi,
            )

    missing_rows = []
    time_ref = first_bus.df["TIMESTAMP"].to_numpy(dtype=float)
    for bus in buses:
        mask = bus.df[MEASUREMENT_COLUMNS].isna().any(axis=1).to_numpy(dtype=int)
        missing_rows.append(mask)
    missing_matrix = np.vstack(missing_rows) if missing_rows else np.empty((0, 0))
    if missing_matrix.size:
        plot_missing_raster(
            time=time_ref,
            missing_matrix=missing_matrix,
            bus_labels=[bus.bus_id for bus in buses],
            out_path=roots["dataset"] / "plots" / "missing_data_raster.png",
            dpi=config.plots_dpi,
        )

    return dataset_spans, correlations, rankings


