from pathlib import Path
import textwrap
from typing import Any

import numpy as np

from src.analysis.cross_bus import cross_bus_correlations, cross_bus_rankings
from src.analysis.dataset_integrity import build_dataset_integrity
from src.analysis.events import event_spans, summarize_event_spans
from src.analysis.hilbert_analysis import hilbert_features
from src.analysis.power import compute_three_phase_power
from src.analysis.scope_analysis import analyze_scope
from src.config.config import DEFAULT_EVENT_DESCRIPTIONS, DEFAULT_EVENT_LABELS, AnalysisConfig
from src.config.constants import MEASUREMENT_COLUMNS
from src.config.models import BusData
from src.io.output_layout import build_bus_dirs
from src.plotter.bus_plots import plot_bus_distribution_panels, plot_bus_hilbert_features, plot_bus_overview, plot_bus_power, plot_bus_signal_window
from src.plotter.dataset_plots import plot_dataset_correlation_heatmap, plot_missing_raster
from src.utils.serialization import save_dataframe_csv, save_json



def export_per_bus(buses: list[BusData], config: AnalysisConfig) -> dict[str, Any]:
    per_bus_full_json: dict[str, Any] = {}

    for bus in buses:
        dirs = build_bus_dirs(config, bus.bus_id)
        full_df = bus.df.copy()
        full_result = analyze_scope(full_df, f"{bus.bus_id}_full", config)
        per_bus_full_json[bus.bus_id] = full_result

        save_dataframe_csv(dirs["csv"] / f"{bus.bus_id}_full.csv", full_df)
        save_dataframe_csv(dirs["csv"] / f"{bus.bus_id}_power_full.csv", compute_three_phase_power(full_df))
        if config.save_per_bus_json:
            save_json(dirs["json"] / "bus_full_analysis.json", full_result)

        plot_bus_overview(bus, dirs["plots"], dpi=config.plots_dpi)
        plot_bus_distribution_panels(bus, dirs["plots"], dpi=config.plots_dpi)
        plot_bus_power(
            bus_df=compute_three_phase_power(full_df),
            bus_id=bus.bus_id,
            out_dir=dirs["plots"],
            dpi=config.plots_dpi,
        )

        time = full_df["TIMESTAMP"].to_numpy(dtype=float)
        for col in MEASUREMENT_COLUMNS:
            plot_bus_signal_window(
                bus=bus,
                column=col,
                out_path=dirs["plots"] / f"{bus.bus_id}_{col}_full.png",
                dpi=config.plots_dpi,
            )

            hf = hilbert_features(full_df[col].to_numpy(dtype=float))
            envelope = hf.get("envelope_series")
            phase = hf.get("phase_series_rad")
            if envelope is not None and phase is not None:
                plot_bus_hilbert_features(
                    time=time,
                    envelope=np.asarray(envelope, dtype=float),
                    phase_rad=np.asarray(phase, dtype=float),
                    title_prefix=f"{bus.bus_id} | {col}",
                    out_path=dirs["plots"] / f"{bus.bus_id}_{col}_hilbert.png",
                    dpi=config.plots_dpi,
                )

        md = [
            f"# {bus.bus_id} report",
            "",
            f"Rows: **{len(full_df)}**",
            "",
            f"Sampling rate estimate: **{bus.sampling_rate_hz:.6f} Hz**" if np.isfinite(bus.sampling_rate_hz) else "Sampling rate estimate unavailable.",
            "",
            "See JSON and CSV outputs for the complete signal, noise, spectral, Hilbert and power summaries.",
        ]
        (dirs["reports"] / f"{bus.bus_id}_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    return per_bus_full_json


