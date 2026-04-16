#!/usr/bin/env python3
from __future__ import annotations

from typing import Any

import pandas as pd

from src.analysis.power import compute_three_phase_power
from src.analysis.scope_analysis import analyze_scope
from src.config.config import AnalysisConfig
from src.config.models import BusData
from src.io.output_layout import build_event_dirs
from src.plotter.event_plots import (
    plot_event_bus_window,
    plot_event_overlay_frequency,
    plot_event_overlay_voltage,
)
from src.utils.serialization import save_dataframe_csv, save_json


def export_events(
    buses: list[BusData],
    dataset_spans: list[dict[str, Any]],
    config: AnalysisConfig,
) -> None:
    """
    Export event-centered artifacts.

    Output structure per event:
        events/
          event_<idx>_<label>/
            general/
              csv/
              plots/
              reports/
              json/
            buses/
              <bus_id>/
                csv/
                plots/
                reports/
                json/

    For each event this function:
    - saves general metadata and overlay plots
    - saves per-bus event window/context CSVs
    - saves per-bus event analysis JSON
    - saves per-bus plots and markdown report
    - saves consolidated all-buses CSVs for the event
    - saves event summary JSON/MD and event bus index CSV
    """
    if not buses:
        return

    if not dataset_spans:
        return

    for idx, sp in enumerate(dataset_spans, start=1):
        label = str(sp.get("label", f"event_{idx}"))
        event_id = int(sp.get("event_id", -1))
        start_time = float(sp.get("start_time", 0.0))
        end_time = float(sp.get("end_time", start_time))
        duration_s = float(sp.get("duration_s", max(0.0, end_time - start_time)))

        context_start = start_time - float(config.event_context_seconds)
        context_end = end_time + float(config.event_context_seconds)

        general_dirs = build_event_dirs(
            config=config,
            event_index=idx,
            label=label,
            bus_id=None,
        )

        # ------------------------------------------------------------------
        # General event-level outputs
        # ------------------------------------------------------------------
        event_metadata = {
            **sp,
            "event_index": idx,
            "event_id": event_id,
            "label": label,
            "start_time": start_time,
            "end_time": end_time,
            "duration_s": duration_s,
            "context_start": context_start,
            "context_end": context_end,
        }
        save_json(general_dirs["json"] / "event_metadata.json", event_metadata)

        plot_event_overlay_frequency(
            buses=buses,
            event_span=event_metadata,
            out_path=general_dirs["plots"] / "overlay_frequency.png",
            context_seconds=config.event_context_seconds,
            dpi=config.plots_zoom_dpi,
        )
        plot_event_overlay_voltage(
            buses=buses,
            event_span=event_metadata,
            out_path=general_dirs["plots"] / "overlay_va_mag.png",
            context_seconds=config.event_context_seconds,
            dpi=config.plots_zoom_dpi,
        )

        event_rows: list[dict[str, Any]] = []
        all_event_bus_frames: list[pd.DataFrame] = []
        all_context_bus_frames: list[pd.DataFrame] = []
        all_power_bus_frames: list[pd.DataFrame] = []

        # ------------------------------------------------------------------
        # Per-bus outputs inside this event
        # ------------------------------------------------------------------
        for bus in buses:
            bus_dirs = build_event_dirs(
                config=config,
                event_index=idx,
                label=label,
                bus_id=bus.bus_id,
            )

            df = bus.df

            mask_event = (df["TIMESTAMP"] >= start_time) & (df["TIMESTAMP"] <= end_time)
            mask_context = (df["TIMESTAMP"] >= context_start) & (df["TIMESTAMP"] <= context_end)

            event_df = df.loc[mask_event].copy()
            context_df = df.loc[mask_context].copy()

            if not event_df.empty:
                all_event_bus_frames.append(event_df.assign(bus_id=bus.bus_id))

            if not context_df.empty:
                all_context_bus_frames.append(context_df.assign(bus_id=bus.bus_id))

            event_result = analyze_scope(
                df=event_df,
                scope_name=f"{bus.bus_id}_event_{idx}",
                config=config,
            )

            # --------------------------------------------------------------
            # CSV exports
            # --------------------------------------------------------------
            if config.save_event_bus_csvs:
                save_dataframe_csv(
                    bus_dirs["csv"] / f"{bus.bus_id}_event_window.csv",
                    event_df,
                )
                save_dataframe_csv(
                    bus_dirs["csv"] / f"{bus.bus_id}_event_context.csv",
                    context_df,
                )

                if not event_df.empty:
                    power_df = compute_three_phase_power(event_df)
                    save_dataframe_csv(
                        bus_dirs["csv"] / f"{bus.bus_id}_event_power.csv",
                        power_df,
                    )
                    all_power_bus_frames.append(power_df.assign(bus_id=bus.bus_id))

            # --------------------------------------------------------------
            # JSON exports
            # --------------------------------------------------------------
            if config.save_per_bus_json:
                save_json(
                    bus_dirs["json"] / f"{bus.bus_id}_event_analysis.json",
                    event_result,
                )

            # --------------------------------------------------------------
            # Plots
            # --------------------------------------------------------------
            if not context_df.empty:
                plot_event_bus_window(
                    bus=BusData(
                        bus_id=bus.bus_id,
                        path=bus.path,
                        df=context_df,
                        sampling_rate_hz=bus.sampling_rate_hz,
                    ),
                    event_span=event_metadata,
                    out_dir=bus_dirs["plots"],
                    context_seconds=0.0,
                    dpi=config.plots_zoom_dpi,
                )

            # --------------------------------------------------------------
            # Per-bus report
            # --------------------------------------------------------------
            md = [
                f"# {bus.bus_id} | {label} | event {idx}",
                "",
                f"Event id: `{event_id}`",
                "",
                f"Exact event window: `{start_time:.3f}` → `{end_time:.3f}` s",
                f"Context window: `{context_start:.3f}` → `{context_end:.3f}` s",
                "",
                f"Rows in exact event window: **{len(event_df)}**",
                f"Rows in context window: **{len(context_df)}**",
                "",
                "See CSV, JSON and plots for the complete per-signal event analysis.",
            ]
            (bus_dirs["reports"] / f"{bus.bus_id}_event_report.md").write_text(
                "\n".join(md) + "\n",
                encoding="utf-8",
            )

            event_rows.append(
                {
                    "bus_id": bus.bus_id,
                    "row_count_event": int(len(event_df)),
                    "row_count_context": int(len(context_df)),
                    "event_id": event_id,
                    "label": label,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration_s": duration_s,
                    "context_start": context_start,
                    "context_end": context_end,
                }
            )

        # ------------------------------------------------------------------
        # Consolidated general CSVs for this event
        # ------------------------------------------------------------------
        save_dataframe_csv(
            general_dirs["csv"] / "event_bus_index.csv",
            pd.DataFrame(event_rows),
        )

        if all_event_bus_frames:
            save_dataframe_csv(
                general_dirs["csv"] / "all_buses_event_window.csv",
                pd.concat(all_event_bus_frames, ignore_index=True),
            )

        if all_context_bus_frames:
            save_dataframe_csv(
                general_dirs["csv"] / "all_buses_event_context.csv",
                pd.concat(all_context_bus_frames, ignore_index=True),
            )

        if all_power_bus_frames:
            save_dataframe_csv(
                general_dirs["csv"] / "all_buses_event_power.csv",
                pd.concat(all_power_bus_frames, ignore_index=True),
            )

        # ------------------------------------------------------------------
        # General event summary
        # ------------------------------------------------------------------
        event_summary_payload = {
            "event_index": idx,
            "event_id": event_id,
            "label": label,
            "start_time": start_time,
            "end_time": end_time,
            "duration_s": duration_s,
            "context_start": context_start,
            "context_end": context_end,
            "bus_count": len(buses),
            "buses_processed": [bus.bus_id for bus in buses],
            "per_bus_rows": event_rows,
        }
        save_json(
            general_dirs["json"] / "event_summary.json",
            event_summary_payload,
        )

        summary_md = [
            f"# {label} | event {idx}",
            "",
            f"Event id: `{event_id}`",
            "",
            f"Window: `{start_time:.3f}` → `{end_time:.3f}` s",
            f"Duration: `{duration_s:.3f}` s",
            "",
            f"Context window: `{context_start:.3f}` → `{context_end:.3f}` s",
            "",
            f"Buses processed: **{len(buses)}**",
            "",
            "Artifacts generated:",
            "- overlay plots across buses",
            "- per-bus event window/context CSVs",
            "- per-bus analysis JSONs",
            "- per-bus event reports",
            "- consolidated all-buses event CSVs",
        ]
        (general_dirs["reports"] / "event_summary.md").write_text(
            "\n".join(summary_md) + "\n",
            encoding="utf-8",
        )