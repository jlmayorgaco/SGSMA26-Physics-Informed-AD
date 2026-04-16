from pathlib import Path
import textwrap
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.baseline import filter_normal_operation, normal_operation_summary
from src.analysis.cross_bus import cross_bus_correlations, cross_bus_rankings
from src.analysis.dataset_integrity import build_dataset_integrity
from src.analysis.events import event_spans, summarize_event_spans
from src.analysis.scope_analysis import analyze_scope
from src.config.config import DEFAULT_EVENT_DESCRIPTIONS, DEFAULT_EVENT_LABELS, AnalysisConfig
from src.config.constants import MEASUREMENT_COLUMNS
from src.config.enums import EventId
from src.config.models import BusData
from src.plotter.dataset_plots import plot_dataset_correlation_heatmap, plot_missing_raster
from src.plotter.general_plots import plot_general_histogram, plot_general_normal_operation_signal
from src.utils.serialization import save_dataframe_csv, save_json, to_builtin



def export_general_normal_operation(buses: list[BusData], config: AnalysisConfig, roots: dict[str, Path]) -> dict[str, Any]:
    per_bus_json: dict[str, Any] = {}
    normal_frames: list[pd.DataFrame] = []
    baseline_rows: list[dict[str, Any]] = []
    noise_rows: list[dict[str, Any]] = []
    power_rows: list[dict[str, Any]] = []

    for bus in buses:
        normal_df = filter_normal_operation(bus.df)
        normal_result = analyze_scope(normal_df, f"{bus.bus_id}_normal_operation", config)
        normal_result["selection_rule"] = {
            "Event": int(EventId.NORMAL_OPERATION),
            "DATA_PRESENT": 1,
            "finite_samples_only": True,
        }
        normal_result["summary"] = normal_operation_summary(bus.df)
        per_bus_json[bus.bus_id] = normal_result

        if not normal_df.empty:
            normal_frames.append(normal_df.assign(bus_id=bus.bus_id))

        for signal, payload in normal_result.get("channels", {}).items():
            baseline_rows.append({"bus_id": bus.bus_id, "signal": signal, **to_builtin(payload.get("basic", {}))})
            noise_rows.append({"bus_id": bus.bus_id, "signal": signal, **to_builtin(payload.get("noise", {}))})

        for signal, payload in normal_result.get("power_summary", {}).items():
            power_rows.append({"bus_id": bus.bus_id, "signal": signal, **to_builtin(payload)})

    general_json = {
        "selection_rule": {
            "Event": int(EventId.NORMAL_OPERATION),
            "DATA_PRESENT": 1,
            "finite_samples_only": True,
        },
        "per_bus": per_bus_json,
    }
    save_json(roots["general"] / "json" / "normal_operation_baseline.json", general_json)
    save_dataframe_csv(roots["general"] / "csv" / "per_bus_operating_baseline.csv", pd.DataFrame(baseline_rows))
    save_dataframe_csv(roots["general"] / "csv" / "per_bus_noise_summary.csv", pd.DataFrame(noise_rows))
    save_dataframe_csv(roots["general"] / "csv" / "per_bus_power_summary.csv", pd.DataFrame(power_rows))

    if normal_frames:
        all_normal = pd.concat(normal_frames, axis=0, ignore_index=True)
        save_dataframe_csv(roots["general"] / "csv" / "normal_operation_samples.csv", all_normal)

        for col in ["VA_mag", "IA_mag", "Frequency", "ROCOF"]:
            plot_general_normal_operation_signal(
                df=all_normal,
                column=col,
                out_path=roots["general"] / "plots" / f"normal_operation_signal_{col}.png",
                dpi=config.plots_dpi,
            )
            plot_general_histogram(
                df=all_normal,
                column=col,
                out_path=roots["general"] / "plots" / f"normal_operation_hist_{col}.png",
                dpi=config.plots_dpi,
            )

    md = [
        "# General normal-operation report",
        "",
        "Normal-operation analysis uses only samples with `Event == 0` and `DATA_PRESENT == 1`.",
        "",
        f"Buses analyzed: **{len(buses)}**",
        "",
        f"Signals analyzed: {', '.join(MEASUREMENT_COLUMNS)}",
        "",
        "See the CSV and JSON outputs for the full per-bus, per-signal structured baseline.",
    ]
    (roots["general"] / "reports" / "normal_operation_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    return per_bus_json

